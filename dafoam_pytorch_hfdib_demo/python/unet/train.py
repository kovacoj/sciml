"""Train a U-Net on the four-port topology dataset.

Two modes:
  --mode supervised: MSE loss against converged HFDIB fields + TV
  --mode physics:    DAFoam HFDIB residual loss (no labels)

Usage:
  python -m unet.train --mode supervised --dataset datasets/four_port_64 --epochs 100
  python -m unet.train --mode physics --dataset datasets/four_port_64 --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from dafoam_residual_function import dafoam_residual  # noqa: E402
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.state_assembly import StateAssembler  # noqa: E402
from pinn.losses import ResidualLossConfig, weighted_residual_loss_torch, weighted_residual_loss_numpy  # noqa: E402
from pinn.mesh_metadata import build_from_polymesh  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from unet.models import FlowUNet  # noqa: E402
from unet.boundary import BoundaryEnforcer  # noqa: E402


def load_dataset_samples(dataset_dir: str, mode: str):
    """Load all topology samples."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir

    samples = []
    for topo_dir in sorted(dataset_dir.iterdir()):
        if not topo_dir.is_dir() or not topo_dir.name.startswith("topology_"):
            continue

        lam = np.load(topo_dir / "lambda.npy")  # [64, 64]
        sample = {
            "topology_id": topo_dir.name,
            "lambda": lam,
            "case_dir": str(topo_dir / "case"),
        }

        if mode == "supervised":
            sample["ux"] = np.load(topo_dir / "ux_hfdib.npy")
            sample["uy"] = np.load(topo_dir / "uy_hfdib.npy")
            sample["p"] = np.load(topo_dir / "pressure_hfdib.npy")
            sample["target"] = np.stack([sample["ux"], sample["uy"], sample["p"]], axis=0)

        samples.append(sample)

    return samples


def tv_loss(pred: torch.Tensor) -> torch.Tensor:
    """Total variation loss."""
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def train_supervised(model, samples, optimizer, device, epochs, tv_beta=0.1):
    """Supervised training: MSE + TV against converged HFDIB fields."""
    model.train()
    bc = BoundaryEnforcer(64, 64).to(device)

    # Prepare tensors
    lams = torch.stack([
        torch.from_numpy(s["lambda"]).unsqueeze(0) for s in samples
    ]).to(device)
    targets = torch.stack([
        torch.from_numpy(s["target"]) for s in samples
    ]).to(device)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(lams)  # [B, 3, 64, 64]
        pred = bc.apply(pred, lams)

        mse = nn.functional.mse_loss(pred, targets)
        tv = tv_loss(pred)
        loss = mse + tv_beta * tv

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[sup] epoch {epoch:4d} mse={mse.item():.6e} tv={tv.item():.6e} "
                  f"total={loss.item():.6e}")
            history.append({"epoch": epoch, "mse": mse.item(), "tv": tv.item(),
                           "total": loss.item()})

    return history


def train_physics(model, samples, optimizer, device, epochs, dataset_dir):
    """Physics training: DAFoam HFDIB residual loss (no labels)."""
    model.train()
    bc = BoundaryEnforcer(64, 64).to(device)

    # Build per-topology context
    topo_contexts = []
    for s in samples:
        case_dir = s["case_dir"]
        lam = torch.from_numpy(s["lambda"]).unsqueeze(0).unsqueeze(0).to(device)

        # Load mesh metadata
        mesh_meta = build_from_polymesh(case_dir)
        layout = build_state_layout("isothermal")

        # Build flux assembler
        flux_asm = FluxAssembler(
            owners=mesh_meta.owners,
            neighbours=mesh_meta.neighbours,
            sf_vec=mesh_meta.face_area_vectors,
            owner_weights=mesh_meta.owner_weights,
            n_cells=mesh_meta.n_cells,
            n_faces=mesh_meta.n_faces,
        )

        # For physics mode: base state is zeros (no warm start)
        # BC enforcement provides the inlet/wall conditions
        n_state = len(layout.indices("U")) + len(layout.indices("p")) + len(layout.indices("phi"))
        base_state = torch.zeros(n_state, dtype=torch.float64)

        state_asm = StateAssembler(base_state, layout, flux_asm,
                                   mesh_meta.n_cells, mesh_meta.n_faces)

        # Initialize bridge in subprocess to avoid conflicts
        # For now, run sequentially
        topo_contexts.append({
            "topology_id": s["topology_id"],
            "lam": lam,
            "case_dir": case_dir,
            "mesh_meta": mesh_meta,
            "layout": layout,
            "state_asm": state_asm,
            "bridge": None,  # will be created per-evaluation
        })

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()

        total_loss = 0.0
        for ctx in topo_contexts:
            # Forward pass
            pred = model(ctx["lam"])  # [1, 3, 64, 64]
            pred = bc.apply(pred, ctx["lam"])
            # pred is [1, 3, 64, 64] = (ux, uy, p)

            # Reshape to cell corrections [n_cells, 3]
            cell_corr = pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

            # Assemble state
            state = ctx["state_asm"].assemble(cell_corr)

            # Evaluate residual via DAFoam
            # Need to run in subprocess to avoid conflicts
            # For now, save state to file and call a worker
            state_np = state.detach().cpu().numpy().copy()

            # Save state and call residual worker
            state_path = f"/tmp/state_{ctx['topology_id']}.npy"
            np.save(state_path, state_np)

            result = subprocess.run(
                [sys.executable, "-m", "unet.residual_evaluator",
                 "--case", ctx["case_dir"],
                 "--state", state_path],
                capture_output=True, check=True,
                cwd=str(PYTHON_ROOT),
            )
            result_data = json.loads(result.stdout)

            loss_val = result_data["loss"]
            grad_state = np.load(f"/tmp/grad_{ctx['topology_id']}.npy")

            grad_t = torch.from_numpy(grad_state).to(device, dtype=cell_corr.dtype)
            # Backprop through state assembly
            state.backward(grad_t / len(topo_contexts))
            total_loss += loss_val

        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[phys] epoch {epoch:4d} loss={total_loss/len(topo_contexts):.6e}")
            history.append({"epoch": epoch, "loss": total_loss / len(topo_contexts)})

    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["supervised", "physics"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.device != "cpu":
        raise ValueError("Currently CPU-only")

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    samples = load_dataset_samples(args.dataset, args.mode)
    print(f"[train] {len(samples)} samples, mode={args.mode}")

    model = FlowUNet(in_channels=1, out_channels=3, base_filters=32, depth=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] FlowUNet: {n_params} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output) if args.output else Path(PROJECT_ROOT) / "outputs" / f"unet_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "supervised":
        history = train_supervised(model, samples, optimizer, args.device, args.epochs)
    else:
        history = train_physics(model, samples, optimizer, args.device, args.epochs,
                                 args.dataset)

    # Save
    torch.save({"model_state_dict": model.state_dict()}, output_dir / "checkpoint.pt")
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "config.json", {
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_params": n_params,
        "uses_flow_labels": args.mode == "supervised",
    })

    print(f"[train] done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
