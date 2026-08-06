"""Train a neural network on the four-port topology dataset.

Modes:
  --mode supervised: MSE + TV against converged HFDIB fields (uses FlowUNet)
  --mode physics:    HFDIB residual loss, no labels (uses SimpleFlowNet)

Usage:
  python -m unet.train --mode supervised --architecture unet --dataset datasets/four_port_64 --epochs 20
  python -m unet.train --mode physics --architecture simple --dataset datasets/four_port_64 --epochs 20 --workers 4
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
from state_layout import build_state_layout  # noqa: E402
from unet.factory import build_model  # noqa: E402
from unet.boundary import BoundaryEnforcer  # noqa: E402


def load_dataset_samples(dataset_dir: str, mode: str):
    """Load topology samples from prepared dataset."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir

    samples = []
    for topo_dir in sorted(dataset_dir.iterdir()):
        if not topo_dir.is_dir() or not topo_dir.name.startswith("topology_"):
            continue
        lam = np.load(topo_dir / "lambda.npy")
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


def load_shared_base_state(dataset_dir: str):
    """Load the shared k=0 base state."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir
    base_path = dataset_dir / "shared" / "base_state_k0.npy"
    if not base_path.exists():
        raise FileNotFoundError(f"Shared base state not found: {base_path}")
    return np.load(base_path)


def load_loss_config(dataset_dir: str):
    """Load the shared physics loss config."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir
    cfg_path = dataset_dir / "shared" / "physics_loss_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Physics loss config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    return ResidualLossConfig(cfg["gamma_u"], cfg["gamma_p"], cfg["gamma_phi"])


def tv_loss(pred: torch.Tensor) -> torch.Tensor:
    dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return dx.abs().mean() + dy.abs().mean()


def project_solid_velocity(pred: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """Zero velocity corrections inside solid cells (lam > 0.5)."""
    fluid = (lam[:, 0] < 0.5).to(dtype=pred.dtype).unsqueeze(1)
    return pred * torch.cat([fluid, fluid, torch.ones_like(fluid)], dim=1)


def train_supervised(model, samples, optimizer, device, epochs, tv_beta=0.1):
    """Supervised: MSE + TV against converged HFDIB fields."""
    model.train()
    lams = torch.stack([torch.from_numpy(s["lambda"]).unsqueeze(0) for s in samples]).to(device)
    targets = torch.stack([torch.from_numpy(s["target"]) for s in samples]).to(device)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(lams)
        mse = nn.functional.mse_loss(pred, targets)
        tv = tv_loss(pred)
        loss = mse + tv_beta * tv
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[sup] e{epoch:3d} mse={mse.item():.4e} tv={tv.item():.4e} total={loss.item():.4e}")
            history.append({"epoch": epoch, "mse": mse.item(), "total": loss.item()})
    return history


def train_physics(model, samples, optimizer, device, epochs, dataset_dir, worker_count=4):
    """Physics: HFDIB residual loss, no labels, persistent workers."""
    model.train()

    base_state_np = load_shared_base_state(dataset_dir)
    loss_config = load_loss_config(dataset_dir)

    # Build per-topology context
    contexts = []
    for s in samples:
        case_dir = s["case_dir"]
        lam_tensor = torch.from_numpy(s["lambda"]).unsqueeze(0).unsqueeze(0).to(device)

        # Build mesh metadata from the case
        from pinn.mesh_metadata import build_from_polymesh
        mesh_meta = build_from_polymesh(case_dir)
        layout = build_state_layout("isothermal")

        flux_asm = FluxAssembler(
            owners=mesh_meta.owners,
            neighbours=mesh_meta.neighbours,
            sf_vec=mesh_meta.face_area_vectors,
            owner_weights=mesh_meta.owner_weights,
            n_cells=mesh_meta.n_cells,
            n_faces=mesh_meta.n_faces,
        )

        base_state = torch.from_numpy(base_state_np).to(device)
        state_asm = StateAssembler(base_state, layout, flux_asm,
                                    mesh_meta.n_cells, mesh_meta.n_faces)

        contexts.append({
            "topology_id": s["topology_id"],
            "lam": lam_tensor,
            "case_dir": case_dir,
            "state_asm": state_asm,
        })

    # Start worker pool
    from multitopology.context import PreparedTopology
    from multitopology.worker_pool import TopologyWorkerPool

    # Build PreparedTopology objects for the pool
    prepared = []
    for ctx in contexts:
        layout = build_state_layout("isothermal")
        prepared.append(PreparedTopology(
            topology_id=ctx["topology_id"],
            case_dir=ctx["case_dir"],
            features=ctx["lam"].squeeze(0),  # not used by pool
            warm_state=torch.from_numpy(base_state_np),
            loss_config=loss_config,
            state_assembler=ctx["state_asm"],
            u_ids=layout.indices("U"),
            p_ids=layout.indices("p"),
            phi_ids=layout.indices("phi"),
        ))

    pool = TopologyWorkerPool(prepared)
    topo_ids = [c["topology_id"] for c in contexts]
    pool.start(topo_ids)

    history = []
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            states = []
            for ctx in contexts:
                pred = model(ctx["lam"])
                pred = project_solid_velocity(pred, ctx["lam"])
                corrections = pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
                state = ctx["state_asm"].assemble(corrections)
                states.append(state)

            results = pool.evaluate(
                topo_ids,
                [s.detach().to(torch.float64).cpu().numpy().copy() for s in states],
            )

            grad_states = []
            for state, result in zip(states, results):
                gs = torch.from_numpy(result["grad_state"]).to(device, dtype=state.dtype)
                grad_states.append(gs / len(states))

            torch.autograd.backward(states, grad_states)
            optimizer.step()

            mean_loss = np.mean([r["loss"] for r in results])
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f"[phys] e{epoch:3d} loss={mean_loss:.4e} "
                      f"U={np.mean([r['loss_u'] for r in results]):.2e} "
                      f"p={np.mean([r['loss_p'] for r in results]):.2e} "
                      f"phi={np.mean([r['loss_phi'] for r in results]):.2e}")
                history.append({"epoch": epoch, "loss": mean_loss})
    finally:
        pool.close()

    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["supervised", "physics"])
    ap.add_argument("--architecture", default=None, choices=["simple", "unet"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.device != "cpu":
        raise ValueError("Currently CPU-only")

    if args.architecture is None:
        args.architecture = "unet" if args.mode == "supervised" else "simple"

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    samples = load_dataset_samples(args.dataset, args.mode)
    print(f"[train] {len(samples)} samples, mode={args.mode}, arch={args.architecture}")

    model_kwargs = {}
    if args.architecture == "simple":
        model = build_model("simple", **model_kwargs)
    else:
        model = build_model("unet", **model_kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] {args.architecture}: {n_params} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output) if args.output else \
        Path(PROJECT_ROOT) / "outputs" / f"unet_{args.mode}_{args.architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "supervised":
        history = train_supervised(model, samples, optimizer, args.device, args.epochs)
    else:
        history = train_physics(model, samples, optimizer, args.device, args.epochs,
                                 args.dataset, args.workers)

    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": args.architecture,
        "model_kwargs": model_kwargs,
        "mode": args.mode,
    }, output_dir / "checkpoint.pt")

    write_json(output_dir / "history.json", history)
    write_json(output_dir / "config.json", {
        "mode": args.mode,
        "architecture": args.architecture,
        "epochs": args.epochs,
        "lr": args.lr,
        "n_params": n_params,
        "uses_flow_labels": args.mode == "supervised",
    })

    print(f"[train] done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
