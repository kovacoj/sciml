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
from state_layout import build_isothermal_layout  # noqa: E402
from unet.factory import build_model  # noqa: E402
from unet.boundary import BoundaryEnforcer  # noqa: E402


def load_dataset_samples(dataset_dir: str, mode: str, split: str = "train"):
    """Load topology samples from prepared dataset, respecting train/test split."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir

    # Load splits
    splits_path = dataset_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        valid_ids = set(splits.get(split, []))
    else:
        valid_ids = None  # load all

    samples = []
    for topo_dir in sorted(dataset_dir.iterdir()):
        if not topo_dir.is_dir() or not topo_dir.name.startswith("topology_"):
            continue
        if valid_ids is not None and topo_dir.name not in valid_ids:
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


def train_physics(model, samples, optimizer, device, steps, dataset_dir,
                  worker_count=4, topology_batch_size=4, log_every=50,
                  checkpoint_dir=None, save_every=25, eval_every=100,
                  start_step=0, prev_history=None):
    """Physics: HFDIB residual loss, no labels, mini-batch SGD.

    One network, one optimizer.  Each optimizer step evaluates a random
    batch of ``topology_batch_size`` topologies (not all 16).  Workers
    are transient: started for each batch, closed after evaluation, to
    bound RAM.  Gradient is averaged over the batch.

    Every ``eval_every`` steps, evaluates the full dataset (no gradient)
    and logs the mean loss for comparison.

    If ``checkpoint_dir`` is set, saves model+optimizer+history every
    ``save_every`` steps so training can resume after a crash.
    """
    model.train()

    n_train = len(samples)
    base_state_np = load_shared_base_state(dataset_dir)
    loss_config = load_loss_config(dataset_dir)

    from pinn.mesh_metadata import MeshMetadata
    ds_dir = Path(dataset_dir)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir
    mesh_meta = MeshMetadata.load(
        str(ds_dir / "shared" / "mesh_metadata.npz"),
        str(ds_dir / "shared" / "mesh_metadata.json"),
    )
    layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)

    contexts = []
    for s in samples:
        case_dir = s["case_dir"]
        lam_tensor = torch.from_numpy(s["lambda"]).unsqueeze(0).unsqueeze(0).to(device)

        flux_asm = FluxAssembler(
            owners=mesh_meta.owners,
            neighbours=mesh_meta.neighbours,
            sf_vec=mesh_meta.face_area_vectors,
            owner_weights=mesh_meta.owner_weights,
            n_cells=mesh_meta.n_cells,
            n_faces=mesh_meta.n_faces,
        )

        # Determine trainable phi faces: internal + outlet patches
        patch_names = list(mesh_meta.patch_names)
        n_internal = mesh_meta.n_internal_faces
        phi_trainable = np.zeros(mesh_meta.n_faces, dtype=bool)
        phi_trainable[:n_internal] = True
        for pname in ["outletLower", "outletUpper"]:
            if pname in patch_names:
                idx = patch_names.index(pname)
                start = int(mesh_meta.patch_start_faces[idx])
                count = int(mesh_meta.patch_face_counts[idx])
                phi_trainable[start:start + count] = True
        phi_trainable_indices = np.flatnonzero(phi_trainable)

        base_state = torch.from_numpy(base_state_np).to(device)
        from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
        state_asm = IndependentPhiStateAssembler(
            base_state, layout, flux_asm,
            mesh_meta.n_cells, mesh_meta.n_faces,
            phi_trainable_indices)

        contexts.append({
            "topology_id": s["topology_id"],
            "lam": lam_tensor,
            "case_dir": case_dir,
            "state_asm": state_asm,
        })

    from multitopology.context import PreparedTopology
    from multitopology.worker_pool import TopologyWorkerPool

    prepared = []
    for ctx in contexts:
        prepared.append(PreparedTopology(
            topology_id=ctx["topology_id"],
            case_dir=ctx["case_dir"],
            features=ctx["lam"].squeeze(0),
            warm_state=torch.from_numpy(base_state_np),
            loss_config=loss_config,
            state_assembler=ctx["state_asm"],
            u_ids=layout.indices("U"),
            p_ids=layout.indices("p"),
            phi_ids=layout.indices("phi"),
            inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"],
        ))

    pool = TopologyWorkerPool(prepared, max_concurrent=worker_count)

    n_passes = (steps * topology_batch_size + n_train - 1) // n_train
    print(f"[phys] {n_train} topologies, batch_size={topology_batch_size}, "
          f"workers={worker_count}, steps={steps}, "
          f"~{n_passes} dataset passes", flush=True)

    history = list(prev_history) if prev_history else []
    t0 = time.time()

    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)

        batch_indices = torch.randperm(n_train)[:topology_batch_size].tolist()
        batch_ctxs = [contexts[i] for i in batch_indices]
        batch_tids = [c["topology_id"] for c in batch_ctxs]
        batch_prepared = [prepared[i] for i in batch_indices]

        pool.topologies = {p.topology_id: p for p in batch_prepared}
        pool.start_wave(batch_tids)

        try:
            batch_states = []
            batch_cell_preds = []
            batch_phi_preds = []
            for ctx in batch_ctxs:
                cell_pred, phi_pred = model(ctx["lam"])
                cell_pred = project_solid_velocity(cell_pred, ctx["lam"])
                corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
                phi_corr = phi_pred.squeeze(0)
                state = ctx["state_asm"].assemble(corrections, phi_corr)
                batch_states.append(state)
                batch_cell_preds.append(cell_pred.detach())
                batch_phi_preds.append(phi_pred.detach())

            batch_results = pool.evaluate_wave(
                batch_tids,
                [s.detach().to(torch.float64).cpu().numpy().copy()
                 for s in batch_states],
            )

            for state, result in zip(batch_states, batch_results):
                gs = torch.from_numpy(result["grad_state"]).to(
                    device=device, dtype=state.dtype)
                state.backward(gs / topology_batch_size)

            batch_losses = [(r["loss"], r["loss_u"], r["loss_p"], r["loss_phi"])
                            for r in batch_results]
        finally:
            pool.close_wave()

        optimizer.step()

        mean_loss = np.mean([l[0] for l in batch_losses])
        mean_u = np.mean([l[1] for l in batch_losses])
        mean_p = np.mean([l[2] for l in batch_losses])
        mean_phi = np.mean([l[3] for l in batch_losses])

        elapsed = time.time() - t0
        n_evals = (step - start_step + 1) * topology_batch_size

        if step % log_every == 0 or step == steps - 1:
            # Compute RMS of dimensionless network outputs
            rms_du = float(torch.sqrt(torch.mean(
                torch.stack([p[:, :2] for p in batch_cell_preds])**2)))
            rms_dp = float(torch.sqrt(torch.mean(
                torch.stack([p[:, 2] for p in batch_cell_preds])**2)))
            rms_dphi = float(torch.sqrt(torch.mean(
                torch.stack([p for p in batch_phi_preds])**2)))

            print(f"[phys] s{step:4d} loss={mean_loss:.4e} "
                  f"U={mean_u:.2e} p={mean_p:.2e} "
                  f"phi={mean_phi:.2e} "
                  f"evals={n_evals} t={elapsed:.0f}s",
                  flush=True)
            print(f"        RMS q_U={rms_du:.2e} q_p={rms_dp:.2e} "
                  f"q_phi={rms_dphi:.2e} "
                  f"phys dU={0.1*rms_du:.2e} dp={0.01*rms_dp:.2e} "
                  f"dphi={4e-7*rms_dphi:.2e}",
                  flush=True)
            history.append({
                "step": step,
                "loss": mean_loss,
                "loss_u": mean_u,
                "loss_p": mean_p,
                "loss_phi": mean_phi,
                "n_evals": n_evals,
                "elapsed_s": elapsed,
            })

        if checkpoint_dir is not None and (step + 1) % save_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_s{step + 1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": "simple",
                "model_kwargs": {},
                "mode": "physics",
                "step": step + 1,
                "history": history,
            }, ckpt_path)
            print(f"[phys] checkpoint: {ckpt_path.name} "
                  f"(step {step + 1}, t={elapsed:.0f}s)", flush=True)
            write_json(Path(checkpoint_dir) / "history.json", history)

        if eval_every > 0 and (step + 1) % eval_every == 0:
            full_loss = _eval_full_dataset(
                model, contexts, prepared, pool, worker_count, device)
            print(f"[phys] FULL s{step:4d} loss={full_loss:.4e}", flush=True)
            history.append({
                "step": step,
                "full_loss": full_loss,
                "elapsed_s": elapsed,
            })

    return history


def _eval_full_dataset(model, contexts, prepared, pool, worker_count, device):
    """Evaluate mean loss on all training topologies (no gradient)."""
    import numpy as np
    n = len(contexts)
    n_batches = (n + worker_count - 1) // worker_count
    all_losses = []

    model.eval()
    with torch.no_grad():
        for bi in range(n_batches):
            start = bi * worker_count
            end = min(start + worker_count, n)
            batch_ctxs = contexts[start:end]
            batch_tids = [c["topology_id"] for c in batch_ctxs]
            batch_prepared = prepared[start:end]

            pool.topologies = {p.topology_id: p for p in batch_prepared}
            pool.start_wave(batch_tids)

            try:
                batch_states = []
                for ctx in batch_ctxs:
                    cell_pred, phi_pred = model(ctx["lam"])
                    cell_pred = project_solid_velocity(cell_pred, ctx["lam"])
                    corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
                    phi_corr = phi_pred.squeeze(0)
                    state = ctx["state_asm"].assemble(corrections, phi_corr)
                    batch_states.append(state)

                batch_results = pool.evaluate_wave(
                    batch_tids,
                    [s.detach().to(torch.float64).cpu().numpy().copy()
                     for s in batch_states],
                )
                all_losses.extend([r["loss"] for r in batch_results])
            finally:
                pool.close_wave()

    model.train()
    return float(np.mean(all_losses))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["supervised", "physics"])
    ap.add_argument("--architecture", default=None, choices=["simple", "unet"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=20,
                    help="Supervised: number of epochs")
    ap.add_argument("--steps", type=int, default=1000,
                    help="Physics: number of optimizer steps")
    ap.add_argument("--topology-batch-size", type=int, default=4,
                    help="Physics: topologies per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=100,
                    help="Evaluate full dataset loss every N steps")
    args = ap.parse_args()

    if args.device != "cpu":
        raise ValueError("Currently CPU-only")

    if args.architecture is None:
        args.architecture = "unet" if args.mode == "supervised" else "simple"

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    samples = load_dataset_samples(args.dataset, args.mode, args.split)
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

    start_step = 0
    prev_history = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", ckpt.get("epoch", 0))
        prev_history = ckpt.get("history", [])
        print(f"[train] resumed from {args.resume} at step {start_step}")

    if args.mode == "supervised":
        history = train_supervised(model, samples, optimizer, args.device, args.epochs)
    else:
        history = train_physics(
            model, samples, optimizer, args.device, args.steps,
            args.dataset, args.workers,
            topology_batch_size=args.topology_batch_size,
            checkpoint_dir=str(output_dir),
            save_every=args.save_every,
            eval_every=args.eval_every,
            start_step=start_step,
            prev_history=prev_history)

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
        "steps": args.steps if args.mode == "physics" else args.epochs,
        "topology_batch_size": args.topology_batch_size if args.mode == "physics" else None,
        "lr": args.lr,
        "n_params": n_params,
        "uses_flow_labels": args.mode == "supervised",
    })

    print(f"[train] done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
