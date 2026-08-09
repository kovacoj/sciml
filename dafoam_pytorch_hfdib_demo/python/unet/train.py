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


def train_fixed_point(model, samples, optimizer, device, steps, dataset_dir,
                      worker_count=4, topology_batch_size=4, log_every=10,
                      checkpoint_dir=None, save_every=25, eval_every=25,
                      alpha_u=0.1, alpha_p=0.1, alpha_phi=0.1,
                      start_step=0, prev_history=None):
    """Fixed-point training: use one SIMPLE step as a self-supervised target.

    No residual JTV needed. The loss is a nondimensional state-space MSE
    between the network's state and the (relaxed) SIMPLE-step output.

    L = mean[(dU/0.1)^2 + (dp/0.01)^2 + (dphi/4e-7)^2] / 3

    where dU, dp, dphi are differences between W_theta and the target.
    The target is W + alpha * (T(W) - W), where T is one SIMPLE step.
    """
    model.train()

    n_train = len(samples)
    base_state_np = load_shared_base_state(dataset_dir)

    U_SCALE = 0.1
    P_SCALE = 0.01
    PHI_SCALE = 4e-7

    from pinn.mesh_metadata import MeshMetadata
    ds_dir = Path(dataset_dir)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir
    mesh_meta = MeshMetadata.load(
        str(ds_dir / "shared" / "mesh_metadata.npz"),
        str(ds_dir / "shared" / "mesh_metadata.json"),
    )
    layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)

    n_cells = mesh_meta.n_cells
    n_faces = mesh_meta.n_faces
    n_internal = mesh_meta.n_internal_faces
    n_u = 3 * n_cells
    n_p = n_cells

    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")

    # Trainable phi indices
    patch_names = list(mesh_meta.patch_names)
    phi_trainable = np.zeros(n_faces, dtype=bool)
    phi_trainable[:n_internal] = True
    for pname in ["outletLower", "outletUpper"]:
        if pname in patch_names:
            idx = patch_names.index(pname)
            start = int(mesh_meta.patch_start_faces[idx])
            count = int(mesh_meta.patch_face_counts[idx])
            phi_trainable[start:start + count] = True
    phi_trainable_indices = np.flatnonzero(phi_trainable)
    phi_state_ids = n_u + n_p + phi_trainable_indices

    contexts = []
    for s in samples:
        case_dir = s["case_dir"]
        lam_tensor = torch.from_numpy(s["lambda"]).unsqueeze(0).unsqueeze(0).to(device)

        flux_asm = FluxAssembler(
            owners=mesh_meta.owners,
            neighbours=mesh_meta.neighbours,
            sf_vec=mesh_meta.face_area_vectors,
            owner_weights=mesh_meta.owner_weights,
            n_cells=n_cells,
            n_faces=n_faces,
        )

        base_state = torch.from_numpy(base_state_np).to(device)
        from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
        state_asm = IndependentPhiStateAssembler(
            base_state, layout, flux_asm, n_cells, n_faces,
            phi_trainable_indices)

        # Load HFDIB reference for diagnostic field errors
        ref_ux = np.load(Path(s["case_dir"]).parent / "ux_hfdib.npy")
        ref_uy = np.load(Path(s["case_dir"]).parent / "uy_hfdib.npy")
        ref_p = np.load(Path(s["case_dir"]).parent / "pressure_hfdib.npy")

        contexts.append({
            "topology_id": s["topology_id"],
            "lam": lam_tensor,
            "case_dir": case_dir,
            "state_asm": state_asm,
            "ref_ux": ref_ux,
            "ref_uy": ref_uy,
            "ref_p": ref_p,
        })

    from multitopology.context import PreparedTopology
    from multitopology.worker_pool import TopologyWorkerPool
    from pinn.losses import ResidualLossConfig

    loss_config = load_loss_config(dataset_dir)

    prepared = []
    for ctx in contexts:
        prepared.append(PreparedTopology(
            topology_id=ctx["topology_id"],
            case_dir=ctx["case_dir"],
            features=ctx["lam"].squeeze(0),
            warm_state=torch.from_numpy(base_state_np),
            loss_config=loss_config,
            state_assembler=ctx["state_asm"],
            u_ids=u_ids,
            p_ids=p_ids,
            phi_ids=phi_ids,
            inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"],
        ))

    pool = TopologyWorkerPool(prepared, max_concurrent=worker_count)

    print(f"[fp] {n_train} topologies, batch={topology_batch_size}, "
          f"workers={worker_count}, steps={steps}, "
          f"alpha_u={alpha_u}, alpha_p={alpha_p}, alpha_phi={alpha_phi}",
          flush=True)

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
            # Forward: predict states
            w_theta_list = []
            cell_preds = []
            phi_preds = []
            for ctx in batch_ctxs:
                cell_pred, phi_pred = model(ctx["lam"])
                cell_pred = project_solid_velocity(cell_pred, ctx["lam"])
                corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
                phi_corr = phi_pred.squeeze(0)
                w_theta = ctx["state_asm"].assemble(corrections, phi_corr)
                w_theta_list.append(w_theta)
                cell_preds.append(cell_pred.detach())
                phi_preds.append(phi_pred.detach())

            # SIMPLE step: get target states (detached, no grad)
            w_np_list = [w.detach().to(torch.float64).cpu().numpy().copy()
                         for w in w_theta_list]

            w_simple_list = pool.simple_step_wave(batch_tids, w_np_list)

            # Compute loss using network outputs directly (not assembled state)
            # The target corrections are derived from the SIMPLE step
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
            batch_losses = []

            for w_theta, w_simple_np, ctx, cell_pred, phi_pred in zip(
                    w_theta_list, w_simple_list, batch_ctxs,
                    cell_preds, phi_preds):

                w_simple = torch.from_numpy(w_simple_np).to(device, dtype=torch.float64)

                # Target: relaxed SIMPLE update
                w_theta_detached = w_theta.detach()

                target_ux = w_theta_detached[0:n_u:3] + alpha_u * (
                    w_simple[0:n_u:3] - w_theta_detached[0:n_u:3])
                target_uy = w_theta_detached[1:n_u:3] + alpha_u * (
                    w_simple[1:n_u:3] - w_theta_detached[1:n_u:3])
                target_p = w_theta_detached[n_u:n_u+n_p] + alpha_p * (
                    w_simple[n_u:n_u+n_p] - w_theta_detached[n_u:n_u+n_p])

                phi_start = n_u + n_p
                phi_idx_long = torch.from_numpy(phi_trainable_indices.astype(np.int64)).to(device)
                target_phi = w_theta_detached[phi_start:][phi_idx_long] + alpha_phi * (
                    w_simple[phi_start:][phi_idx_long] - w_theta_detached[phi_start:][phi_idx_long])

                pred_ux = w_theta[0:n_u:3]
                pred_uy = w_theta[1:n_u:3]
                pred_p = w_theta[n_u:n_u+n_p]
                pred_phi = w_theta[phi_start:][phi_idx_long]

                # Nondimensional loss
                du = (pred_ux - target_ux) / U_SCALE
                dv = (pred_uy - target_uy) / U_SCALE
                dp = (pred_p - target_p) / P_SCALE
                dphi = (pred_phi - target_phi) / PHI_SCALE

                loss_u = torch.mean(du**2 + dv**2)
                loss_p = torch.mean(dp**2)
                loss_phi = torch.mean(dphi**2)
                loss = (loss_u + loss_p + loss_phi) / 3.0

                total_loss = total_loss + loss / topology_batch_size
                batch_losses.append({
                    "loss": loss.item(),
                    "loss_u": loss_u.item(),
                    "loss_p": loss_p.item(),
                    "loss_phi": loss_phi.item(),
                })

            total_loss.backward()
            optimizer.step()
        finally:
            pool.close_wave()

        mean_loss = np.mean([b["loss"] for b in batch_losses])
        mean_u = np.mean([b["loss_u"] for b in batch_losses])
        mean_p = np.mean([b["loss_p"] for b in batch_losses])
        mean_phi = np.mean([b["loss_phi"] for b in batch_losses])

        elapsed = time.time() - t0
        n_evals = (step - start_step + 1) * topology_batch_size

        if step % log_every == 0 or step == steps - 1:
            print(f"[fp] s{step:4d} loss={mean_loss:.4e} "
                  f"U={mean_u:.2e} p={mean_p:.2e} "
                  f"phi={mean_phi:.2e} "
                  f"evals={n_evals} t={elapsed:.0f}s",
                  flush=True)
            history.append({
                "step": step,
                "loss": mean_loss,
                "loss_u": mean_u,
                "loss_p": mean_p,
                "loss_phi": mean_phi,
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
                "physics_objective": "fixed-point",
                "step": step + 1,
                "history": history,
            }, ckpt_path)
            print(f"[fp] checkpoint: {ckpt_path.name}", flush=True)
            write_json(Path(checkpoint_dir) / "history.json", history)

        # Field error diagnostic (uses stored HFDIB refs, no gradient)
        if eval_every > 0 and (step + 1) % eval_every == 0:
            rel_us = []
            rel_ps = []
            model.eval()
            with torch.no_grad():
                for ctx in contexts[:8]:  # subset for speed
                    cell_pred, phi_pred = model(ctx["lam"])
                    cell_pred = project_solid_velocity(cell_pred, ctx["lam"])
                    corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
                    phi_corr = phi_pred.squeeze(0)
                    w = ctx["state_asm"].assemble(corrections, phi_corr)

                    u_pred = w[:n_u].reshape(n_cells, 3).cpu().numpy()
                    p_pred = w[n_u:n_u+n_p].cpu().numpy()

                    ref_ux = ctx["ref_ux"]
                    ref_uy = ctx["ref_uy"]
                    ref_p = ctx["ref_p"]

                    rel_u = np.sqrt(np.mean((ref_ux - u_pred[:,0].reshape(64,64))**2 +
                                           (ref_uy - u_pred[:,1].reshape(64,64))**2)) / \
                            (np.sqrt(np.mean(ref_ux**2 + ref_uy**2)) + 1e-30)
                    rel_p_err = np.sqrt(np.mean((ref_p - p_pred.reshape(64,64))**2)) / \
                                (np.sqrt(np.mean(ref_p**2)) + 1e-30)
                    rel_us.append(rel_u)
                    rel_ps.append(rel_p_err)

            model.train()
            mean_rel_u = float(np.mean(rel_us))
            mean_rel_p = float(np.mean(rel_ps))
            print(f"[field] s{step:4d} train rel_U={mean_rel_u:.4e} "
                  f"rel_p={mean_rel_p:.4e}", flush=True)
            history.append({
                "step": step,
                "field_rel_u": mean_rel_u,
                "field_rel_p": mean_rel_p,
            })

    return history


def train_solver_distilled(model, samples, optimizer, device, steps, dataset_dir,
                           batch_size=4, log_every=10,
                           checkpoint_dir=None, save_every=25, eval_every=25,
                           start_step=0, prev_history=None):
    """Solver-distilled training: fixed K-step SIMPLE targets, pure PyTorch.

    No DAFoam in the training loop. Targets are precomputed truncated-SIMPLE
    states converted to network coordinates (q_cell, q_phi).

    Loss = (L_U/E_U + L_p/E_p + L_phi/E_phi) / 3
    where L_* are MSE between network output and target.
    """
    model.train()

    n_train = len(samples)
    ds_dir = Path(dataset_dir)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir

    targets_dir = ds_dir / "solver_targets"

    # Load normalization
    import json as _json
    norm_path = targets_dir / "target_normalization_k010.json"
    with open(norm_path) as f:
        norm = _json.load(f)
    E_u = norm["E_u"]
    E_p = norm["E_p"]
    E_phi = norm["E_phi"]

    print(f"[distill] E_u={E_u:.4e} E_p={E_p:.4e} E_phi={E_phi:.4e}")

    # Load targets for each sample
    from pinn.mesh_metadata import MeshMetadata
    mesh_meta = MeshMetadata.load(
        str(ds_dir / "shared" / "mesh_metadata.npz"),
        str(ds_dir / "shared" / "mesh_metadata.json"),
    )
    n_cells = mesh_meta.n_cells

    contexts = []
    for s in samples:
        tid = s["topology_id"]
        tdir = targets_dir / tid
        q_cell = np.load(tdir / "q_cell_k010.npy")  # [n_cells, 3]
        q_phi = np.load(tdir / "q_phi_k010.npy")    # [n_phi_trainable]

        lam_tensor = torch.from_numpy(s["lambda"]).unsqueeze(0).unsqueeze(0).to(device)
        target_cell = torch.from_numpy(q_cell).to(device)
        target_phi = torch.from_numpy(q_phi).to(device)

        # Load HFDIB refs for diagnostics
        ref_ux = np.load(Path(s["case_dir"]).parent / "ux_hfdib.npy")
        ref_uy = np.load(Path(s["case_dir"]).parent / "uy_hfdib.npy")
        ref_p = np.load(Path(s["case_dir"]).parent / "pressure_hfdib.npy")

        contexts.append({
            "topology_id": tid,
            "lam": lam_tensor,
            "target_cell": target_cell,
            "target_phi": target_phi,
            "ref_ux": ref_ux,
            "ref_uy": ref_uy,
            "ref_p": ref_p,
        })

    print(f"[distill] {n_train} samples, batch={batch_size}, "
          f"steps={steps}, pure PyTorch (no DAFoam)", flush=True)

    history = list(prev_history) if prev_history else []
    t0 = time.time()

    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)

        batch_indices = torch.randperm(n_train)[:batch_size].tolist()
        batch_ctxs = [contexts[i] for i in batch_indices]

        total_loss_u = 0.0
        total_loss_p = 0.0
        total_loss_phi = 0.0

        for ctx in batch_ctxs:
            cell_pred, phi_pred = model(ctx["lam"])
            cell_pred = project_solid_velocity(cell_pred, ctx["lam"])

            # Convert to [n_cells, 3]
            pred_cell = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            pred_phi = phi_pred.squeeze(0)

            # MSE per block
            loss_u = torch.mean((pred_cell[:, :2] - ctx["target_cell"][:, :2])**2)
            loss_p = torch.mean((pred_cell[:, 2] - ctx["target_cell"][:, 2])**2)
            loss_phi = torch.mean((pred_phi - ctx["target_phi"])**2)

            total_loss_u += loss_u / batch_size
            total_loss_p += loss_p / batch_size
            total_loss_phi += loss_phi / batch_size

        # Normalized loss
        loss = (total_loss_u / (E_u + 1e-30) +
                total_loss_p / (E_p + 1e-30) +
                total_loss_phi / (E_phi + 1e-30)) / 3.0

        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0

        if step % log_every == 0 or step == steps - 1:
            print(f"[distill] s{step:4d} loss={loss.item():.4e} "
                  f"U={total_loss_u.item():.2e} "
                  f"p={total_loss_p.item():.2e} "
                  f"phi={total_loss_phi.item():.2e} "
                  f"t={elapsed:.1f}s",
                  flush=True)
            history.append({
                "step": step,
                "loss": loss.item(),
                "loss_u": total_loss_u.item(),
                "loss_p": total_loss_p.item(),
                "loss_phi": total_loss_phi.item(),
                "elapsed_s": elapsed,
            })

        if checkpoint_dir is not None and (step + 1) % save_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"checkpoint_s{step + 1}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "architecture": "simple",
                "model_kwargs": {},
                "mode": "solver-distilled",
                "step": step + 1,
                "history": history,
            }, ckpt_path)
            print(f"[distill] checkpoint: {ckpt_path.name}", flush=True)
            write_json(Path(checkpoint_dir) / "history.json", history)

        # Field diagnostic
        if eval_every > 0 and (step + 1) % eval_every == 0:
            rel_us = []
            rel_ps = []
            model.eval()
            with torch.no_grad():
                for ctx in contexts[:8]:
                    cell_pred, phi_pred = model(ctx["lam"])
                    cell_pred = project_solid_velocity(cell_pred, ctx["lam"])
                    pred_cell = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)

                    # Reconstruct physical state
                    n_u = 3 * n_cells
                    ux_phys = pred_cell[:, 0].cpu().numpy() * 0.1
                    uy_phys = pred_cell[:, 1].cpu().numpy() * 0.1
                    p_phys = pred_cell[:, 2].cpu().numpy() * 0.01

                    ux_grid = ux_phys.reshape(64, 64)
                    uy_grid = uy_phys.reshape(64, 64)
                    p_grid = p_phys.reshape(64, 64)

                    # Add base state
                    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")
                    u0 = w0[:n_u].reshape(n_cells, 3)
                    p0 = w0[n_u:n_u + n_cells]

                    ux_full = u0[:, 0].reshape(64, 64) + ux_grid
                    uy_full = u0[:, 1].reshape(64, 64) + uy_grid
                    p_full = p0.reshape(64, 64) + p_grid

                    rel_u = np.sqrt(
                        np.sum((ctx["ref_ux"] - ux_full)**2 +
                               (ctx["ref_uy"] - uy_full)**2)
                    ) / (np.sqrt(
                        np.sum(ctx["ref_ux"]**2 + ctx["ref_uy"]**2)) + 1e-30)
                    rel_p = np.sqrt(
                        np.sum((ctx["ref_p"] - p_full)**2)
                    ) / (np.sqrt(np.sum(ctx["ref_p"]**2)) + 1e-30)
                    rel_us.append(rel_u)
                    rel_ps.append(rel_p)

            model.train()
            mean_rel_u = float(np.mean(rel_us))
            mean_rel_p = float(np.mean(rel_ps))
            print(f"[field] s{step:4d} train rel_U={mean_rel_u:.4e} "
                  f"rel_p={mean_rel_p:.4e}", flush=True)
            history.append({
                "step": step,
                "field_rel_u": mean_rel_u,
                "field_rel_p": mean_rel_p,
            })

    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["supervised", "physics", "solver-distilled"])
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
    ap.add_argument("--physics-objective", default="residual",
                    choices=["residual", "fixed-point"],
                    help="Physics loss type: residual (JTV) or fixed-point (SIMPLE step)")
    ap.add_argument("--alpha-p", type=float, default=0.1,
                    help="Pressure relaxation for fixed-point target")
    ap.add_argument("--alpha-u", type=float, default=0.1,
                    help="Velocity relaxation for fixed-point target")
    ap.add_argument("--alpha-phi", type=float, default=0.1,
                    help="Flux relaxation for fixed-point target")
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
    elif args.mode == "solver-distilled":
        history = train_solver_distilled(
            model, samples, optimizer, args.device, args.steps,
            args.dataset, args.topology_batch_size,
            checkpoint_dir=str(output_dir),
            save_every=args.save_every,
            eval_every=args.eval_every,
            start_step=start_step,
            prev_history=prev_history)
    elif args.mode == "physics" and args.physics_objective == "fixed-point":
        history = train_fixed_point(
            model, samples, optimizer, args.device, args.steps,
            args.dataset, args.workers,
            topology_batch_size=args.topology_batch_size,
            checkpoint_dir=str(output_dir),
            save_every=args.save_every,
            eval_every=args.eval_every,
            alpha_u=args.alpha_u,
            alpha_p=args.alpha_p,
            alpha_phi=args.alpha_phi,
            start_step=start_step,
            prev_history=prev_history)
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
