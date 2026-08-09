"""Diagnostic: test SIMPLE fixed-point defect on known states.

Phase B-D: Validate that one SIMPLE step moves toward the CFD solution.

Usage (inside container):
  python -m diagnostics.diagnose_fixed_point \
    --topology topology_000 \
    --dataset datasets/four_port_64 \
    --checkpoint outputs/physics_independent_phi/checkpoint_s500.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_isothermal_layout  # noqa: E402


def compute_weighted_loss(r, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi):
    lu = 0.5 * gamma_u * float(np.dot(r[u_ids], r[u_ids]))
    lp = 0.5 * gamma_p * float(np.dot(r[p_ids], r[p_ids]))
    lphi = 0.5 * gamma_phi * float(np.dot(r[phi_ids], r[phi_ids]))
    return lu + lp + lphi, lu, lp, lphi


def nondim_fp_norm(delta_w, n_cells, n_internal, n_u, n_p, phi_trainable_indices):
    """Compute nondimensional fixed-point norm L_FP."""
    U_SCALE = 0.1
    P_SCALE = 0.01
    PHI_SCALE = 4e-7

    du = delta_w[:n_u].reshape(n_cells, 3)
    dp = delta_w[n_u:n_u + n_p]
    dphi = delta_w[n_u + n_p:]

    d_u = np.mean((du[:, 0] / U_SCALE)**2 + (du[:, 1] / U_SCALE)**2)
    d_p = np.mean((dp / P_SCALE)**2)
    d_phi = np.mean((dphi[phi_trainable_indices] / PHI_SCALE)**2)

    return (d_u + d_p + d_phi) / 3.0, d_u, d_p, d_phi


def rel_field_error(w, w_star, n_cells, n_u, n_p, n_internal, phi_indices):
    """Compute relative field errors vs W*."""
    u = w[:n_u].reshape(n_cells, 3)
    u_star = w_star[:n_u].reshape(n_cells, 3)
    p = w[n_u:n_u + n_p]
    p_star = w_star[n_u:n_u + n_p]
    phi = w[n_u + n_p:]
    phi_star = w_star[n_u + n_p:]

    rel_u = np.linalg.norm(u_star[:, :2] - u[:, :2]) / (
        np.linalg.norm(u_star[:, :2]) + 1e-30)
    rel_p = np.linalg.norm(p_star - p) / (
        np.linalg.norm(p_star) + 1e-30)
    rel_phi = np.linalg.norm(phi_star[phi_indices] - phi[phi_indices]) / (
        np.linalg.norm(phi_star[phi_indices]) + 1e-30)

    return float(rel_u), float(rel_p), float(rel_phi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="topology_000")
    ap.add_argument("--dataset", default="datasets/four_port_64")
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import torch
    torch.set_default_dtype(torch.float64)

    ds_dir = Path(args.dataset)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir

    topo_dir = ds_dir / args.topology
    case_dir = str(topo_dir / "case")

    diag_dir = (Path(PROJECT_ROOT) / "outputs" /
                "fixed_point_diagnostic" / args.topology)
    diag_dir.mkdir(parents=True, exist_ok=True)

    from pinn.mesh_metadata import MeshMetadata
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

    with open(ds_dir / "shared" / "physics_loss_config.json") as f:
        cfg = json.load(f)
    gamma_u = cfg["gamma_u"]
    gamma_p = cfg["gamma_p"]
    gamma_phi = cfg["gamma_phi"]

    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")

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

    # Load reference state
    ref_path = (Path(PROJECT_ROOT) / "outputs" / "diagnostics" /
                args.topology / "reference_state.npy")
    if ref_path.exists():
        w_star = np.load(ref_path)
    else:
        print("[diag] No reference state found. Run diagnose_state_manifold first.")
        return 1

    # Initialize bridge
    print("[diag] Initializing DAFoam bridge...")
    os.chdir(case_dir)
    from mpi4py import MPI
    bridge = DAFoamResidualBridge(
        case_dir,
        hfdib_signed_distance_options(
            case_dir,
            inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"],
        ),
        comm=MPI.COMM_SELF,
    )

    # Load network checkpoint
    w_net = None
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(PROJECT_ROOT) / ckpt_path

    if ckpt_path and ckpt_path.exists():
        print("[diag] Loading network checkpoint...")
        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                           weights_only=False)
        from unet.factory import build_model
        model = build_model(ckpt.get("architecture", "simple"),
                             **ckpt.get("model_kwargs", {}))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        from pinn.flux_assembly import FluxAssembler
        from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
        flux_asm = FluxAssembler(
            owners=mesh_meta.owners,
            neighbours=mesh_meta.neighbours,
            sf_vec=mesh_meta.face_area_vectors,
            owner_weights=mesh_meta.owner_weights,
            n_cells=n_cells,
            n_faces=n_faces,
        )
        base_state = torch.from_numpy(w0)
        state_asm = IndependentPhiStateAssembler(
            base_state, layout, flux_asm, n_cells, n_faces,
            phi_trainable_indices)

        lam = np.load(topo_dir / "lambda.npy")
        lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cell_pred, phi_pred = model(lam_t)
        from unet.train import project_solid_velocity
        cell_pred = project_solid_velocity(cell_pred, lam_t)
        corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        phi_corr = phi_pred.squeeze(0)
        w_net = state_asm.assemble(corrections, phi_corr).detach().numpy()
        print(f"[diag] W_net loaded: ||W_net||={np.linalg.norm(w_net):.3e}")

    # ---- Phase B: Apply one SIMPLE step to each state ----
    results = {}

    for name, w_in in [("W0", w0), ("W*", w_star)] + (
            [("W_net500", w_net)] if w_net is not None else []):

        print(f"\n[diag] Applying SIMPLE step to {name}...")

        # Before
        r_before = bridge.residual(w_in)
        loss_before, _, _, _ = compute_weighted_loss(
            r_before, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)
        rel_u_before, rel_p_before, rel_phi_before = rel_field_error(
            w_in, w_star, n_cells, n_u, n_p, n_internal,
            phi_trainable_indices)

        # One SIMPLE step
        w_step = bridge.simple_step(w_in)
        np.save(diag_dir / f"{name.replace('*','star')}_step.npy", w_step)

        # After
        r_after = bridge.residual(w_step)
        loss_after, _, _, _ = compute_weighted_loss(
            r_after, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)
        rel_u_after, rel_p_after, rel_phi_after = rel_field_error(
            w_step, w_star, n_cells, n_u, n_p, n_internal,
            phi_trainable_indices)

        # Fixed-point defect
        delta_w = w_step - w_in
        lfp, d_u, d_p, d_phi = nondim_fp_norm(
            delta_w, n_cells, n_internal, n_u, n_p, phi_trainable_indices)

        print(f"  {name}:")
        print(f"    L_R before/after:  {loss_before:.4e} -> {loss_after:.4e}")
        print(f"    L_FP:              {lfp:.4e}  (U={d_u:.4e} p={d_p:.4e} phi={d_phi:.4e})")
        print(f"    rel_U before/after: {rel_u_before:.4e} -> {rel_u_after:.4e}")
        print(f"    rel_p before/after: {rel_p_before:.4e} -> {rel_p_after:.4e}")
        print(f"    rel_phi before/after: {rel_phi_before:.4e} -> {rel_phi_after:.4e}")

        results[name] = {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "fp_loss": lfp,
            "fp_d_u": d_u,
            "fp_d_p": d_p,
            "fp_d_phi": d_phi,
            "rel_u_before": rel_u_before,
            "rel_u_after": rel_u_after,
            "rel_p_before": rel_p_before,
            "rel_p_after": rel_p_after,
            "rel_phi_before": rel_phi_before,
            "rel_phi_after": rel_phi_after,
        }

    with open(diag_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'State':<12} {'L_FP':>12} {'rel_U bef':>12} {'rel_U aft':>12} {'rel_p bef':>12} {'rel_p aft':>12}")
    for name, r in results.items():
        print(f"{name:<12} {r['fp_loss']:>12.4e} "
              f"{r['rel_u_before']:>12.4e} {r['rel_u_after']:>12.4e} "
              f"{r['rel_p_before']:>12.4e} {r['rel_p_after']:>12.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
