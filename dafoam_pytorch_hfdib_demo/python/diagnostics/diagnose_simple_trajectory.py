"""Diagnostic: teacher quality + pure SIMPLE trajectory from W0.

Phase A: For 4 optimizer steps, compare field errors of:
  - network state W_theta
  - raw SIMPLE target T(W_theta)
  - relaxed target W + alpha*(T(W)-W)
  - network state after Adam step

Phase B: Pure SIMPLE trajectory from W0 for k=0,1,2,3,5,10,20

Usage (inside container):
  python -m diagnostics.diagnose_simple_trajectory \
    --topology topology_000 \
    --dataset datasets/four_port_64 \
    --checkpoint outputs/physics_fixed_point/checkpoint_s4.pt \
    --alpha-u 0.1 --alpha-p 0.1 --alpha-phi 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_isothermal_layout  # noqa: E402


def rel_field_error(w, w_star, n_cells, n_u, n_p, n_internal, phi_indices):
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
    ap.add_argument("--alpha-u", type=float, default=0.1)
    ap.add_argument("--alpha-p", type=float, default=0.1)
    ap.add_argument("--alpha-phi", type=float, default=0.1)
    ap.add_argument("--trajectory-topologies", default="topology_000,topology_005,topology_010,topology_015")
    ap.add_argument("--trajectory-steps", default="0,1,2,3,5,10,20")
    args = ap.parse_args()

    import torch
    torch.set_default_dtype(torch.float64)

    ds_dir = Path(args.dataset)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir

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

    diag_dir = Path(PROJECT_ROOT) / "outputs" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Load reference states for topologies
    ref_states = {}
    for tid in [args.topology] + args.trajectory_topologies.split(","):
        tid = tid.strip()
        ref_path = Path(PROJECT_ROOT) / "outputs" / "diagnostics" / tid / "reference_state.npy"
        if ref_path.exists():
            ref_states[tid] = np.load(ref_path)

    # ================================================================
    # Phase B: Pure SIMPLE trajectory from W0
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE B: PURE SIMPLE TRAJECTORY FROM W0")
    print("=" * 60)

    traj_topologies = [t.strip() for t in args.trajectory_topologies.split(",")]
    traj_steps = [int(s) for s in args.trajectory_steps.split(",")]

    traj_results = {}

    for tid in traj_topologies:
        print(f"\n[traj] {tid}")
        topo_dir = ds_dir / tid
        case_dir = str(topo_dir / "case")

        w_star = ref_states.get(tid)
        if w_star is None:
            print(f"  [traj] No reference state for {tid}, skipping")
            continue

        # Save trajectory states
        traj_dir = topo_dir / "simple_trajectory"
        traj_dir.mkdir(exist_ok=True)

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

        w = w0.copy()
        np.save(traj_dir / "state_k000.npy", w)

        results = []
        for k in range(max(traj_steps) + 1):
            if k > 0:
                w = bridge.simple_step(w)
                if k in traj_steps:
                    np.save(traj_dir / f"state_k{k:03d}.npy", w)

            if k in traj_steps:
                rel_u, rel_p, rel_phi = rel_field_error(
                    w, w_star, n_cells, n_u, n_p, n_internal,
                    phi_trainable_indices)

                # Also compute residual loss
                r = bridge.residual(w)
                with open(ds_dir / "shared" / "physics_loss_config.json") as f:
                    cfg = json.load(f)
                gu, gp, gphi = cfg["gamma_u"], cfg["gamma_p"], cfg["gamma_phi"]
                lr = 0.5 * gu * np.dot(r[u_ids], r[u_ids]) + \
                     0.5 * gp * np.dot(r[p_ids], r[p_ids]) + \
                     0.5 * gphi * np.dot(r[phi_ids], r[phi_ids])

                # Fixed-point defect
                if k < max(traj_steps):
                    w_next = bridge.simple_step(w)
                    dw = w_next - w
                    U_S, P_S, PHI_S = 0.1, 0.01, 4e-7
                    d_u = np.mean((dw[:n_u].reshape(n_cells,3)[:, 0] / U_S)**2 +
                                  (dw[:n_u].reshape(n_cells,3)[:, 1] / U_S)**2)
                    d_p = np.mean((dw[n_u:n_u+n_p] / P_S)**2)
                    d_phi = np.mean((dw[n_u+n_p:][phi_trainable_indices] / PHI_S)**2)
                    lfp = (d_u + d_p + d_phi) / 3.0
                else:
                    lfp = -1.0

                print(f"  k={k:3d}  rel_U={rel_u:.4e}  rel_p={rel_p:.4e}  "
                      f"rel_phi={rel_phi:.4e}  L_R={lr:.4e}  L_FP={lfp:.4e}")

                results.append({
                    "k": k,
                    "rel_u": rel_u,
                    "rel_p": rel_p,
                    "rel_phi": rel_phi,
                    "loss_R": float(lr),
                    "L_FP": float(lfp),
                })

        traj_results[tid] = results
        del bridge

    with open(diag_dir / "simple_trajectory.json", "w") as f:
        json.dump(traj_results, f, indent=2)

    # ================================================================
    # Phase A: Teacher quality diagnostic
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE A: TEACHER QUALITY DIAGNOSTIC")
    print("=" * 60)

    tid = args.topology
    topo_dir = ds_dir / tid
    case_dir = str(topo_dir / "case")
    w_star = ref_states.get(tid)

    if w_star is None:
        print(f"[teacher] No reference state for {tid}")
        return 1

    # Load network checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(PROJECT_ROOT) / ckpt_path

    if not ckpt_path or not ckpt_path.exists():
        print(f"[teacher] No checkpoint found: {ckpt_path}")
        return 1

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    from unet.factory import build_model
    model = build_model(ckpt.get("architecture", "simple"),
                         **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()

    from pinn.flux_assembly import FluxAssembler
    from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
    from unet.train import project_solid_velocity

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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    teacher_results = []
    alpha_u, alpha_p, alpha_phi = args.alpha_u, args.alpha_p, args.alpha_phi

    for step in range(4):
        print(f"\n[teacher] step {step}")

        # Network forward
        cell_pred, phi_pred = model(lam_t)
        cell_proj = project_solid_velocity(cell_pred, lam_t)
        corrections = cell_proj.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        phi_corr = phi_pred.squeeze(0)
        w_net = state_asm.assemble(corrections, phi_corr).detach().numpy()

        rel_u_net, rel_p_net, rel_phi_net = rel_field_error(
            w_net, w_star, n_cells, n_u, n_p, n_internal,
            phi_trainable_indices)

        # SIMPLE step
        w_simple = bridge.simple_step(w_net)

        rel_u_sim, rel_p_sim, rel_phi_sim = rel_field_error(
            w_simple, w_star, n_cells, n_u, n_p, n_internal,
            phi_trainable_indices)

        # Relaxed target
        w_relaxed = w_net.copy()
        w_relaxed[:n_u] = w_net[:n_u] + alpha_u * (w_simple[:n_u] - w_net[:n_u])
        w_relaxed[n_u:n_u+n_p] = w_net[n_u:n_u+n_p] + alpha_p * (
            w_simple[n_u:n_u+n_p] - w_net[n_u:n_u+n_p])
        phi_s = n_u + n_p
        w_relaxed[phi_s:][phi_trainable_indices] = (
            w_net[phi_s:][phi_trainable_indices] + alpha_phi * (
                w_simple[phi_s:][phi_trainable_indices] -
                w_net[phi_s:][phi_trainable_indices]))

        rel_u_rel, rel_p_rel, rel_phi_rel = rel_field_error(
            w_relaxed, w_star, n_cells, n_u, n_p, n_internal,
            phi_trainable_indices)

        # Optimizer step
        optimizer.zero_grad(set_to_none=True)
        w_torch = state_asm.assemble(corrections, phi_corr)

        U_S, P_S, PHI_S = 0.1, 0.01, 4e-7
        phi_idx_t = torch.from_numpy(phi_trainable_indices.astype(np.int64))

        target_ux = torch.from_numpy(
            w_net[0:n_u:3] + alpha_u * (w_simple[0:n_u:3] - w_net[0:n_u:3]))
        target_uy = torch.from_numpy(
            w_net[1:n_u:3] + alpha_u * (w_simple[1:n_u:3] - w_net[1:n_u:3]))
        target_p = torch.from_numpy(
            w_net[n_u:n_u+n_p] + alpha_p * (
                w_simple[n_u:n_u+n_p] - w_net[n_u:n_u+n_p]))
        target_phi = torch.from_numpy(
            w_net[phi_s:][phi_trainable_indices] + alpha_phi * (
                w_simple[phi_s:][phi_trainable_indices] -
                w_net[phi_s:][phi_trainable_indices]))

        du = (w_torch[0:n_u:3] - target_ux) / U_S
        dv = (w_torch[1:n_u:3] - target_uy) / U_S
        dp = (w_torch[n_u:n_u+n_p] - target_p) / P_S
        dphi = (w_torch[phi_s:][phi_idx_t] - target_phi) / PHI_S

        loss = (torch.mean(du**2 + dv**2) + torch.mean(dp**2) +
                torch.mean(dphi**2)) / 3.0
        loss.backward()
        optimizer.step()

        # Network after step
        with torch.no_grad():
            cell_pred2, phi_pred2 = model(lam_t)
            cell_proj2 = project_solid_velocity(cell_pred2, lam_t)
            corrections2 = cell_proj2.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            phi_corr2 = phi_pred2.squeeze(0)
            w_net2 = state_asm.assemble(corrections2, phi_corr2).detach().numpy()

        rel_u_after, rel_p_after, rel_phi_after = rel_field_error(
            w_net2, w_star, n_cells, n_u, n_p, n_internal,
            phi_trainable_indices)

        print(f"  network before:  rel_U={rel_u_net:.4e}  rel_p={rel_p_net:.4e}")
        print(f"  SIMPLE target:   rel_U={rel_u_sim:.4e}  rel_p={rel_p_sim:.4e}")
        print(f"  relaxed target:  rel_U={rel_u_rel:.4e}  rel_p={rel_p_rel:.4e}")
        print(f"  network after:   rel_U={rel_u_after:.4e}  rel_p={rel_p_after:.4e}")

        teacher_results.append({
            "step": step,
            "net_before": {"rel_u": rel_u_net, "rel_p": rel_p_net},
            "simple_target": {"rel_u": rel_u_sim, "rel_p": rel_p_sim},
            "relaxed_target": {"rel_u": rel_u_rel, "rel_p": rel_p_rel},
            "net_after": {"rel_u": rel_u_after, "rel_p": rel_p_after},
        })

    with open(diag_dir / "fp_teacher_quality.json", "w") as f:
        json.dump(teacher_results, f, indent=2)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nB. SIMPLE trajectory from W0:")
    print(f"{'k':>4} | {'rel_U':>12} | {'rel_p':>12} | {'L_FP':>12}")
    for tid in traj_topologies:
        if tid in traj_results:
            print(f"\n  {tid}:")
            for r in traj_results[tid]:
                print(f"  k={r['k']:3d}  rel_U={r['rel_u']:.4e}  "
                      f"rel_p={r['rel_p']:.4e}  L_FP={r['L_FP']:.4e}")

    print("\nA. Teacher quality:")
    print(f"{'step':>4} | {'net bef U':>12} | {'SIMPLE U':>12} | "
          f"{'relaxed U':>12} | {'net aft U':>12}")
    for r in teacher_results:
        print(f"  s={r['step']}  "
              f"net={r['net_before']['rel_u']:.4e}  "
              f"SIM={r['simple_target']['rel_u']:.4e}  "
              f"rel={r['relaxed_target']['rel_u']:.4e}  "
              f"aft={r['net_after']['rel_u']:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
