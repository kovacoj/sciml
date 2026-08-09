"""Comprehensive state-manifold diagnostic with patch-level phi analysis,
independent-phi assembler, and normalized direct-state optimization.

Phases:
  1. Compute L(W*), L(W_proj_old), L(W0)
  2. Patch-by-patch phi error analysis
  3. Build IndependentPhiStateAssembler, verify L(W_independent_proj) ≈ L(W*)
  4. Normalized direct-state optimization (dimensionless variables)

Usage (inside container):
  python -m diagnostics.diagnose_independent_phi \
    --topology topology_000 \
    --dataset datasets/four_port_64 \
    --checkpoint outputs/unet_physics_simple/checkpoint_s950.pt \
    [--direct-optimize] [--direct-steps 200] [--direct-lr 1e-3]
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
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.state_assembly import StateAssembler  # noqa: E402
from state_layout import build_isothermal_layout  # noqa: E402


def compute_weighted_loss(r, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi):
    lu = 0.5 * gamma_u * float(np.dot(r[u_ids], r[u_ids]))
    lp = 0.5 * gamma_p * float(np.dot(r[p_ids], r[p_ids]))
    lphi = 0.5 * gamma_phi * float(np.dot(r[phi_ids], r[phi_ids]))
    return lu + lp + lphi, lu, lp, lphi


def patch_phi_analysis(w_star, w_projected, mesh_meta):
    """Analyze phi discrepancy patch by patch."""
    n_u = 3 * mesh_meta.n_cells
    n_p = mesh_meta.n_cells
    n_int = mesh_meta.n_internal_faces

    phi_star = w_star[n_u + n_p:]
    phi_proj = w_projected[n_u + n_p:]

    results = []
    for i, name in enumerate(mesh_meta.patch_names):
        start = int(mesh_meta.patch_start_faces[i])
        count = int(mesh_meta.patch_face_counts[i])
        end = start + count

        phi_s = phi_star[start:end]
        phi_p = phi_proj[start:end]

        diff_norm = float(np.linalg.norm(phi_s - phi_p))
        star_norm = float(np.linalg.norm(phi_s))
        proj_norm = float(np.linalg.norm(phi_p))
        rel = diff_norm / (star_norm + 1e-30)

        results.append({
            "patch": name,
            "n_faces": count,
            "start_face": start,
            "phi_star_norm": star_norm,
            "phi_projected_norm": proj_norm,
            "difference_norm": diff_norm,
            "relative_error": rel,
        })

    # Also internal faces
    diff_int = float(np.linalg.norm(
        phi_star[:n_int] - phi_proj[:n_int]))
    star_int = float(np.linalg.norm(phi_star[:n_int]))
    results.insert(0, {
        "patch": "internal",
        "n_faces": n_int,
        "start_face": 0,
        "phi_star_norm": star_int,
        "phi_projected_norm": float(np.linalg.norm(phi_proj[:n_int])),
        "difference_norm": diff_int,
        "relative_error": diff_int / (star_int + 1e-30),
    })

    return results


def build_independent_phi_state(w0, w_star, mesh_meta, layout,
                                  phi_trainable_indices):
    """Construct W_independent_proj: exact U*, p*, exact phi* on trainable
    faces, base phi on frozen faces."""
    n_u = 3 * mesh_meta.n_cells
    n_p = mesh_meta.n_cells
    n_phi = mesh_meta.n_faces

    w_indep = w0.copy()

    # Exact U*
    w_indep[:n_u] = w_star[:n_u]

    # Exact p*
    w_indep[n_u:n_u + n_p] = w_star[n_u:n_u + n_p]

    # phi: base + exact correction on trainable faces
    phi0 = w0[n_u + n_p:]
    phi_star = w_star[n_u + n_p:]
    delta_phi = np.zeros(n_phi, dtype=np.float64)
    delta_phi[phi_trainable_indices] = (
        phi_star[phi_trainable_indices] - phi0[phi_trainable_indices]
    )
    w_indep[n_u + n_p:] = phi0 + delta_phi

    return w_indep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="topology_000")
    ap.add_argument("--dataset", default="datasets/four_port_64")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--direct-optimize", action="store_true")
    ap.add_argument("--direct-steps", type=int, default=200)
    ap.add_argument("--direct-lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    torch.set_default_dtype(torch.float64)

    ds_dir = Path(args.dataset)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir

    topo_dir = ds_dir / args.topology
    case_dir = str(topo_dir / "case")

    diag_dir = (Path(PROJECT_ROOT) / "outputs" /
                "diagnostics_independent_phi" / args.topology)
    diag_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load shared data ----
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
    n_phi = n_faces
    state_size = n_u + n_p + n_phi

    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")

    with open(ds_dir / "shared" / "physics_loss_config.json") as f:
        cfg = json.load(f)
    gamma_u = cfg["gamma_u"]
    gamma_p = cfg["gamma_p"]
    gamma_phi = cfg["gamma_phi"]

    print(f"[diag] topology={args.topology}")
    print(f"[diag] gamma_u={gamma_u:.6e}, gamma_p={gamma_p:.6e}, "
          f"gamma_phi={gamma_phi:.6e}")

    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")

    # ---- Load reference state (from earlier diagnostic) ----
    old_ref_path = (Path(PROJECT_ROOT) / "outputs" / "diagnostics" /
                    args.topology / "reference_state.npy")
    if old_ref_path.exists():
        w_star = np.load(old_ref_path)
        r_star_path = old_ref_path.parent / "reference_residual.npy"
        if r_star_path.exists():
            r_star = np.load(r_star_path)
        else:
            r_star = None
    else:
        print("[diag] Running primal solver to get W*...")
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
        bridge.solver()
        w_star = np.ascontiguousarray(
            bridge.solver.getStates().copy(), dtype=np.float64)
        r_star = bridge.residual(w_star)

    if r_star is None:
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
        r_star = bridge.residual(w_star)

    # ---- Phase 1: L(W*) ----
    loss_star, lu_star, lp_star, lphi_star = compute_weighted_loss(
        r_star, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

    print()
    print("REFERENCE CFD STATE")
    print("-------------------")
    print(f"  ||R(W*)||   = {np.linalg.norm(r_star):.6e}")
    print(f"  L(W*)       = {loss_star:.6e}")
    print(f"  L_U(W*)     = {lu_star:.6e}")
    print(f"  L_p(W*)     = {lp_star:.6e}")
    print(f"  L_phi(W*)   = {lphi_star:.6e}")

    # ---- Phase 2: Old projected state ----
    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=n_cells,
        n_faces=n_faces,
    )
    base_state = torch.from_numpy(w0)
    state_asm_old = StateAssembler(base_state, layout, flux_asm,
                                    n_cells, n_faces)

    u_star = w_star[:n_u].reshape(n_cells, 3)
    u0 = w0[:n_u].reshape(n_cells, 3)
    du_star = u_star[:, :2] - u0[:, :2]
    dp_star = w_star[n_u:n_u + n_p] - w0[n_u:n_u + n_p]
    cell_corr = np.column_stack([du_star[:, 0], du_star[:, 1], dp_star])

    w_proj_old = state_asm_old.assemble(
        torch.from_numpy(cell_corr)).detach().numpy()

    # ---- Phase 2a: Patch-level phi error ----
    print()
    print("[diag] Patch-level phi error analysis...")
    patch_results = patch_phi_analysis(w_star, w_proj_old, mesh_meta)

    print()
    print("PATCH PHI ERROR ANALYSIS")
    print("------------------------")
    print(f"  {'Patch':<16} {'nFaces':>6} {'||phi*||':>12} "
          f"{'||phi_proj||':>12} {'||diff||':>12} {'rel_err':>10}")
    for p in patch_results:
        print(f"  {p['patch']:<16} {p['n_faces']:>6} "
              f"{p['phi_star_norm']:>12.4e} "
              f"{p['phi_projected_norm']:>12.4e} "
              f"{p['difference_norm']:>12.4e} "
              f"{p['relative_error']:>10.4e}")

    with open(diag_dir / "phi_patch_errors.json", "w") as f:
        json.dump(patch_results, f, indent=2)

    # ---- Phase 2b: Determine trainable phi faces ----
    # Internal faces + outlet patches
    patch_names = list(mesh_meta.patch_names)
    trainable_patches = ["outletLower", "outletUpper"]

    phi_trainable = np.zeros(n_phi, dtype=bool)
    phi_trainable[:n_internal] = True  # all internal faces

    for pname in trainable_patches:
        if pname in patch_names:
            idx = patch_names.index(pname)
            start = int(mesh_meta.patch_start_faces[idx])
            count = int(mesh_meta.patch_face_counts[idx])
            phi_trainable[start:start + count] = True

    n_phi_trainable = int(np.sum(phi_trainable))
    phi_trainable_indices = np.flatnonzero(phi_trainable)

    print()
    print(f"[diag] Trainable phi faces: {n_phi_trainable} / {n_phi}")
    print(f"[diag]   internal: {n_internal}")
    for pname in trainable_patches:
        if pname in patch_names:
            idx = patch_names.index(pname)
            count = int(mesh_meta.patch_face_counts[idx])
            print(f"[diag]   {pname}: {count}")

    # ---- Phase 3: Independent-phi projected state ----
    print()
    print("[diag] Building independent-phi projected state...")

    w_indep_proj = build_independent_phi_state(
        w0, w_star, mesh_meta, layout, phi_trainable_indices)
    np.save(diag_dir / "independent_projected_state.npy", w_indep_proj)

    os.chdir(case_dir)
    from mpi4py import MPI
    bridge_indep = DAFoamResidualBridge(
        case_dir,
        hfdib_signed_distance_options(
            case_dir,
            inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"],
        ),
        comm=MPI.COMM_SELF,
    )
    r_indep_proj = bridge_indep.residual(w_indep_proj)

    loss_indep, lu_indep, lp_indep, lphi_indep = compute_weighted_loss(
        r_indep_proj, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

    print()
    print("INDEPENDENT-PHI PROJECTED STATE")
    print("-------------------------------")
    print(f"  ||R(W_indep)||  = {np.linalg.norm(r_indep_proj):.6e}")
    print(f"  L(W_indep)      = {loss_indep:.6e}")
    print(f"  L_U(W_indep)    = {lu_indep:.6e}")
    print(f"  L_p(W_indep)    = {lp_indep:.6e}")
    print(f"  L_phi(W_indep)  = {lphi_indep:.6e}")

    # Also compute old projected for comparison
    r_proj_old = bridge_indep.residual(w_proj_old)
    loss_proj_old, _, _, _ = compute_weighted_loss(
        r_proj_old, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

    # ---- Phase 3: W0 baseline ----
    r0 = bridge_indep.residual(w0)
    loss_0, lu_0, lp_0, lphi_0 = compute_weighted_loss(
        r0, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

    # ---- Phase 3: Network state ----
    metrics = {
        "topology_id": args.topology,
        "loss_config": {
            "gamma_u": gamma_u,
            "gamma_p": gamma_p,
            "gamma_phi": gamma_phi,
        },
        "reference": {
            "loss": loss_star,
            "loss_u": lu_star,
            "loss_p": lp_star,
            "loss_phi": lphi_star,
            "residual_norm": float(np.linalg.norm(r_star)),
        },
        "old_projected": {
            "loss": loss_proj_old,
            "residual_norm": float(np.linalg.norm(r_proj_old)),
        },
        "independent_projected": {
            "loss": loss_indep,
            "loss_u": lu_indep,
            "loss_p": lp_indep,
            "loss_phi": lphi_indep,
            "residual_norm": float(np.linalg.norm(r_indep_proj)),
            "n_phi_trainable": n_phi_trainable,
        },
        "base_state": {
            "loss": loss_0,
            "residual_norm": float(np.linalg.norm(r0)),
        },
        "patch_errors": patch_results,
    }

    # Network checkpoint
    ckpt_path = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(PROJECT_ROOT) / ckpt_path
        if not ckpt_path.exists():
            print(f"[diag] WARNING: checkpoint not found: {ckpt_path}")
            ckpt_path = None

    if ckpt_path and ckpt_path.exists():
        print()
        print("[diag] Loading network checkpoint...")
        ckpt = torch.load(str(ckpt_path), map_location="cpu",
                           weights_only=False)
        from unet.factory import build_model
        model = build_model(ckpt.get("architecture", "simple"),
                             **ckpt.get("model_kwargs", {}))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        lam = np.load(topo_dir / "lambda.npy")
        lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(lam_t)
        from unet.train import project_solid_velocity
        pred = project_solid_velocity(pred, lam_t)
        corrections = pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        w_network = state_asm_old.assemble(corrections).detach().numpy()

        r_network = bridge_indep.residual(w_network)
        loss_net, lu_net, lp_net, lphi_net = compute_weighted_loss(
            r_network, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

        u_net = w_network[:n_u].reshape(n_cells, 3)
        p_net = w_network[n_u:n_u + n_p]
        phi_net = w_network[n_u + n_p:]

        rel_u_net = np.linalg.norm(
            u_star[:, :2] - u_net[:, :2]) / (
            np.linalg.norm(u_star[:, :2]) + 1e-30)
        rel_p_net = np.linalg.norm(
            w_star[n_u:n_u + n_p] - p_net) / (
            np.linalg.norm(w_star[n_u:n_u + n_p]) + 1e-30)
        rel_phi_net = np.linalg.norm(
            w_star[n_u + n_p:n_u + n_p + n_internal] -
            phi_net[:n_internal]) / (
            np.linalg.norm(w_star[n_u + n_p:n_u + n_p + n_internal]) + 1e-30)

        print()
        print("NETWORK STATE")
        print("-------------")
        print(f"  L(W_net)     = {loss_net:.6e}")
        print(f"  rel_U        = {rel_u_net:.6e}")
        print(f"  rel_p        = {rel_p_net:.6e}")
        print(f"  rel_phi_int  = {rel_phi_net:.6e}")

        metrics["network"] = {
            "loss": loss_net,
            "loss_u": lu_net,
            "loss_p": lp_net,
            "loss_phi": lphi_net,
            "rel_u": float(rel_u_net),
            "rel_p": float(rel_p_net),
            "rel_phi_internal": float(rel_phi_net),
        }

    # ---- Phase 4: Normalized direct-state optimization ----
    if args.direct_optimize:
        print()
        print("=" * 60)
        print("PHASE 4: NORMALIZED DIRECT STATE OPTIMIZATION")
        print("=" * 60)

        # Physical scales
        U_SCALE = 0.1
        P_SCALE = U_SCALE ** 2  # 0.01
        FACE_AREA = 0.002 * 0.002  # 4e-6
        PHI_SCALE = U_SCALE * FACE_AREA  # 4e-7

        print(f"[diag] U_SCALE={U_SCALE}, P_SCALE={P_SCALE}, "
              f"PHI_SCALE={PHI_SCALE:.2e}")

        # Trainable mask
        trainable = np.zeros(state_size, dtype=bool)
        for cell in range(n_cells):
            trainable[3 * cell + 0] = True  # Ux
            trainable[3 * cell + 1] = True  # Uy
        trainable[n_u:n_u + n_p] = True  # p
        trainable[n_u + n_p:] = phi_trainable  # trainable phi faces

        n_trainable = int(np.sum(trainable))
        print(f"[diag] trainable dims: {n_trainable} / {state_size}")

        # Build scaling vector
        scale = np.ones(state_size, dtype=np.float64)
        scale[:n_u:3] = U_SCALE  # Ux
        scale[1:n_u:3] = U_SCALE  # Uy
        scale[n_u:n_u + n_p] = P_SCALE  # p
        scale[n_u + n_p:] = PHI_SCALE  # phi

        # Initialize dimensionless variables
        w_phys = w0.copy()
        q = np.zeros(state_size, dtype=np.float64)
        q[trainable] = (w_phys - w0)[trainable] / scale[trainable]

        q_param = torch.nn.Parameter(torch.from_numpy(q).double())
        optimizer = torch.optim.Adam([q_param], lr=args.direct_lr)

        w0_tensor = torch.from_numpy(w0).double()
        scale_tensor = torch.from_numpy(scale)
        mask_tensor = torch.from_numpy(trainable)

        direct_history = []

        for step in range(args.direct_steps):
            optimizer.zero_grad()

            # Reconstruct physical state
            w_np = w0 + scale * q_param.detach().cpu().numpy()
            w_np = np.ascontiguousarray(w_np, dtype=np.float64)

            r = bridge_indep.residual(w_np)
            loss, lu, lp, lphi = compute_weighted_loss(
                r, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

            if not np.isfinite(loss) or loss > 1e4:
                print(f"  [diag] Divergence detected at step {step}: "
                      f"loss={loss:.4e}")
                break

            seed = np.zeros_like(r)
            seed[u_ids] = gamma_u * r[u_ids]
            seed[p_ids] = gamma_p * r[p_ids]
            seed[phi_ids] = gamma_phi * r[phi_ids]

            grad_w = bridge_indep.residual_jacobian_transpose_vector(
                w_np, seed)

            # Chain rule: grad_q = scale * grad_w (elementwise)
            grad_q = scale * grad_w
            grad_q[~trainable] = 0.0

            q_param.grad = torch.from_numpy(grad_q).to(dtype=q_param.dtype)
            optimizer.step()

            with torch.no_grad():
                q_param[~mask_tensor] = 0.0

            # Field errors vs W*
            u_cur = w_np[:n_u].reshape(n_cells, 3)
            p_cur = w_np[n_u:n_u + n_p]
            phi_cur = w_np[n_u + n_p:]

            rel_u = np.linalg.norm(
                u_star[:, :2] - u_cur[:, :2]) / (
                np.linalg.norm(u_star[:, :2]) + 1e-30)
            rel_p = np.linalg.norm(
                w_star[n_u:n_u + n_p] - p_cur) / (
                np.linalg.norm(w_star[n_u:n_u + n_p]) + 1e-30)
            rel_phi = np.linalg.norm(
                w_star[n_u + n_p:n_u + n_p + n_internal] -
                phi_cur[:n_internal]) / (
                np.linalg.norm(w_star[n_u + n_p:n_u + n_p + n_internal])
                + 1e-30)

            grad_u_norm = np.linalg.norm(grad_q[:n_u])
            grad_p_norm = np.linalg.norm(grad_q[n_u:n_u + n_p])
            grad_phi_norm = np.linalg.norm(grad_q[n_u + n_p:])

            if step % 10 == 0 or step == args.direct_steps - 1:
                print(f"  s{step:4d} L={loss:.6e} "
                      f"U={lu:.2e} p={lp:.2e} phi={lphi:.2e} "
                      f"rel_U={rel_u:.4e} rel_p={rel_p:.4e} "
                      f"rel_phi={rel_phi:.4e}")
                print(f"        ||gU||={grad_u_norm:.2e} "
                      f"||gp||={grad_p_norm:.2e} "
                      f"||gphi||={grad_phi_norm:.2e}")
                direct_history.append({
                    "step": step,
                    "loss": loss,
                    "loss_u": lu,
                    "loss_p": lp,
                    "loss_phi": lphi,
                    "rel_u": float(rel_u),
                    "rel_p": float(rel_p),
                    "rel_phi": float(rel_phi),
                    "grad_u_norm": float(grad_u_norm),
                    "grad_p_norm": float(grad_p_norm),
                    "grad_phi_norm": float(grad_phi_norm),
                })

            if loss < 1.01 * loss_star:
                print(f"  [diag] Reached reference loss at step {step}")
                break

        np.save(diag_dir / "direct_state_final.npy",
                w0 + scale * q_param.detach().cpu().numpy())
        with open(diag_dir / "direct_state_history.json", "w") as f:
            json.dump(direct_history, f, indent=2)

        metrics["direct_optimization"] = {
            "initial_loss": direct_history[0]["loss"] if direct_history else None,
            "final_loss": direct_history[-1]["loss"] if direct_history else None,
            "final_rel_u": direct_history[-1].get("rel_u") if direct_history else None,
            "final_rel_p": direct_history[-1].get("rel_p") if direct_history else None,
            "final_rel_phi": direct_history[-1].get("rel_phi") if direct_history else None,
            "n_steps": len(direct_history),
            "lr": args.direct_lr,
        }

    # ---- Save metrics ----
    with open(diag_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  L(W*)           = {loss_star:.6e}")
    print(f"  L(W_proj_old)   = {loss_proj_old:.6e}")
    print(f"  L(W_indep_proj) = {loss_indep:.6e}")
    print(f"  L(W0)           = {loss_0:.6e}")
    if "network" in metrics:
        print(f"  L(W_net)        = {metrics['network']['loss']:.6e}")
    if args.direct_optimize and "direct_optimization" in metrics:
        d = metrics["direct_optimization"]
        print(f"  Direct opt: {d['initial_loss']:.4e} -> "
              f"{d['final_loss']:.4e} ({d['n_steps']} steps)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
