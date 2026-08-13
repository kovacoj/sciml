"""Diagnose whether the StateAssembler's phi=phi0+A_phi*dU constraint
excludes the converged CFD state.

Phases 1-3: compute L(W*), L(W_projected), L(W_network) for one topology.
Phase 4 (optional): direct state optimization without neural network.

Usage (inside container):
  python -m diagnostics.diagnose_state_manifold \
    --topology topology_000 \
    --dataset datasets/four_port_64 \
    --checkpoint outputs/unet_physics_simple/checkpoint_s950.pt

Outputs:
  outputs/diagnostics/topology_000/reference_state.npy
  outputs/diagnostics/topology_000/reference_residual.npy
  outputs/diagnostics/topology_000/projected_state.npy
  outputs/diagnostics/topology_000/metrics.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="topology_000")
    ap.add_argument("--dataset", default="datasets/four_port_64")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--direct-optimize", action="store_true",
                    help="Run Phase 4: direct state optimization")
    ap.add_argument("--direct-steps", type=int, default=100)
    ap.add_argument("--direct-lr", type=float, default=1e-3)
    args = ap.parse_args()

    import torch
    torch.set_default_dtype(torch.float64)

    ds_dir = Path(args.dataset)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir

    topo_dir = ds_dir / args.topology
    case_dir = str(topo_dir / "case")

    diag_dir = Path(PROJECT_ROOT) / "outputs" / "diagnostics" / args.topology
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

    assert n_cells == 4096, f"n_cells={n_cells}"
    assert n_internal == 8064, f"n_internal={n_internal}"
    assert n_faces == 16512, f"n_faces={n_faces}"
    assert state_size == 32896, f"state_size={state_size}"

    print(f"[diag] topology={args.topology}")
    print(f"[diag] n_cells={n_cells}, n_internal={n_internal}, "
          f"n_faces={n_faces}, state_size={state_size}")

    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")

    with open(ds_dir / "shared" / "physics_loss_config.json") as f:
        cfg = json.load(f)
    gamma_u = cfg["gamma_u"]
    gamma_p = cfg["gamma_p"]
    gamma_phi = cfg["gamma_phi"]

    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")
    print(f"[diag] w0 shape={w0.shape}, ||w0||={np.linalg.norm(w0):.3e}")

    # ---- Phase 1: Get reference CFD state W* ----
    ref_state_path = diag_dir / "reference_state.npy"
    ref_resid_path = diag_dir / "reference_residual.npy"

    if ref_state_path.exists():
        print("[diag] Loading cached reference state")
        w_star = np.load(ref_state_path)
        r_star = np.load(ref_resid_path) if ref_resid_path.exists() else None
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
        np.save(ref_state_path, w_star)
        print(f"[diag] W* saved: ||W*||={np.linalg.norm(w_star):.3e}")

        r_star = bridge.residual(w_star)
        np.save(ref_resid_path, r_star)
        print(f"[diag] R(W*) saved: ||R(W*)||={np.linalg.norm(r_star):.3e}")

    if r_star is None:
        print("[diag] Re-evaluating residual for W*...")
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
        np.save(ref_resid_path, r_star)

    # ---- Phase 1: Compute L(W*) ----
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

    # ---- Phase 2: Project W* through StateAssembler ----
    print()
    print("[diag] Projecting W* through StateAssembler...")

    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=n_cells,
        n_faces=n_faces,
    )

    base_state = torch.from_numpy(w0)
    state_asm = StateAssembler(base_state, layout, flux_asm,
                                n_cells, n_faces)

    # Extract reference corrections
    u_star_flat = w_star[:n_u]
    p_star = w_star[n_u:n_u + n_p]
    phi_star = w_star[n_u + n_p:]

    u0_flat = w0[:n_u]
    p0 = w0[n_u:n_u + n_p]
    phi0 = w0[n_u + n_p:]

    u_star = u_star_flat.reshape(n_cells, 3)
    u0 = u0_flat.reshape(n_cells, 3)

    du_star = u_star[:, :2] - u0[:, :2]
    dp_star = p_star - p0

    cell_corrections = np.column_stack([
        du_star[:, 0],
        du_star[:, 1],
        dp_star,
    ]).astype(np.float64)

    w_projected = state_asm.assemble(
        torch.from_numpy(cell_corrections)
    ).detach().numpy()
    np.save(diag_dir / "projected_state.npy", w_projected)

    # ---- Phase 2.2: Flux-manifold error ----
    phi_projected = w_projected[n_u + n_p:]

    rel_phi_internal = np.linalg.norm(
        phi_star[:n_internal] - phi_projected[:n_internal]
    ) / (np.linalg.norm(phi_star[:n_internal]) + 1e-30)

    rel_phi_all = np.linalg.norm(
        phi_star - phi_projected
    ) / (np.linalg.norm(phi_star) + 1e-30)

    print()
    print("STATE-MANIFOLD TEST")
    print("-------------------")
    print(f"  rel phi error internal = {rel_phi_internal:.6e}")
    print(f"  rel phi error all      = {rel_phi_all:.6e}")

    # ---- Phase 2.3: Residual of projected state ----
    print()
    print("[diag] Computing residual of projected state...")
    os.chdir(case_dir)
    from mpi4py import MPI
    bridge_proj = DAFoamResidualBridge(
        case_dir,
        hfdib_signed_distance_options(
            case_dir,
            inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"],
        ),
        comm=MPI.COMM_SELF,
    )
    r_projected = bridge_proj.residual(w_projected)

    loss_proj, lu_proj, lp_proj, lphi_proj = compute_weighted_loss(
        r_projected, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

    print()
    print("PROJECTED STATE (exact U*,p* but phi=A_phi*dU)")
    print("-------------------")
    print(f"  ||R(W_proj)||  = {np.linalg.norm(r_projected):.6e}")
    print(f"  L(W_proj)      = {loss_proj:.6e}")
    print(f"  L_U(W_proj)    = {lu_proj:.6e}")
    print(f"  L_p(W_proj)    = {lp_proj:.6e}")
    print(f"  L_phi(W_proj)  = {lphi_proj:.6e}")

    # ---- Phase 3: Network state ----
    metrics = {
        "topology_id": args.topology,
        "reference": {
            "loss": loss_star,
            "loss_u": lu_star,
            "loss_p": lp_star,
            "loss_phi": lphi_star,
            "residual_norm": float(np.linalg.norm(r_star)),
        },
        "projected_reference": {
            "loss": loss_proj,
            "loss_u": lu_proj,
            "loss_p": lp_proj,
            "loss_phi": lphi_proj,
            "rel_phi_internal": float(rel_phi_internal),
            "rel_phi_all": float(rel_phi_all),
            "residual_norm": float(np.linalg.norm(r_projected)),
        },
    }

    # Resolve checkpoint path before any chdir
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
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model_state = ckpt["model_state_dict"]

        from unet.factory import build_model
        model = build_model(ckpt.get("architecture", "simple"),
                             **ckpt.get("model_kwargs", {}))
        model.load_state_dict(model_state)
        model.eval()

        lam = np.load(topo_dir / "lambda.npy")
        lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = model(lam_t)
        from unet.train import project_solid_velocity
        pred = project_solid_velocity(pred, lam_t)
        corrections = pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        w_network = state_asm.assemble(corrections).detach().numpy()

        r_network = bridge_proj.residual(w_network)
        loss_net, lu_net, lp_net, lphi_net = compute_weighted_loss(
            r_network, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

        # Field errors vs W*
        u_net = w_network[:n_u].reshape(n_cells, 3)
        p_net = w_network[n_u:n_u + n_p]
        phi_net = w_network[n_u + n_p:]

        rel_u_net = np.linalg.norm(
            u_star[:, :2] - u_net[:, :2]
        ) / (np.linalg.norm(u_star[:, :2]) + 1e-30)

        rel_p_net = np.linalg.norm(
            p_star - p_net
        ) / (np.linalg.norm(p_star) + 1e-30)

        rel_phi_net = np.linalg.norm(
            phi_star[:n_internal] - phi_net[:n_internal]
        ) / (np.linalg.norm(phi_star[:n_internal]) + 1e-30)

        print()
        print("NETWORK STATE (step 950)")
        print("-------------------")
        print(f"  L(W_net)       = {loss_net:.6e}")
        print(f"  L_U(W_net)     = {lu_net:.6e}")
        print(f"  L_p(W_net)     = {lp_net:.6e}")
        print(f"  L_phi(W_net)   = {lphi_net:.6e}")
        print(f"  rel_U vs W*    = {rel_u_net:.6e}")
        print(f"  rel_p vs W*    = {rel_p_net:.6e}")
        print(f"  rel_phi_int    = {rel_phi_net:.6e}")

        metrics["network"] = {
            "loss": loss_net,
            "loss_u": lu_net,
            "loss_p": lp_net,
            "loss_phi": lphi_net,
            "rel_u": float(rel_u_net),
            "rel_p": float(rel_p_net),
            "rel_phi_internal": float(rel_phi_net),
            "residual_norm": float(np.linalg.norm(r_network)),
        }

    # ---- Phase 3: W0 baseline ----
    r0 = bridge_proj.residual(w0)
    loss_0, lu_0, lp_0, lphi_0 = compute_weighted_loss(
        r0, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

    print()
    print("BASE STATE W0")
    print("-------------------")
    print(f"  L(W0)       = {loss_0:.6e}")
    print(f"  L_U(W0)     = {lu_0:.6e}")
    print(f"  L_p(W0)     = {lp_0:.6e}")
    print(f"  L_phi(W0)   = {lphi_0:.6e}")

    metrics["base_state"] = {
        "loss": loss_0,
        "loss_u": lu_0,
        "loss_p": lp_0,
        "loss_phi": lphi_0,
        "residual_norm": float(np.linalg.norm(r0)),
    }

    # ---- Phase 4: Direct state optimization (optional) ----
    if args.direct_optimize:
        print()
        print("=" * 60)
        print("PHASE 4: DIRECT STATE OPTIMIZATION (no neural network)")
        print("=" * 60)

        # Trainable mask: Ux, Uy, p, internal phi
        trainable = np.zeros(state_size, dtype=bool)
        for cell in range(n_cells):
            trainable[3 * cell + 0] = True
            trainable[3 * cell + 1] = True
        trainable[n_u:n_u + n_p] = True
        trainable[n_u + n_p:n_u + n_p + n_internal] = True

        n_trainable = int(np.sum(trainable))
        print(f"[diag] trainable dims: {n_trainable} / {state_size}")

        w_param = torch.nn.Parameter(torch.from_numpy(w0.copy()).double())
        optimizer = torch.optim.Adam([w_param], lr=args.direct_lr)

        w0_tensor = torch.from_numpy(w0).double()
        mask_tensor = torch.from_numpy(trainable)

        direct_history = []

        for step in range(args.direct_steps):
            optimizer.zero_grad()

            w_np = np.ascontiguousarray(
                w_param.detach().cpu().numpy(), dtype=np.float64)

            r = bridge_proj.residual(w_np)

            loss, lu, lp, lphi = compute_weighted_loss(
                r, u_ids, p_ids, phi_ids, gamma_u, gamma_p, gamma_phi)

            seed = np.zeros_like(r)
            seed[u_ids] = gamma_u * r[u_ids]
            seed[p_ids] = gamma_p * r[p_ids]
            seed[phi_ids] = gamma_phi * r[phi_ids]

            grad = bridge_proj.residual_jacobian_transpose_vector(w_np, seed)
            grad[~trainable] = 0.0

            w_param.grad = torch.from_numpy(grad).to(dtype=w_param.dtype)
            optimizer.step()

            with torch.no_grad():
                w_param[~mask_tensor] = w0_tensor[~mask_tensor]

            # Field errors vs W*
            u_cur = w_np[:n_u].reshape(n_cells, 3)
            p_cur = w_np[n_u:n_u + n_p]
            phi_cur = w_np[n_u + n_p:]

            rel_u = np.linalg.norm(
                u_star[:, :2] - u_cur[:, :2]
            ) / (np.linalg.norm(u_star[:, :2]) + 1e-30)
            rel_p = np.linalg.norm(
                p_star - p_cur
            ) / (np.linalg.norm(p_star) + 1e-30)
            rel_phi = np.linalg.norm(
                phi_star[:n_internal] - phi_cur[:n_internal]
            ) / (np.linalg.norm(phi_star[:n_internal]) + 1e-30)

            if step % 10 == 0 or step == args.direct_steps - 1:
                print(f"  s{step:4d} L={loss:.6e} "
                      f"U={lu:.2e} p={lp:.2e} phi={lphi:.2e} "
                      f"rel_U={rel_u:.4e} rel_p={rel_p:.4e} "
                      f"rel_phi={rel_phi:.4e}")
                direct_history.append({
                    "step": step,
                    "loss": loss,
                    "loss_u": lu,
                    "loss_p": lp,
                    "loss_phi": lphi,
                    "rel_u": float(rel_u),
                    "rel_p": float(rel_p),
                    "rel_phi": float(rel_phi),
                })

            if loss < 1.01 * loss_star:
                print(f"  [diag] Reached reference loss at step {step}")
                break

        np.save(diag_dir / "direct_state_final.npy",
                w_param.detach().cpu().numpy())
        with open(diag_dir / "direct_state_history.json", "w") as f:
            json.dump(direct_history, f, indent=2)

        metrics["direct_optimization"] = {
            "initial_loss": direct_history[0]["loss"] if direct_history else None,
            "final_loss": direct_history[-1]["loss"] if direct_history else None,
            "final_rel_u": direct_history[-1]["rel_u"] if direct_history else None,
            "final_rel_p": direct_history[-1]["rel_p"] if direct_history else None,
            "final_rel_phi": direct_history[-1]["rel_phi"] if direct_history else None,
            "n_steps": len(direct_history),
        }

    # ---- Save metrics ----
    with open(diag_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  L(W*)       = {loss_star:.6e}")
    print(f"  L(W_proj)   = {loss_proj:.6e}")
    print(f"  L(W0)       = {loss_0:.6e}")
    if "network" in metrics:
        print(f"  L(W_net)    = {metrics['network']['loss']:.6e}")
    print(f"  e_phi_int   = {rel_phi_internal:.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
