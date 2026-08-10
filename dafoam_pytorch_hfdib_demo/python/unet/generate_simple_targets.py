"""Generate truncated-SIMPLE targets for training topologies.

For each of the 16 training topologies, start from W0 and apply
K=20 SIMPLE iterations, saving states at k=0,1,3,5,10,20.

Also converts W_10 and W_20 to network coordinates (q_cell, q_phi)
and computes block normalization energies.

Usage (inside container):
  python -m unet.generate_simple_targets \
    --dataset datasets/four_port_64 \
    --k-max 20 \
    --k-primary 10
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/four_port_64")
    ap.add_argument("--k-max", type=int, default=80,
                    help="Maximum SIMPLE steps to run")
    ap.add_argument("--k-primary", type=int, default=20,
                    help="Primary target step (saves q_cell, q_phi, normalization)")
    ap.add_argument("--save-ks", default="",
                    help="Comma-separated extra K values to save (e.g. 5,10,20,40,80)")
    args = ap.parse_args()

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

    # Save indices — always include 0, primary, max; add extra requested Ks
    save_ks = {0, args.k_primary, args.k_max}
    if args.save_ks:
        for s in args.save_ks.split(","):
            s = s.strip()
            if s:
                save_ks.add(int(s))
    save_ks = sorted(save_ks)

    # Output directory
    targets_dir = ds_dir / "solver_targets"
    targets_dir.mkdir(exist_ok=True)

    # Load flux assembler for coordinate conversion
    import torch
    torch.set_default_dtype(torch.float64)
    from pinn.flux_assembly import FluxAssembler
    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=n_cells,
        n_faces=n_faces,
    )

    U_SCALE = 0.1
    P_SCALE = 0.01
    PHI_SCALE = 4e-7

    # Find training topologies
    with open(ds_dir / "splits.json") as f:
        splits = json.load(f)
    train_topologies = splits["train"]

    print(f"[targets] {len(train_topologies)} training topologies")
    print(f"[targets] K_max={args.k_max}, K_primary={args.k_primary}")
    print(f"[targets] Saving at k={save_ks}")

    t0 = time.time()
    all_q_cell = {k: [] for k in k_targets_to_convert}
    all_q_phi = {k: [] for k in k_targets_to_convert}
    all_metrics = []

    for ti, tid in enumerate(train_topologies):
        topo_dir = ds_dir / tid
        case_dir = str(topo_dir / "case")
        out_dir = targets_dir / tid
        out_dir.mkdir(exist_ok=True)

        print(f"\n[targets] {tid} ({ti+1}/{len(train_topologies)})")

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
        states = {}

        for k in range(args.k_max + 1):
            if k > 0:
                w = bridge.simple_step(w)
            if k in save_ks:
                states[k] = w.copy()
                np.save(out_dir / f"state_k{k:03d}.npy", w)

    # Convert all saved K values to network coordinates
    k_targets_to_convert = sorted(set([args.k_primary, args.k_max] +
        [k for k in save_ks if k > 0]))

    for k_target in k_targets_to_convert:
        if k_target not in states:
            continue
        w_k = states[k_target]

        # q_cell: [n_cells, 3] = (dUx/U_S, dUy/U_S, dp/P_S)
        u_k = w_k[:n_u].reshape(n_cells, 3)
        u0 = w0[:n_u].reshape(n_cells, 3)
        p_k = w_k[n_u:n_u + n_p]
        p0 = w0[n_u:n_u + n_p]

        q_ux = (u_k[:, 0] - u0[:, 0]) / U_SCALE
        q_uy = (u_k[:, 1] - u0[:, 1]) / U_SCALE
        q_p = (p_k - p0) / P_SCALE
        q_cell = np.column_stack([q_ux, q_uy, q_p])

        # q_phi: independent correction on trainable faces
        phi_k = w_k[n_u + n_p:]
        phi0 = w0[n_u + n_p:]
        du = torch.from_numpy(u_k[:, :2] - u0[:, :2]).double()
        a_phi_du_full = np.zeros(n_faces, dtype=np.float64)
        a_phi_du_internal = flux_asm(du).numpy()
        a_phi_du_full[:n_internal] = a_phi_du_internal
        q_phi = (phi_k[phi_trainable_indices] - phi0[phi_trainable_indices]
                 - a_phi_du_full[phi_trainable_indices]) / PHI_SCALE

        np.save(out_dir / f"q_cell_k{k_target:03d}.npy", q_cell)
        np.save(out_dir / f"q_phi_k{k_target:03d}.npy", q_phi)

        all_q_cell[k_target].append(q_cell)
        all_q_phi[k_target].append(q_phi)

        # Diagnostic: field errors vs converged HFDIB
        ref_ux = np.load(topo_dir / "ux_hfdib.npy")
        ref_uy = np.load(topo_dir / "uy_hfdib.npy")
        ref_p = np.load(topo_dir / "pressure_hfdib.npy")

        u_k10 = states[args.k_primary][:n_u].reshape(n_cells, 3)
        p_k10 = states[args.k_primary][n_u:n_u + n_p]

        rel_u_k = np.sqrt(np.sum((ref_ux - u_k10[:, 0].reshape(64, 64))**2 +
                                  (ref_uy - u_k10[:, 1].reshape(64, 64))**2)) / \
                   (np.sqrt(np.sum(ref_ux**2 + ref_uy**2)) + 1e-30)
        rel_p_k = np.sqrt(np.sum((ref_p - p_k10.reshape(64, 64))**2)) / \
                   (np.sqrt(np.sum(ref_p**2)) + 1e-30)

        print(f"  K={args.k_primary}: rel_U={rel_u_k:.4e}  rel_p={rel_p_k:.4e}")

        metrics = {
            "topology_id": tid,
            "k_primary": args.k_primary,
            "rel_u_vs_converged": float(rel_u_k),
            "rel_p_vs_converged": float(rel_p_k),
        }
        all_metrics.append(metrics)

        # Save per-topology metrics
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        del bridge

    # ================================================================
    # Verify target representability for topology_000
    # ================================================================
    print("\n[targets] Verifying representability for topology_000...")
    tid = "topology_000"
    out_dir = targets_dir / tid

    w_k10 = np.load(out_dir / f"state_k{args.k_primary:03d}.npy")
    q_cell = np.load(out_dir / f"q_cell_k{args.k_primary:03d}.npy")
    q_phi = np.load(out_dir / f"q_phi_k{args.k_primary:03d}.npy")

    from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
    base_state = torch.from_numpy(w0)
    state_asm = IndependentPhiStateAssembler(
        base_state, layout, flux_asm, n_cells, n_faces,
        phi_trainable_indices)

    w_reconstructed = state_asm.assemble(
        torch.from_numpy(q_cell), torch.from_numpy(q_phi)).detach().numpy()

    rel_full = np.linalg.norm(w_k10 - w_reconstructed) / (
        np.linalg.norm(w_k10) + 1e-30)
    rel_u_r = np.linalg.norm(w_k10[:n_u] - w_reconstructed[:n_u]) / (
        np.linalg.norm(w_k10[:n_u]) + 1e-30)
    rel_p_r = np.linalg.norm(w_k10[n_u:n_u+n_p] - w_reconstructed[n_u:n_u+n_p]) / (
        np.linalg.norm(w_k10[n_u:n_u+n_p]) + 1e-30)
    rel_phi_r = np.linalg.norm(
        w_k10[n_u+n_p:][phi_trainable_indices] -
        w_reconstructed[n_u+n_p:][phi_trainable_indices]) / (
        np.linalg.norm(w_k10[n_u+n_p:][phi_trainable_indices]) + 1e-30)

    print(f"  rel_full = {rel_full:.2e}")
    print(f"  rel_U   = {rel_u_r:.2e}")
    print(f"  rel_p   = {rel_p_r:.2e}")
    print(f"  rel_phi = {rel_phi_r:.2e}")

    # Compute and save block normalization for each K
    for k_target in k_targets_to_convert:
        if not all_q_cell[k_target]:
            continue
        q_cell_arr = np.stack(all_q_cell[k_target])
        q_phi_arr = np.stack(all_q_phi[k_target])

        E_u = float(np.mean(q_cell_arr[:, :, :2] ** 2))
        E_p = float(np.mean(q_cell_arr[:, :, 2] ** 2))
        E_phi = float(np.mean(q_phi_arr ** 2))

        print(f"  K={k_target}: E_U={E_u:.6e} E_p={E_p:.6e} E_phi={E_phi:.6e}")

        norm = {
            "E_u": E_u, "E_p": E_p, "E_phi": E_phi,
            "definition": "mean squared dimensionless displacement from W0",
            "k": k_target,
        }
        with open(targets_dir / f"target_normalization_k{k_target:03d}.json", "w") as f:
            json.dump(norm, f, indent=2)

    # Global metadata
    metadata = {
        "target_type": "truncated_SIMPLE",
        "max_simple_steps": args.k_max,
        "primary_training_target": args.k_primary,
        "uses_converged_flow_labels_for_training": False,
        "initial_state": "shared/base_state_k0.npy",
        "solver": "DASimpleFoam",
        "hfdib": "signed-distance",
        "save_ks": save_ks,
        "n_training_topologies": len(train_topologies),
        "total_simple_calls": len(train_topologies) * args.k_max,
        "generation_time_s": time.time() - t0,
        "representability": {
            "rel_full": float(rel_full),
            "rel_u": float(rel_u_r),
            "rel_p": float(rel_p_r),
            "rel_phi": float(rel_phi_r),
        },
        "mean_teacher_quality": {
            "rel_u_vs_converged": float(np.mean([m["rel_u_vs_converged"] for m in all_metrics])),
            "rel_p_vs_converged": float(np.mean([m["rel_p_vs_converged"] for m in all_metrics])),
        },
    }
    with open(targets_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n[targets] Done in {elapsed:.0f}s")
    print(f"  {len(train_topologies)} topologies × {args.k_max} steps = "
          f"{len(train_topologies) * args.k_max} SIMPLE calls")
    print(f"  Mean teacher K={args.k_primary}: "
          f"rel_U={metadata['mean_teacher_quality']['rel_u_vs_converged']:.4e}  "
          f"rel_p={metadata['mean_teacher_quality']['rel_p_vs_converged']:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
