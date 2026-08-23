"""Evaluate article-style metrics + CFD warm-start acceleration.

For each held-out test topology (016-019):
  1. Article metrics: continuity error, pressure-drop ratio, MSE-TV
  2. Warm-start: compare W0→T^k(W0) vs W_NN→T^k(W_NN) for k=0,1,2,5,10

Usage (inside container):
  python -m diagnostics.evaluate_warm_start \
    --dataset datasets/four_port_64 \
    --checkpoint outputs/solver_distilled_k20/checkpoint_s1000.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_isothermal_layout  # noqa: E402
from unet.generate_case import write_signed_distance_file  # noqa: E402


def compute_continuity_error(state, mesh_meta, n_u, n_p, n_internal):
    """Estimate global mass imbalance from phi residuals."""
    phi = state[n_u + n_p:]
    # Flux imbalance: sum of internal face flux divergences
    owners = mesh_meta.owners[:n_internal]
    neighbours = mesh_meta.neighbours
    div = np.zeros(mesh_meta.n_cells)
    for i in range(n_internal):
        div[owners[i]] += phi[i]
        div[neighbours[i]] -= phi[i]
    return float(np.sqrt(np.mean(div**2)))


def compute_pressure_drop(w, n_u, n_p, mesh_meta, patch_names):
    """Compute pressure drop from inlet to outlet."""
    p = w[n_u:n_u + n_p]

    # Average pressure on inlet vs outlet patches
    def patch_avg_p(pname):
        if pname not in patch_names:
            return 0.0
        idx = patch_names.index(pname)
        start = int(mesh_meta.patch_start_faces[idx])
        count = int(mesh_meta.patch_face_counts[idx])
        # For boundary patches, face values correspond to adjacent cells
        # Approximate: use owner cell pressure
        owners = mesh_meta.owners[start:start+count]
        return float(np.mean(p[owners]))

    p_inlet = (patch_avg_p("inletLower") + patch_avg_p("inletUpper")) / 2
    p_outlet = (patch_avg_p("outletLower") + patch_avg_p("outletUpper")) / 2
    return p_inlet - p_outlet


def rel_field_error(w, w_star, n_cells, n_u, n_p, phi_indices):
    u = w[:n_u].reshape(n_cells, 3)
    u_s = w_star[:n_u].reshape(n_cells, 3)
    p = w[n_u:n_u + n_p]
    p_s = w_star[n_u:n_u + n_p]
    phi = w[n_u + n_p:]
    phi_s = w_star[n_u + n_p:]

    rel_u = np.linalg.norm(u_s[:, :2] - u[:, :2]) / (
        np.linalg.norm(u_s[:, :2]) + 1e-30)
    rel_p = np.linalg.norm(p_s - p) / (np.linalg.norm(p_s) + 1e-30)
    return float(rel_u), float(rel_p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/four_port_64")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--simple-steps", default="0,1,2,5,10")
    ap.add_argument("--output", default="outputs/warm_start_evaluation/metrics.json")
    ap.add_argument("--topology-limit", type=int, default=None)
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
    patch_names = list(mesh_meta.patch_names)

    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")

    # Trainable phi indices
    phi_trainable = np.zeros(n_faces, dtype=bool)
    phi_trainable[:n_internal] = True
    for pname in ["outletLower", "outletUpper"]:
        if pname in patch_names:
            idx = patch_names.index(pname)
            start = int(mesh_meta.patch_start_faces[idx])
            count = int(mesh_meta.patch_face_counts[idx])
            phi_trainable[start:start + count] = True
    phi_trainable_indices = np.flatnonzero(phi_trainable)

    # Load NN model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path(PROJECT_ROOT) / ckpt_path
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    from unet.factory import build_model
    model = build_model(ckpt.get("architecture", "simple"),
                         **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    # Load test topologies
    with open(ds_dir / "splits.json") as f:
        splits = json.load(f)
    test_topologies = splits["test"]
    if args.topology_limit is not None:
        test_topologies = test_topologies[:args.topology_limit]
    simple_steps = [int(s) for s in args.simple_steps.split(",")]

    output = Path(args.output)
    if not output.is_absolute():
        output = Path(PROJECT_ROOT) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(output.read_text()) if output.exists() else {}

    for tid in test_topologies:
        if tid in results:
            print(f"Skipping completed {tid}", flush=True)
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating {tid}")
        print(f"{'='*60}")

        topo_dir = ds_dir / tid
        case_dir = str(topo_dir / "case")

        subprocess.run(["blockMesh", "-case", case_dir], check=True,
                       capture_output=True)
        write_signed_distance_file(
            case_dir, np.load(topo_dir / "signed_distance.npy"))

        # Reconstruct full converged state for rel_error
        # Use the bridge to get W*
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
        w_star = np.ascontiguousarray(
            bridge.solver.getStates().copy(), dtype=np.float64)

        # Run primal to get converged state
        bridge.solver()
        w_star = np.ascontiguousarray(
            bridge.solver.getStates().copy(), dtype=np.float64)
        w_teacher = np.load(ds_dir / "solver_targets" / tid / "state_k020.npy")
        teacher_rel_u, teacher_rel_p = rel_field_error(
            w_teacher, w_star, n_cells, n_u, n_p, phi_trainable_indices)

        # NN prediction
        lam = np.load(topo_dir / "lambda.npy")
        lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cell_pred, phi_pred = model(lam_t)
            cell_pred = project_solid_velocity(cell_pred, lam_t)
            corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            phi_corr = phi_pred.squeeze(0)
            w_nn = state_asm.assemble(corrections, phi_corr).detach().numpy()

        # ================================================================
        # 1. Article metrics
        # ================================================================
        print(f"\n[metrics] Article-style metrics for {tid}")

        # Continuity error
        cont_nn = compute_continuity_error(w_nn, mesh_meta, n_u, n_p, n_internal)
        cont_ref = compute_continuity_error(w_star, mesh_meta, n_u, n_p, n_internal)
        print(f"  Continuity error:  NN={cont_nn:.4e}  HFDIB={cont_ref:.4e}")

        # Pressure drop ratio
        dp_nn = compute_pressure_drop(w_nn, n_u, n_p, mesh_meta, patch_names)
        dp_ref = compute_pressure_drop(w_star, n_u, n_p, mesh_meta, patch_names)
        dp_ratio = dp_nn / (dp_ref + 1e-30)
        print(f"  Pressure drop:     NN={dp_nn:.4e}  HFDIB={dp_ref:.4e}  "
              f"ratio={dp_ratio:.4f}")

        # MSE-TV
        u_nn = w_nn[:n_u].reshape(n_cells, 3)
        u_ref = w_star[:n_u].reshape(n_cells, 3)
        p_nn = w_nn[n_u:n_u + n_p]
        p_ref = w_star[n_u:n_u + n_p]

        ux_nn_grid = u_nn[:, 0].reshape(64, 64)
        uy_nn_grid = u_nn[:, 1].reshape(64, 64)
        p_nn_grid = p_nn.reshape(64, 64)
        ux_ref_grid = u_ref[:, 0].reshape(64, 64)
        uy_ref_grid = u_ref[:, 1].reshape(64, 64)
        p_ref_grid = p_ref.reshape(64, 64)

        mse_u = float(np.mean((ux_nn_grid - ux_ref_grid)**2 +
                              (uy_nn_grid - uy_ref_grid)**2))
        mse_p = float(np.mean((p_nn_grid - p_ref_grid)**2))

        # TV of NN prediction
        tv_u = float(np.mean(np.abs(ux_nn_grid[:, 1:] - ux_nn_grid[:, :-1])) +
                     np.mean(np.abs(uy_nn_grid[:, 1:] - uy_nn_grid[:, :-1])))
        print(f"  MSE(U): {mse_u:.4e}  MSE(p): {mse_p:.4e}  TV(U): {tv_u:.4e}")

        rel_u_nn, rel_p_nn = rel_field_error(
            w_nn, w_star, n_cells, n_u, n_p, phi_trainable_indices)
        print(f"  rel_U={rel_u_nn:.4e}  rel_p={rel_p_nn:.4e}")

        topo_result = {
            "topology_id": tid,
            "article_metrics": {
                "continuity_error_nn": cont_nn,
                "continuity_error_ref": cont_ref,
                "pressure_drop_nn": dp_nn,
                "pressure_drop_ref": dp_ref,
                "pressure_drop_ratio": dp_ratio,
                "mse_u": mse_u,
                "mse_p": mse_p,
            },
            "rel_errors": {
                "rel_u": rel_u_nn,
                "rel_p": rel_p_nn,
            },
            "teacher_rel_errors": {
                "rel_u": teacher_rel_u,
                "rel_p": teacher_rel_p,
            },
        }

        # ================================================================
        # 2. Warm-start: W0→T^k vs W_NN→T^k
        # ================================================================
        print(f"\n[warm] CFD warm-start comparison for {tid}")

        warm_results = []

        for label, w_start in [("W0", w0.copy()), ("W_NN", w_nn.copy())]:
            w = w_start.copy()

            for k in range(max(simple_steps) + 1):
                if k > 0:
                    t0 = time.time()
                    w = bridge.simple_step(w)
                    elapsed = time.time() - t0
                else:
                    elapsed = 0.0

                if k in simple_steps:
                    rel_u, rel_p = rel_field_error(
                        w, w_star, n_cells, n_u, n_p,
                        phi_trainable_indices)
                    cont = compute_continuity_error(
                        w, mesh_meta, n_u, n_p, n_internal)
                    dp = compute_pressure_drop(
                        w, n_u, n_p, mesh_meta, patch_names)
                    dp_ratio_k = dp / (dp_ref + 1e-30)

                    print(f"  {label} k={k:2d}: rel_U={rel_u:.4e} "
                          f"rel_p={rel_p:.4e} cont={cont:.2e} "
                          f"dp_ratio={dp_ratio_k:.4f} t={elapsed:.1f}s")

                    warm_results.append({
                        "start": label,
                        "k": k,
                        "rel_u": rel_u,
                        "rel_p": rel_p,
                        "continuity": cont,
                        "dp_ratio": dp_ratio_k,
                        "time_s": elapsed,
                    })

        topo_result["warm_start"] = warm_results
        results[tid] = topo_result
        output.write_text(json.dumps(results, indent=2) + "\n")

        del bridge

    # ================================================================
    # Summary
    # ================================================================
    output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'topology':<14} {'rel_U':>10} {'rel_p':>10} "
          f"{'cont_NN':>10} {'dp_ratio':>10}")
    for tid in test_topologies:
        r = results[tid]
        am = r["article_metrics"]
        re = r["rel_errors"]
        print(f"{tid:<14} {re['rel_u']:>10.4e} {re['rel_p']:>10.4e} "
              f"{am['continuity_error_nn']:>10.4e} "
              f"{am['pressure_drop_ratio']:>10.4f}")

    print(f"\nWarm-start: rel_U after k SIMPLE steps")
    print(f"{'topology':<14} {'start':>6} {'k=0':>10} {'k=1':>10} "
          f"{'k=2':>10} {'k=5':>10} {'k=10':>10}")
    for tid in test_topologies:
        for label in ["W0", "W_NN"]:
            vals = [r for r in results[tid]["warm_start"] if r["start"] == label]
            row = f"{tid:<14} {label:>6}"
            for k in simple_steps:
                match = [v for v in vals if v["k"] == k]
                if match:
                    row += f" {match[0]['rel_u']:>10.4e}"
                else:
                    row += f" {'---':>10}"
            print(row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
