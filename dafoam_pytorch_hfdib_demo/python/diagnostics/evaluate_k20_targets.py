"""Compute K=20 teacher quality, block normalization, and topology-independence baseline.

No DAFoam calls — uses precomputed state_k020.npy files only.

Usage (inside container):
  python -m diagnostics.evaluate_k20_targets --dataset datasets/four_port_64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import PROJECT_ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/four_port_64")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    ds_dir = Path(args.dataset)
    if not ds_dir.is_absolute():
        ds_dir = Path(PROJECT_ROOT) / ds_dir

    k = args.k
    targets_dir = ds_dir / "solver_targets"

    with open(ds_dir / "splits.json") as f:
        splits = json.load(f)
    train_topologies = splits["train"]
    test_topologies = splits["test"]

    # Load shared data
    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")
    n_cells = 4096
    n_u = 3 * n_cells
    n_p = n_cells

    U_SCALE = 0.1
    P_SCALE = 0.01
    PHI_SCALE = 4e-7

    # ================================================================
    # 1. Teacher quality: K=k vs converged HFDIB
    # ================================================================
    print(f"\n[eval] K={k} teacher quality vs converged HFDIB")
    print(f"{'topology':<16} {'rel_U':>12} {'rel_p':>12}")

    teacher_results = []
    all_q_cell = []
    all_q_phi = []

    for tid in train_topologies:
        tdir = targets_dir / tid
        w_k = np.load(tdir / f"state_k{k:03d}.npy")
        q_cell = np.load(tdir / f"q_cell_k{k:03d}.npy")
        q_phi = np.load(tdir / f"q_phi_k{k:03d}.npy")

        ref_ux = np.load(ds_dir / tid / "ux_hfdib.npy")
        ref_uy = np.load(ds_dir / tid / "uy_hfdib.npy")
        ref_p = np.load(ds_dir / tid / "pressure_hfdib.npy")

        u_k = w_k[:n_u].reshape(n_cells, 3)
        p_k = w_k[n_u:n_u + n_p]

        rel_u = np.sqrt(
            np.sum((ref_ux - u_k[:, 0].reshape(64, 64))**2 +
                   (ref_uy - u_k[:, 1].reshape(64, 64))**2)) / \
            (np.sqrt(np.sum(ref_ux**2 + ref_uy**2)) + 1e-30)
        rel_p = np.sqrt(np.sum((ref_p - p_k.reshape(64, 64))**2)) / \
                (np.sqrt(np.sum(ref_p**2)) + 1e-30)

        print(f"{tid:<16} {rel_u:>12.4e} {rel_p:>12.4e}")

        teacher_results.append({
            "topology_id": tid,
            "rel_u": float(rel_u),
            "rel_p": float(rel_p),
        })
        all_q_cell.append(q_cell)
        all_q_phi.append(q_phi)

    rel_us = [r["rel_u"] for r in teacher_results]
    rel_ps = [r["rel_p"] for r in teacher_results]

    print(f"\n{'mean':<16} {np.mean(rel_us):>12.4e} {np.mean(rel_ps):>12.4e}")
    print(f"{'median':<16} {np.median(rel_us):>12.4e} {np.median(rel_ps):>12.4e}")
    print(f"{'min':<16} {np.min(rel_us):>12.4e} {np.min(rel_ps):>12.4e}")
    print(f"{'max':<16} {np.max(rel_us):>12.4e} {np.max(rel_ps):>12.4e}")

    with open(targets_dir / f"teacher_quality_k{k:03d}.json", "w") as f:
        json.dump({
            "k": k,
            "results": teacher_results,
            "mean": {"rel_u": float(np.mean(rel_us)), "rel_p": float(np.mean(rel_ps))},
            "median": {"rel_u": float(np.median(rel_us)), "rel_p": float(np.median(rel_ps))},
            "min": {"rel_u": float(np.min(rel_us)), "rel_p": float(np.min(rel_ps))},
            "max": {"rel_u": float(np.max(rel_us)), "rel_p": float(np.max(rel_ps))},
        }, f, indent=2)

    # ================================================================
    # 2. Block normalization
    # ================================================================
    q_cell_arr = np.stack(all_q_cell)  # [n_topo, n_cells, 3]
    q_phi_arr = np.stack(all_q_phi)    # [n_topo, n_phi_trainable]

    E_u = float(np.mean(q_cell_arr[:, :, :2] ** 2))
    E_p = float(np.mean(q_cell_arr[:, :, 2] ** 2))
    E_phi = float(np.mean(q_phi_arr ** 2))

    print(f"\n[eval] K={k} block normalization:")
    print(f"  E_U   = {E_u:.6e}")
    print(f"  E_p   = {E_p:.6e}")
    print(f"  E_phi = {E_phi:.6e}")

    norm = {
        "E_u": E_u,
        "E_p": E_p,
        "E_phi": E_phi,
        "definition": "mean squared dimensionless displacement from W0",
        "k": k,
    }
    with open(targets_dir / f"target_normalization_k{k:03d}.json", "w") as f:
        json.dump(norm, f, indent=2)

    # ================================================================
    # 3. Topology-independence baseline: mean K=k target
    # ================================================================
    print(f"\n[eval] Topology-independence check:")
    print(f"  Computing mean K={k} target across {len(train_topologies)} training topologies...")

    mean_q_cell = np.mean(q_cell_arr, axis=0)  # [n_cells, 3]
    mean_u = w0[:n_u].reshape(n_cells, 3).copy()
    mean_u[:, 0] += U_SCALE * mean_q_cell[:, 0]
    mean_u[:, 1] += U_SCALE * mean_q_cell[:, 1]
    mean_p = w0[n_u:n_u + n_p] + P_SCALE * mean_q_cell[:, 2]

    print(f"\n  {'topology':<16} {'NN rel_U':>12} {'mean-tgt rel_U':>16} {'W0 rel_U':>12}")
    print(f"  {'(test topologies)':<16}")

    # For test topologies, we need the converged refs
    mean_baseline_results = []
    for tid in test_topologies:
        ref_ux = np.load(ds_dir / tid / "ux_hfdib.npy")
        ref_uy = np.load(ds_dir / tid / "uy_hfdib.npy")

        rel_u_mean = np.sqrt(
            np.sum((ref_ux - mean_u[:, 0].reshape(64, 64))**2 +
                   (ref_uy - mean_u[:, 1].reshape(64, 64))**2)) / \
            (np.sqrt(np.sum(ref_ux**2 + ref_uy**2)) + 1e-30)

        rel_u_w0 = 1.0  # by definition

        print(f"  {tid:<16} {'(see eval)':>12} {rel_u_mean:>16.4e} {rel_u_w0:>12.4e}")
        mean_baseline_results.append({
            "topology_id": tid,
            "mean_target_rel_u": float(rel_u_mean),
            "w0_rel_u": 1.0,
        })

    with open(targets_dir / f"topology_independence_k{k:03d}.json", "w") as f:
        json.dump({
            "k": k,
            "mean_target_test_results": mean_baseline_results,
            "mean_test_rel_u_mean": float(np.mean([r["mean_target_rel_u"] for r in mean_baseline_results])),
        }, f, indent=2)

    print(f"\n  Mean test rel_U (mean-target baseline): "
          f"{np.mean([r['mean_target_rel_u'] for r in mean_baseline_results]):.4e}")
    print(f"  (NN must beat this to prove topology-sensitivity)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
