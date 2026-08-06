"""Generate converged HFDIB solutions for the four-port topology dataset.

For each topology:
  1. Copy the four_port_64x64 case
  2. Compute signed distance from mask
  3. Run blockMesh
  4. Run DAFoam primal solve to convergence
  5. Extract ux, uy, p fields
  6. Save as .npy arrays

Usage:
  python -m unet.generate_dataset --dataset topologies/four_port_64 --output datasets/four_port_64
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from unet.generate_case import mask_to_signed_distance_64, write_signed_distance_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--case-template", default="cases/four_port_64x64")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(PROJECT_ROOT) / output_dir
    case_template = args.case_template
    if not os.path.isabs(case_template):
        case_template = os.path.join(PROJECT_ROOT, case_template)

    # Discover topologies
    topo_dirs = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("topology_")])
    print(f"[dataset] {len(topo_dirs)} topologies found")

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, topo_dir in enumerate(topo_dirs):
        tid = topo_dir.name
        mask = np.load(topo_dir / "mask.npy")
        out_dir = output_dir / tid
        case_dir = out_dir / "case"

        print(f"\n[dataset] {tid} ({idx+1}/{len(topo_dirs)})")

        # Copy case template
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.copytree(case_template, case_dir)
        for d in os.listdir(case_dir):
            if d[0].isdigit() and d != "0":
                shutil.rmtree(os.path.join(case_dir, d), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "constant", "polyMesh"), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "postProcessing"), ignore_errors=True)

        # Run blockMesh
        subprocess.run(["blockMesh", "-case", case_dir], check=True, capture_output=True)

        # Compute and write signed distance
        psi = mask_to_signed_distance_64(mask)
        write_signed_distance_file(case_dir, psi)
        np.save(out_dir / "signed_distance.npy", psi)
        np.save(out_dir / "mask.npy", mask)

        # Compute lambda field
        h = np.sqrt((0.128/64) * (0.128/64))
        lam = 0.5 * (1.0 - np.tanh(psi / (1.5 * h)))
        np.save(out_dir / "lambda.npy", lam)

        # Run DAFoam primal solve to convergence
        os.chdir(case_dir)
        from mpi4py import MPI
        options = hfdib_signed_distance_options(case_dir)
        bridge = DAFoamResidualBridge(case_dir, options, comm=MPI.COMM_SELF)
        bridge.solver()

        # Extract fields from state vector
        # Layout: [U: 3*4096][p: 4096][phi: ...] (isothermal, no T)
        # Actually 64x64 = 4096 cells
        # State: [Ux0,Uy0,Uz0, Ux1,Uy1,Uz1, ...] [p0, p1, ...] [phi...]
        w = np.ascontiguousarray(bridge.solver.getStates().copy(), dtype=np.float64)
        r = bridge.residual(w)
        n_cells = 4096

        ux = w[0::3]  # [n_cells]
        uy = w[1::3]
        p_vals = w[3*n_cells : 4*n_cells]

        # Reshape to 64x64 (row-major: cell_id = i + j*64)
        # OpenFOAM blockMesh: x-fastest, so cell_id = i + j*NX
        ux_grid = ux.reshape(64, 64)  # [j, i] = [row, col]
        uy_grid = uy.reshape(64, 64)
        p_grid = p_vals.reshape(64, 64)

        np.save(out_dir / "ux_hfdib.npy", ux_grid)
        np.save(out_dir / "uy_hfdib.npy", uy_grid)
        np.save(out_dir / "pressure_hfdib.npy", p_grid)

        write_json(out_dir / "metadata.json", {
            "topology_id": tid,
            "residual_l2": float(np.linalg.norm(r)),
            "n_cells": n_cells,
            "grid_size": [64, 64],
            "converged": True,
        })

        print(f"[dataset] {tid}: ||R||={np.linalg.norm(r):.3e}")

    print(f"\n[dataset] done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
