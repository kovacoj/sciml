#!/usr/bin/env python3
"""Prepare topology cases: copy, mesh, signed distance, warm start.

Each topology's DAFoam warm-start runs in an isolated subprocess.
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

from common import write_json, PROJECT_ROOT  # noqa: E402
from pinn.mesh_metadata import build_from_polymesh as build_mesh_meta  # noqa: E402
from topology.specification import load_dataset  # noqa: E402
from topology.signed_distance import (mask_to_signed_distance,  # noqa: E402
    compute_geometry_fields, write_openfoam_scalar_list)
from topology.features import build_features  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--case-template", default="cases/single_obstacle")
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, default=8)
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

    specs, meta = load_dataset(dataset_dir)
    print(f"[prepare] {len(specs)} topologies found")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(meta, f, indent=2)

    shared_mesh_dir = output_dir / "shared_mesh"
    shared_mesh_dir.mkdir(exist_ok=True)

    domain_bounds = (0.0, 1.0, 0.0, 0.1)
    u_ref = 0.2
    p_ref = max(u_ref ** 2, 1e-8)

    for idx, spec in enumerate(specs):
        topo_id = spec.topology_id
        topo_dir = output_dir / topo_id
        case_dir = topo_dir / "case"
        print(f"\n[prepare] {topo_id} ({idx+1}/{len(specs)})")

        # 1. Copy case template
        shutil.rmtree(topo_dir, ignore_errors=True)
        shutil.copytree(case_template, case_dir)
        for d in os.listdir(case_dir):
            if d[0].isdigit() and d != "0":
                shutil.rmtree(os.path.join(case_dir, d), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "constant", "polyMesh"), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "postProcessing"), ignore_errors=True)

        # 2. Run blockMesh
        subprocess.run(["blockMesh", "-case", case_dir], check=True,
                       capture_output=True)

        # 3. Build mesh metadata (shared)
        mesh_meta = build_mesh_meta(case_dir)
        if idx == 0:
            mesh_meta.save(str(shared_mesh_dir / "mesh_metadata.npz"),
                           str(shared_mesh_dir / "mesh_metadata.json"))

        # 4. Compute signed distance
        psi = mask_to_signed_distance(
            mask=spec.mask,
            cell_to_grid=mesh_meta.cell_to_grid,
            grid_shape=(mesh_meta.n_cells // 40, 40),
            design_bounds=(spec.design_x_min, spec.design_x_max,
                          spec.design_y_min, spec.design_y_max),
            domain_bounds=domain_bounds,
        )

        nx, ny = 40, mesh_meta.n_cells // 40
        dx = (domain_bounds[1] - domain_bounds[0]) / nx
        dy = (domain_bounds[3] - domain_bounds[2]) / ny
        h = np.sqrt(dx * dy)

        geo = compute_geometry_fields(psi, h)

        # 5. Write signed-distance file
        sd_path = os.path.join(case_dir, "constant", "hfdibGeometry", "signedDistance")
        write_openfoam_scalar_list(Path(sd_path), psi)

        # Save topology data
        np.save(topo_dir / "mask.npy", spec.mask)
        np.save(topo_dir / "signed_distance.npy", psi)
        np.save(topo_dir / "solid_mask.npy", geo["chi"])
        np.save(topo_dir / "interface_mask.npy", geo["interface"])

        # 6. Launch prepare_worker subprocess for warm start
        warm_path = topo_dir / f"warm_state_k{args.k}.npy"
        res_path = topo_dir / f"residual_k{args.k}.npy"
        loss_path = topo_dir / "loss_config.json"

        subprocess.run(
            [
                sys.executable, "-m", "topology.prepare_worker",
                "--case", str(case_dir),
                "--k", str(args.k),
                "--warm-state", str(warm_path),
                "--residual", str(res_path),
                "--loss-config", str(loss_path),
            ],
            check=True,
            cwd=str(PYTHON_ROOT),
        )

        # 7. Load warm state + residual
        w_k = np.load(warm_path)
        r_k = np.load(res_path)

        # 8. Build CNN features
        features = build_features(
            cell_centres=mesh_meta.cell_centres,
            cell_to_grid=mesh_meta.cell_to_grid,
            grid_shape=(ny, nx),
            psi=psi,
            lam=geo["lambda"],
            chi=geo["chi"],
            interface=geo["interface"],
            warm_state=w_k,
            u_ref=u_ref,
            p_ref=p_ref,
            domain_bounds=domain_bounds,
        )
        np.save(topo_dir / "features.npy", features)

        # 9. Save preparation metadata
        write_json(topo_dir / "preparation.json", {
            "topology_id": topo_id,
            "k": args.k,
            "warm_state_residual_l2": float(np.linalg.norm(r_k)),
            "mesh_hash": mesh_meta.mesh_hash,
            "n_cells": mesh_meta.n_cells,
            "h": h,
        })

        print(f"[prepare] {topo_id}: ||R_k{args.k}||={np.linalg.norm(r_k):.3e}")

    print(f"\n[prepare] done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
