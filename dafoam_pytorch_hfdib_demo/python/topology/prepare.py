"""Prepare topology cases: copy, mesh, signed distance, warm start."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import (isothermal_channel_options, hfdib_signed_distance_options,  # noqa: E402
                     write_json, PROJECT_ROOT)
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from pinn.mesh_metadata import build_from_polymesh as build_structured_duct_40x16_metadata  # noqa: E402
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.state_assembly import StateAssembler  # noqa: E402
from topology.specification import load_dataset  # noqa: E402
from topology.signed_distance import (mask_to_signed_distance,  # noqa: E402
    compute_geometry_fields, write_openfoam_scalar_list)
from topology.features import build_features  # noqa: E402


def run_partial_primal(bridge, k, case_dir):
    """Run exactly k primal iterations using endTime control."""
    import re
    cd_path = os.path.join(case_dir, "system", "controlDict")
    with open(cd_path) as f:
        s = f.read()
    s = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {max(k, 1)};", s)
    with open(cd_path, "w") as f:
        f.write(s)

    opts = bridge.solver.getOption("")
    bridge.solver.solverAD.calcPrimalResidualStatistics("calc")
    if k > 0:
        bridge.solver()
    w = np.ascontiguousarray(
        bridge.solver.getStates().copy(), dtype=np.float64)
    r = bridge.residual(w)
    return w, r


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
    layout = build_state_layout("isothermal")

    for idx, spec in enumerate(specs):
        topo_id = spec.topology_id
        topo_dir = output_dir / topo_id
        case_dir = topo_dir / "case"
        print(f"\n[prepare] {topo_id} ({idx+1}/{len(specs)})")

        # Copy case template
        shutil.rmtree(topo_dir, ignore_errors=True)
        shutil.copytree(case_template, case_dir)
        # Clean generated dirs (keep 0/)
        for d in os.listdir(case_dir):
            if d[0].isdigit() and d != "0":
                shutil.rmtree(os.path.join(case_dir, d), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "constant", "polyMesh"), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "postProcessing"), ignore_errors=True)

        # Run blockMesh
        import subprocess
        subprocess.run(["blockMesh", "-case", case_dir], check=True,
                       capture_output=True)

        # Build mesh metadata (shared across topologies)
        mesh_meta = build_structured_duct_40x16_metadata(case_dir)
        if idx == 0:
            mesh_meta.save(str(shared_mesh_dir / "mesh_metadata.npz"),
                           str(shared_mesh_dir / "mesh_metadata.json"))

        # Compute signed distance
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

        # Write signed-distance file
        sd_path = os.path.join(case_dir, "constant", "hfdibGeometry", "signedDistance")
        write_openfoam_scalar_list(Path(sd_path), psi)

        # Save topology data
        np.save(topo_dir / "mask.npy", spec.mask)
        np.save(topo_dir / "signed_distance.npy", psi)
        np.save(topo_dir / "solid_mask.npy", geo["chi"])
        np.save(topo_dir / "interface_mask.npy", geo["interface"])

        # Initialize DAFoam with hfdibSignedDistance
        os.chdir(case_dir)
        bridge = DAFoamResidualBridge(
            case_dir, hfdib_signed_distance_options(case_dir))

        # Run partial primal
        w_k, r_k = run_partial_primal(bridge, args.k, case_dir)
        np.save(topo_dir / f"warm_state_k{args.k}.npy", w_k)

        # Compute loss weights
        u_ids = layout.indices("U")
        p_ids = layout.indices("p")
        phi_ids = layout.indices("phi")
        lu = 0.5 * float((r_k[u_ids] * r_k[u_ids]).sum())
        lp = 0.5 * float((r_k[p_ids] * r_k[p_ids]).sum())
        lphi = 0.5 * float((r_k[phi_ids] * r_k[phi_ids]).sum())
        loss_config = {
            "gamma_u": 1.0 / (lu + 1e-30),
            "gamma_p": 1.0 / (lp + 1e-30),
            "gamma_phi": 1.0 / (lphi + 1e-30),
        }
        with open(topo_dir / "loss_config.json", "w") as f:
            json.dump(loss_config, f, indent=2)

        # Build CNN features
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

        # Save preparation metadata
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
