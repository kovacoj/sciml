"""Prepare dataset for fixed-budget study: blockMesh + signed distance for all
topologies, converged HFDIB only for test (128-143). No converged CFD for
training topologies.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, PROJECT_ROOT, write_json
from dafoam_bridge import DAFoamResidualBridge
from state_layout import build_isothermal_layout
from unet.generate_case import mask_to_signed_distance_64, write_signed_distance_file


def main():
    ds_dir = Path(PROJECT_ROOT) / "datasets" / "four_port_64"
    case_template = str(ds_dir / "case_template")

    # Load topology dirs
    topo_src = Path(PROJECT_ROOT) / "topologies" / "four_port_64"
    topo_dirs = sorted([d for d in topo_src.iterdir()
                        if d.is_dir() and d.name.startswith("topology_")])
    print(f"Found {len(topo_dirs)} topologies", flush=True)

    for idx, topo_dir in enumerate(topo_dirs):
        tid = topo_dir.name
        tid_num = int(tid.split("_")[1])
        mask = np.load(topo_dir / "mask.npy")
        out_dir = ds_dir / tid
        case_dir = out_dir / "case"

        is_test = 128 <= tid_num < 144
        has_hfdib = (out_dir / "ux_hfdib.npy").exists()
        has_mesh = (case_dir / "constant" / "polyMesh").exists()

        if has_mesh and (out_dir / "signed_distance.npy").exists():
            if is_test and has_hfdib:
                continue
            elif not is_test:
                continue

        print(f"[{idx+1}/{len(topo_dirs)}] {tid}...", flush=True)

        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.copytree(case_template, case_dir)
        for d in os.listdir(case_dir):
            if d[0].isdigit() and d != "0":
                shutil.rmtree(os.path.join(case_dir, d), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "constant", "polyMesh"),
                       ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "postProcessing"),
                       ignore_errors=True)

        subprocess.run(["blockMesh", "-case", case_dir], check=True,
                       capture_output=True)

        psi = mask_to_signed_distance_64(mask)
        write_signed_distance_file(case_dir, psi)
        np.save(out_dir / "signed_distance.npy", psi)
        np.save(out_dir / "mask.npy", mask)

        h = np.sqrt((0.128/64) * (0.128/64))
        lam = 0.5 * (1.0 - np.tanh(psi / (1.5 * h)))
        np.save(out_dir / "lambda.npy", lam)

        if is_test:
            print(f"  Running converged HFDIB for {tid}...", flush=True)
            os.chdir(case_dir)
            from mpi4py import MPI
            bridge = DAFoamResidualBridge(
                str(case_dir),
                hfdib_signed_distance_options(
                    str(case_dir),
                    inlet_patches=["inletLower", "inletUpper"],
                    outlet_patches=["outletLower", "outletUpper"]),
                comm=MPI.COMM_SELF)
            bridge.solver()
            w = np.ascontiguousarray(
                bridge.solver.getStates().copy(), dtype=np.float64)

            n_cells = 4096
            n_u = 3 * n_cells
            u_cells = w[:n_u].reshape(n_cells, 3)
            p_cells = w[n_u:n_u + n_cells]

            np.save(out_dir / "ux_hfdib.npy", u_cells[:, 0].reshape(64, 64))
            np.save(out_dir / "uy_hfdib.npy", u_cells[:, 1].reshape(64, 64))
            np.save(out_dir / "pressure_hfdib.npy", p_cells.reshape(64, 64))
            write_json(out_dir / "metadata.json",
                       {"topology_id": tid, "converged": True})
            del bridge
            os.chdir(str(PROJECT_ROOT))
        else:
            write_json(out_dir / "metadata.json",
                       {"topology_id": tid, "converged": False})

    # Build shared mesh metadata
    from unet.generate_dataset import build_64x64_mesh_metadata
    first_case = str(ds_dir / "topology_000" / "case")
    mesh_meta = build_64x64_mesh_metadata(first_case)
    mesh_meta.save(str(ds_dir / "shared" / "mesh_metadata.npz"),
                   str(ds_dir / "shared" / "mesh_metadata.json"))

    # Build base state k=0
    os.chdir(first_case)
    from mpi4py import MPI
    bridge_k0 = DAFoamResidualBridge(
        first_case,
        hfdib_signed_distance_options(
            first_case,
            inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"]),
        comm=MPI.COMM_SELF)
    w0 = np.ascontiguousarray(
        bridge_k0.solver.getStates().copy(), dtype=np.float64)
    np.save(ds_dir / "shared" / "base_state_k0.npy", w0)

    # Loss config
    r0 = bridge_k0.residual(w0)
    layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)
    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")
    lu = 0.5 * float(np.dot(r0[u_ids], r0[u_ids]))
    lp = 0.5 * float(np.dot(r0[p_ids], r0[p_ids]))
    lphi = 0.5 * float(np.dot(r0[phi_ids], r0[phi_ids]))
    loss_config = {"gamma_u": 1.0 / (lu + 1e-30),
                   "gamma_p": 1.0 / (lp + 1e-30),
                   "gamma_phi": 1.0 / (lphi + 1e-30)}
    with open(ds_dir / "shared" / "physics_loss_config.json", "w") as f:
        json.dump(loss_config, f, indent=2)

    print(f"Done: {len(topo_dirs)} topologies prepared")
    print(f"  Test (128-143): converged HFDIB")
    print(f"  Train: blockMesh + signed distance only")


if __name__ == "__main__":
    main()
