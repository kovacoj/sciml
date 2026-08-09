"""Generate converged HFDIB solutions for the four-port topology dataset.

Phase A: Create all 6 cases, run blockMesh, write signed distance, save lambda
Phase B: Use topology_000 to build shared mesh metadata + k=0 base state + loss config
Phase C: Run converged HFDIB solve for each topology, extract reference fields
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
from state_layout import build_isothermal_layout  # noqa: E402
from unet.generate_case import mask_to_signed_distance_64, write_signed_distance_file  # noqa: E402


def _read_int_list(path):
    with open(path) as f:
        lines = f.readlines()
    start = end = None
    for i, l in enumerate(lines):
        if "(" in l and start is None: start = i + 1
        if ")" in l and start is not None: end = i; break
    return np.array([int(l.strip()) for l in lines[start:end]], dtype=int)


def _read_boundary(path):
    import re
    with open(path) as f:
        content = f.read()
    patches = []
    pattern = r"(\w+)\s*\n\s*\{[^}]*nFaces\s+(\d+);[^}]*startFace\s+(\d+);"
    for m in re.finditer(pattern, content):
        patches.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return patches


def build_64x64_mesh_metadata(case_dir: str):
    """Build mesh metadata for the 64x64 structured four-port case."""
    import hashlib
    poly = os.path.join(case_dir, "constant", "polyMesh")
    for f in ["owner", "neighbour", "boundary"]:
        gz = os.path.join(poly, f + ".gz")
        asc = os.path.join(poly, f)
        if not os.path.exists(asc) and os.path.exists(gz):
            import gzip
            with gzip.open(gz, "rt") as gz_f:
                with open(asc, "w") as out_f:
                    out_f.write(gz_f.read())

    owners = _read_int_list(os.path.join(poly, "owner"))
    neighbours = _read_int_list(os.path.join(poly, "neighbour"))
    n_internal = len(neighbours)
    n_faces = len(owners)
    n_cells = int(owners.max()) + 1

    nx, ny, nz = 64, 64, 1
    dx, dy, dz = 0.128 / nx, 0.128 / ny, 0.002

    cell_centres = np.zeros((n_cells, 3))
    cell_volumes = np.full(n_cells, dx * dy * dz)
    cell_to_grid = np.zeros((n_cells, 2), dtype=int)
    for j in range(ny):
        for i in range(nx):
            cid = i + j * nx
            cell_centres[cid] = [i * dx + dx/2, j * dy + dy/2, dz/2]
            cell_to_grid[cid] = [j, i]

    # Derive face area vectors from actual owner-neighbour geometry
    internal_owners = owners[:n_internal]
    owner_centres = cell_centres[internal_owners]
    neighbour_centres = cell_centres[neighbours]
    deltas = neighbour_centres - owner_centres

    sf_int = np.zeros((n_internal, 3), dtype=np.float64)
    x_faces = np.abs(deltas[:, 0]) > np.abs(deltas[:, 1])
    y_faces = ~x_faces
    sf_int[x_faces, 0] = np.sign(deltas[x_faces, 0]) * dy * dz
    sf_int[y_faces, 1] = np.sign(deltas[y_faces, 1]) * dx * dz

    # Verify cell ordering matches polyMesh connectivity
    owner_grid = cell_to_grid[internal_owners]
    neighbour_grid = cell_to_grid[neighbours]
    grid_distance = np.abs(neighbour_grid - owner_grid).sum(axis=1)
    if not np.all(grid_distance == 1):
        bad = np.flatnonzero(grid_distance != 1)[:10]
        raise RuntimeError(
            "The assumed 64x64 cell ordering does not match "
            f"polyMesh connectivity; first bad faces: {bad.tolist()}")

    owner_weights = np.full(n_internal, 0.5)

    patches = _read_boundary(os.path.join(poly, "boundary"))
    patch_names = tuple(p[0] for p in patches)
    patch_start = np.array([p[2] for p in patches], dtype=int)
    patch_counts = np.array([p[1] for p in patches], dtype=int)

    h = hashlib.sha256()
    h.update(owners.tobytes())
    h.update(neighbours.tobytes())
    mesh_hash = h.hexdigest()[:16]

    # Build a simple dataclass-compatible dict
    from pinn.mesh_metadata import MeshMetadata
    return MeshMetadata(
        n_cells=n_cells, n_internal_faces=n_internal, n_faces=n_faces,
        owners=owners, neighbours=neighbours, face_area_vectors=sf_int,
        owner_weights=owner_weights, cell_centres=cell_centres,
        cell_volumes=cell_volumes, cell_to_grid=cell_to_grid,
        patch_names=patch_names, patch_start_faces=patch_start,
        patch_face_counts=patch_counts, mesh_hash=mesh_hash,
    )


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

    topo_dirs = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("topology_")])
    print(f"[dataset] {len(topo_dirs)} topologies found")

    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = output_dir / "shared"
    shared_dir.mkdir(exist_ok=True)

    # Write splits.json
    train_ids = [d.name for d in topo_dirs[:16]]
    test_ids = [d.name for d in topo_dirs[16:]]
    with open(output_dir / "splits.json", "w") as f:
        json.dump({"train": train_ids, "test": test_ids}, f, indent=2)
    print(f"[dataset] train: {train_ids}")
    print(f"[dataset] test:  {test_ids}")

    # ---- Phase A: Create all cases, blockMesh, signed distance, lambda ----
    for idx, topo_dir in enumerate(topo_dirs):
        tid = topo_dir.name
        mask = np.load(topo_dir / "mask.npy")
        out_dir = output_dir / tid
        case_dir = out_dir / "case"

        print(f"\n[A] {tid} ({idx+1}/{len(topo_dirs)})")
        shutil.rmtree(out_dir, ignore_errors=True)
        shutil.copytree(case_template, case_dir)
        for d in os.listdir(case_dir):
            if d[0].isdigit() and d != "0":
                shutil.rmtree(os.path.join(case_dir, d), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "constant", "polyMesh"), ignore_errors=True)
        shutil.rmtree(os.path.join(case_dir, "postProcessing"), ignore_errors=True)

        subprocess.run(["blockMesh", "-case", case_dir], check=True, capture_output=True)

        psi = mask_to_signed_distance_64(mask)
        write_signed_distance_file(case_dir, psi)
        np.save(out_dir / "signed_distance.npy", psi)
        np.save(out_dir / "mask.npy", mask)

        h = np.sqrt((0.128/64) * (0.128/64))
        lam = 0.5 * (1.0 - np.tanh(psi / (1.5 * h)))
        np.save(out_dir / "lambda.npy", lam)
        print(f"[A] {tid}: psi range [{psi.min():.4f}, {psi.max():.4f}]")

    # ---- Phase B: Shared mesh metadata + k=0 base state + loss config ----
    print("\n[B] Building shared data...")
    first_case_dir = str(output_dir / topo_dirs[0].name / "case")
    mesh_meta = build_64x64_mesh_metadata(first_case_dir)
    mesh_meta.save(str(shared_dir / "mesh_metadata.npz"),
                  str(shared_dir / "mesh_metadata.json"))
    print(f"[B] n_cells={mesh_meta.n_cells}, n_int={mesh_meta.n_internal_faces}, "
          f"n_faces={mesh_meta.n_faces}")

    os.chdir(first_case_dir)
    from mpi4py import MPI
    bridge_k0 = DAFoamResidualBridge(
        first_case_dir,
        hfdib_signed_distance_options(first_case_dir,
            inlet_patches=["inletLower","inletUpper"],
            outlet_patches=["outletLower","outletUpper"]),
        comm=MPI.COMM_SELF,
    )
    w0 = np.ascontiguousarray(bridge_k0.solver.getStates().copy(), dtype=np.float64)
    np.save(shared_dir / "base_state_k0.npy", w0)

    r0 = bridge_k0.residual(w0)
    layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)
    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")

    lu = 0.5 * float(np.dot(r0[u_ids], r0[u_ids]))
    lp = 0.5 * float(np.dot(r0[p_ids], r0[p_ids]))
    lphi = 0.5 * float(np.dot(r0[phi_ids], r0[phi_ids]))

    loss_config = {
        "gamma_u": 1.0 / (lu + 1e-30),
        "gamma_p": 1.0 / (lp + 1e-30),
        "gamma_phi": 1.0 / (lphi + 1e-30),
    }
    with open(shared_dir / "physics_loss_config.json", "w") as f:
        json.dump(loss_config, f, indent=2)
    print(f"[B] k=0 state: ||R||={np.linalg.norm(r0):.3e}")
    print(f"[B] loss config: {loss_config}")

    # ---- Phase C: Converged HFDIB solve + field extraction ----
    for idx, topo_dir in enumerate(topo_dirs):
        tid = topo_dir.name
        out_dir = output_dir / tid
        case_dir = out_dir / "case"

        print(f"\n[C] {tid} ({idx+1}/{len(topo_dirs)})")
        os.chdir(case_dir)
        bridge = DAFoamResidualBridge(
            case_dir,
            hfdib_signed_distance_options(case_dir,
                inlet_patches=["inletLower","inletUpper"],
                outlet_patches=["outletLower","outletUpper"]),
            comm=MPI.COMM_SELF,
        )
        bridge.solver()

        w = np.ascontiguousarray(bridge.solver.getStates().copy(), dtype=np.float64)
        r = bridge.residual(w)

        n_cells = mesh_meta.n_cells
        n_u = 3 * n_cells
        u_cells = w[:n_u].reshape(n_cells, 3)
        p_cells = w[n_u:n_u + n_cells]

        assert u_cells.shape == (4096, 3), f"u_cells shape {u_cells.shape}"
        assert p_cells.shape == (4096,), f"p_cells shape {p_cells.shape}"

        ux_grid = u_cells[:, 0].reshape(64, 64)
        uy_grid = u_cells[:, 1].reshape(64, 64)
        p_grid = p_cells.reshape(64, 64)

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
        print(f"[C] {tid}: ||R||={np.linalg.norm(r):.3e}")

    print(f"\n[dataset] done: {output_dir}")
    print(f"  {len(topo_dirs)} cases generated")
    print(f"  {mesh_meta.n_cells} cells per case")
    print(f"  {mesh_meta.n_cells} signed-distance values per case")
    print(f"  {len(train_ids)} train IDs, {len(test_ids)} test IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
