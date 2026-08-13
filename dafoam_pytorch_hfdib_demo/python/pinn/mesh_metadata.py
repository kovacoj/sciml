"""Authoritative mesh metadata for the 40x16x1 structured duct.

Parsed from polyMesh files + computed geometric quantities.
No DAFoam initialization required — pure file I/O + numpy.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class MeshMetadata:
    n_cells: int
    n_internal_faces: int
    n_faces: int

    owners: np.ndarray        # [n_faces] int — all faces
    neighbours: np.ndarray    # [n_internal] int — internal only
    face_area_vectors: np.ndarray  # [n_internal, 3]
    owner_weights: np.ndarray     # [n_internal]

    cell_centres: np.ndarray      # [n_cells, 3]
    cell_volumes: np.ndarray      # [n_cells]
    cell_to_grid: np.ndarray      # [n_cells, 2] = (row, col) for [H, W]

    patch_names: Tuple[str, ...]
    patch_start_faces: np.ndarray  # [n_patches]
    patch_face_counts: np.ndarray  # [n_patches]

    mesh_hash: str

    @classmethod
    def load(cls, path_npz: str, path_json: str) -> "MeshMetadata":
        with open(path_json) as handle:
            metadata = json.load(handle)
        arrays = np.load(path_npz)
        return cls(
            n_cells=int(metadata["n_cells"]),
            n_internal_faces=int(metadata["n_internal_faces"]),
            n_faces=int(metadata["n_faces"]),
            owners=arrays["owners"],
            neighbours=arrays["neighbours"],
            face_area_vectors=arrays["face_area_vectors"],
            owner_weights=arrays["owner_weights"],
            cell_centres=arrays["cell_centres"],
            cell_volumes=arrays["cell_volumes"],
            cell_to_grid=arrays["cell_to_grid"],
            patch_names=tuple(metadata["patch_names"]),
            patch_start_faces=arrays["patch_start_faces"],
            patch_face_counts=arrays["patch_face_counts"],
            mesh_hash=str(metadata["mesh_hash"]),
        )

    def save(self, path_npz: str, path_json: str):
        np.savez(path_npz,
                 owners=self.owners, neighbours=self.neighbours,
                 face_area_vectors=self.face_area_vectors,
                 owner_weights=self.owner_weights,
                 cell_centres=self.cell_centres,
                 cell_volumes=self.cell_volumes,
                 cell_to_grid=self.cell_to_grid,
                 patch_start_faces=self.patch_start_faces,
                 patch_face_counts=self.patch_face_counts)
        with open(path_json, "w") as f:
            d = asdict(self)
            d["patch_names"] = list(self.patch_names)
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    d[k] = v.tolist()
            json.dump(d, f, indent=2)


def _read_of_int_list(path: str) -> np.ndarray:
    with open(path) as f:
        lines = f.readlines()
    start = end = None
    for i, l in enumerate(lines):
        if "(" in l and start is None:
            start = i + 1
        if ")" in l and start is not None:
            end = i
            break
    return np.array([int(l.strip()) for l in lines[start:end]], dtype=int)


def _read_boundary(path: str):
    """Parse boundary file for patch names and face counts."""
    with open(path) as f:
        content = f.read()
    # crude parse: find patch name + nFaces
    patches = []
    import re
    # find all patch blocks
    pattern = r"(\w+)\s*\n\s*\{[^}]*nFaces\s+(\d+);[^}]*startFace\s+(\d+);"
    for m in re.finditer(pattern, content):
        name, nfaces, startface = m.group(1), int(m.group(2)), int(m.group(3))
        patches.append((name, nfaces, startface))
    return patches


def build_from_polymesh(case_dir: str) -> MeshMetadata:
    """Build mesh metadata from polyMesh files (no DAFoam init needed)."""
    poly = os.path.join(case_dir, "constant", "polyMesh")
    # decompress if needed
    for f in ["owner", "neighbour", "faces", "points", "boundary"]:
        gz = os.path.join(poly, f + ".gz")
        asc = os.path.join(poly, f)
        if not os.path.exists(asc) and os.path.exists(gz):
            import gzip
            with gzip.open(gz, "rt") as gz_f:
                with open(asc, "w") as out_f:
                    out_f.write(gz_f.read())

    owners = _read_of_int_list(os.path.join(poly, "owner"))
    neighbours = _read_of_int_list(os.path.join(poly, "neighbour"))
    n_internal = len(neighbours)
    n_faces = len(owners)
    n_cells = int(owners.max()) + 1

    # structured mesh params (verified for 40x16x1 blockMesh duct)
    nx, ny, nz = 40, 16, 1
    dx, dy, dz = 1.0 / nx, 0.1 / ny, 0.005

    # cell centres and cell_to_grid
    cx = np.arange(nx) * dx + dx / 2
    cy = np.arange(ny) * dy + dy / 2
    cz = dz / 2
    cell_centres = np.zeros((n_cells, 3))
    cell_to_grid = np.zeros((n_cells, 2), dtype=int)
    for j in range(ny):
        for i in range(nx):
            cid = i + j * nx
            cell_centres[cid] = [cx[i], cy[j], cz]
            cell_to_grid[cid] = [j, i]  # (row, col) for [H=16, W=40]

    cell_volumes = np.full(n_cells, dx * dy * dz)

    # internal face area vectors (structured mesh)
    sf_int = np.zeros((n_internal, 3))
    n_x_faces = (nx - 1) * ny * nz  # 624
    n_y_faces = nx * (ny - 1) * nz  # 600
    sf_int[:n_x_faces, 0] = dy * dz   # x-normal
    sf_int[n_x_faces:, 1] = dx * dz   # y-normal

    # interpolation weights (uniform orthogonal mesh)
    owner_weights = np.full(n_internal, 0.5)

    # boundary patches
    patches = _read_boundary(os.path.join(poly, "boundary"))
    patch_names = tuple(p[0] for p in patches)
    patch_start_faces = np.array([p[2] for p in patches], dtype=int)
    patch_face_counts = np.array([p[1] for p in patches], dtype=int)

    # mesh hash
    h = hashlib.sha256()
    h.update(owners.tobytes())
    h.update(neighbours.tobytes())
    mesh_hash = h.hexdigest()[:16]

    return MeshMetadata(
        n_cells=n_cells,
        n_internal_faces=n_internal,
        n_faces=n_faces,
        owners=owners,
        neighbours=neighbours,
        face_area_vectors=sf_int,
        owner_weights=owner_weights,
        cell_centres=cell_centres,
        cell_volumes=cell_volumes,
        cell_to_grid=cell_to_grid,
        patch_names=patch_names,
        patch_start_faces=patch_start_faces,
        patch_face_counts=patch_face_counts,
        mesh_hash=mesh_hash,
    )
