"""Read the authoritative geometry manifest written by DAFvSourceHFDIBStaticRect.

The C++ class writes postProcessing/hfdibGeometry/{geometry.csv,stencils.json}
when daOptions["fvSource"]["<name>"]["writeGeometryManifest"] is set.
Python must consume this instead of recomputing lambda/sigma independently.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass


@dataclass
class GeometryManifest:
    cells: list[dict]          # per-cell geometry
    stencils: list[dict]        # per-interface-cell stencil
    n_fluid: int
    n_solid: int
    n_interface: int

    @property
    def n_total(self) -> int:
        return len(self.cells)


def load(case_dir: str) -> GeometryManifest:
    geo_path = os.path.join(case_dir, "postProcessing", "hfdibGeometry",
                            "geometry.csv")
    sten_path = os.path.join(case_dir, "postProcessing", "hfdibGeometry",
                             "stencils.json")

    cells = []
    with open(geo_path) as f:
        for row in csv.DictReader(f):
            cells.append({
                "cell_id": int(row["cell_id"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "cz": float(row["cz"]),
                "lambda": float(row["lambda"]),
                "chi": float(row["chi"]),
                "sigma": float(row["sigma"]),
                "is_interface": int(row["is_interface"]) == 1,
                "surface_x": float(row["surface_x"]),
                "surface_y": float(row["surface_y"]),
                "surface_z": float(row["surface_z"]),
                "normal_x": float(row["normal_x"]),
                "normal_y": float(row["normal_y"]),
                "normal_z": float(row["normal_z"]),
                "coeff": float(row["interpolation_coefficient"]),
            })

    with open(sten_path) as f:
        stencils = json.load(f)

    n_fluid = sum(1 for c in cells if c["lambda"] <= 0.0)
    n_solid = sum(1 for c in cells if c["lambda"] >= 1.0)
    n_iface = sum(1 for c in cells if c["is_interface"])

    return GeometryManifest(cells, stencils, n_fluid, n_solid, n_iface)


def validate(manifest: GeometryManifest) -> list[str]:
    """Return list of error messages (empty = valid)."""
    import math
    errors = []
    cells = manifest.cells
    n = len(cells)

    # cell IDs contiguous and unique
    ids = [c["cell_id"] for c in cells]
    if ids != list(range(n)):
        errors.append("cell IDs are not contiguous range(0,n)")
    if len(set(ids)) != n:
        errors.append("cell IDs are not unique")

    if manifest.n_fluid == 0:
        errors.append("zero fluid cells")
    if manifest.n_solid == 0:
        errors.append("zero solid cells")
    if manifest.n_interface == 0:
        errors.append("zero interface cells")

    # stencil IDs match interface cells
    iface_ids = set(interface_cell_ids(manifest))
    sten_ids = set(s["cell_id"] for s in manifest.stencils)
    if sten_ids != iface_ids:
        errors.append(f"stencil cell IDs ({len(sten_ids)}) != "
                      f"interface cell IDs ({len(iface_ids)})")

    for s in manifest.stencils:
        cid = s["cell_id"]
        if not s["source_cells"]:
            errors.append(f"empty stencil for cell {cid}")
            continue
        if len(s["source_cells"]) != len(s["source_weights"]):
            errors.append(f"source/weight length mismatch for cell {cid}")
            continue
        wsum = sum(s["source_weights"])
        if abs(wsum - 1.0) > 1e-12:
            errors.append(f"weights sum {wsum} != 1 for cell {cid}")
        for src in s["source_cells"]:
            if src < 0 or src >= n:
                errors.append(f"source cell {src} out of range for cell {cid}")
                break
            if cells[src]["chi"] != 0.0:
                errors.append(f"source cell {src} not pure fluid "
                              f"(chi={cells[src]['chi']}) for cell {cid}")
                break
        for w in s["source_weights"]:
            if not math.isfinite(w):
                errors.append(f"non-finite weight for cell {cid}")
                break

    for c in cells:
        if not math.isfinite(c["lambda"]):
            errors.append(f"non-finite lambda for cell {c['cell_id']}")
        if not math.isfinite(c["sigma"]):
            errors.append(f"non-finite sigma for cell {c['cell_id']}")
        if c["is_interface"]:
            if not math.isfinite(c["coeff"]):
                errors.append(f"non-finite coeff for cell {c['cell_id']}")
            elif c["coeff"] < 0.0 or c["coeff"] > 1.0:
                errors.append(f"coeff {c['coeff']} outside [0,1] for "
                              f"cell {c['cell_id']}")
    return errors


def solid_cell_ids(manifest: GeometryManifest) -> list[int]:
    return [c["cell_id"] for c in manifest.cells if c["lambda"] >= 1.0]


def interface_cell_ids(manifest: GeometryManifest) -> list[int]:
    return [c["cell_id"] for c in manifest.cells if c["is_interface"]]
