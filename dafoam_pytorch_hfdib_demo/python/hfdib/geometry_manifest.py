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
    errors = []
    if manifest.n_fluid == 0:
        errors.append("zero fluid cells")
    if manifest.n_solid == 0:
        errors.append("zero solid cells")
    if manifest.n_interface == 0:
        errors.append("zero interface cells")
    for s in manifest.stencils:
        if not s["source_cells"]:
            errors.append(f"empty stencil for cell {s['cell_id']}")
        wsum = sum(s["source_weights"])
        if abs(wsum - 1.0) > 1e-12:
            errors.append(f"weights sum {wsum} != 1 for cell {s['cell_id']}")
    return errors


def solid_cell_ids(manifest: GeometryManifest) -> list[int]:
    return [c["cell_id"] for c in manifest.cells if c["lambda"] >= 1.0]


def interface_cell_ids(manifest: GeometryManifest) -> list[int]:
    return [c["cell_id"] for c in manifest.cells if c["is_interface"]]
