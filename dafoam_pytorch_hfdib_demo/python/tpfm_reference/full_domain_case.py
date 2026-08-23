"""Construct a bounded TPFM full-domain hypothesis with fixed port channels."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

H = 0.002
ROI_N = 64
PORT_BANDS = ((8, 16), (48, 56))


def blockmesh_text(extension_cells: int) -> str:
    nx = ROI_N + 2 * extension_cells
    xmin = -extension_cells * H
    xmax = (ROI_N + extension_cells) * H
    ys = (0.0, 0.016, 0.032, 0.096, 0.112, 0.128)
    vertices = []
    for z in (0.0, H):
        for y in ys:
            vertices.extend(((xmin, y, z), (xmax, y, z)))
    vertex_lines = "\n".join(f"    ({x:g} {y:g} {z:g})" for x, y, z in vertices)
    blocks = []
    for band, ny in enumerate((8, 8, 32, 8, 8)):
        a = 2 * band
        b = a + 1
        c = a + 3
        d = a + 2
        blocks.append(f"    hex ({a} {b} {c} {d} {a+12} {b+12} {c+12} {d+12}) ({nx} {ny} 1) simpleGrading (1 1 1)")
    front = []
    back = []
    for band in range(5):
        a = 2 * band
        front.append(f"            ({a} {a+2} {a+3} {a+1})")
        back.append(f"            ({a+12} {a+13} {a+15} {a+14})")
    return f"""FoamFile
{{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
scale 1;
vertices
(
{vertex_lines}
);
blocks
(
{chr(10).join(blocks)}
);
boundary
(
    inletLower {{ type patch; faces ((2 14 16 4)); }}
    inletUpper {{ type patch; faces ((6 18 20 8)); }}
    outletLower {{ type patch; faces ((3 5 17 15)); }}
    outletUpper {{ type patch; faces ((7 9 21 19)); }}
    sideWalls
    {{ type wall; faces ((0 12 14 2) (1 3 15 13) (4 16 18 6) (5 7 19 17) (8 20 22 10) (9 11 23 21)); }}
    topBottomWalls {{ type wall; faces ((0 1 13 12) (10 22 23 11)); }}
    frontAndBack
    {{ type symmetry; faces
        (
{chr(10).join(front + back)}
        );
    }}
);
"""


def create_case(template: Path, destination: Path, extension_cells: int) -> dict:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(template, destination)
    shutil.rmtree(destination / "constant/polyMesh", ignore_errors=True)
    (destination / "system/blockMeshDict").write_text(blockmesh_text(extension_cells))
    return mesh_metadata(extension_cells)


def mesh_metadata(extension_cells: int) -> dict:
    nx = ROI_N + 2 * extension_cells
    centres = np.array([
        [(i - extension_cells + 0.5) * H, (j + 0.5) * H, H / 2]
        for j in range(ROI_N) for i in range(nx)
    ])
    return {
        "cell_centres": centres,
        "n_cells": int(nx * ROI_N),
        "nx": nx,
        "ny": ROI_N,
        "extension_cells": extension_cells,
        "roi_bounds": [0.0, 0.128, 0.0, 0.128],
    }


def full_sigma(lam: np.ndarray, extension_cells: int, width_factor: float = 1.0) -> np.ndarray:
    nx = ROI_N + 2 * extension_cells
    solid = np.ones((ROI_N, nx), dtype=bool)
    roi = slice(extension_cells, extension_cells + ROI_N)
    solid[:, roi] = lam > 0.5
    for start, stop in PORT_BANDS:
        solid[start:stop, :extension_cells] = False
        solid[start:stop, extension_cells + ROI_N:] = False
    sigma = distance_transform_edt(~solid, sampling=(H, H)) - distance_transform_edt(solid, sampling=(H, H))
    interface = (lam > 1e-10) & (lam < 1 - 1e-10)
    roi_sigma = sigma[:, roi]
    roi_sigma[interface] = width_factor * H * np.arctanh(1 - 2 * lam[interface])
    sigma[:, roi] = roi_sigma
    return sigma


def write_sigma(case: Path, sigma: np.ndarray) -> None:
    path = case / "constant/hfdibGeometry/signedDistance"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(sigma, dtype=np.float64).reshape(-1)
    body = "\n".join(f"{value:.16e}" for value in values)
    path.write_text(
        "FoamFile\n{ version 2.0; format ascii; class scalarList; "
        'location "constant/hfdibGeometry"; object signedDistance; }\n\n'
        f"{values.size}\n(\n{body}\n)\n"
    )
