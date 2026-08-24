"""Prepare and analyze a genuine 3D DAFoam/HFDIB case.

Creates a 64x64x8 mesh with no-slip front/back walls and a partial-depth
rectangular obstacle to produce genuine u(x,y,z) flow.

Run inside the DAFoam container:
    python -m tpfm_reference.run_3d_hfdib --output outputs/3d_hfdib/rect_smoke
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options

NX, NY, NZ = 64, 64, 8
H = 0.002
LX, LY, LZ = NX * H, NY * H, NZ * H

def blockmesh_3d() -> str:
    ys = (0.0, 0.016, 0.032, 0.096, 0.112, 0.128)
    vertices = []
    for z in (0.0, LZ):
        for y in ys:
            vertices.extend(((0.0, y, z), (LX, y, z)))
    vertex_lines = "\n".join(f"    ({x:g} {y:g} {z:g})" for x, y, z in vertices)
    blocks = []
    for band, ny in enumerate((8, 8, 32, 8, 8)):
        a = 2 * band; b = a + 1; c = a + 3; d = a + 2
        blocks.append(f"    hex ({a} {b} {c} {d} {a+12} {b+12} {c+12} {d+12}) ({NX} {ny} {NZ}) simpleGrading (1 1 1)")
    front = []; back = []
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
    front {{ type wall; faces
        (
{chr(10).join(front)}
        );
    }}
    back {{ type wall; faces
        (
{chr(10).join(back)}
        );
    }}
);
"""

def write_3d_signed_distance(path: Path, obstacle_bounds: tuple):
    """Write signed distance for a partial-depth rectangular obstacle."""
    x0, y0, z0, x1, y1, z1 = obstacle_bounds
    # Cell centers: (i+0.5)*H in x, (j+0.5)*H in y (within port bands), (k+0.5)*H in z
    sigma = np.zeros((NX * NY * NZ), dtype=np.float64)
    idx = 0
    # The mesh is structured by blocks; for simplicity compute on the full 64x64x8 grid
    # using the same cell-center convention as the 2D case.
    for k in range(NZ):
        cz = (k + 0.5) * H
        for j in range(NY):
            cy = (j + 0.5) * H
            for i in range(NX):
                cx = (i + 0.5) * H
                # Signed distance to rectangle (positive outside, negative inside)
                dx = max(x0 - cx, cx - x1, 0.0)
                dy = max(y0 - cy, cy - y1, 0.0)
                dz = max(z0 - cz, cz - z1, 0.0)
                if dx == 0 and dy == 0 and dz == 0:
                    # Inside obstacle
                    dx = min(cx - x0, x1 - cx)
                    dy = min(cy - y0, y1 - cy)
                    dz = min(cz - z0, z1 - cz)
                    sigma[idx] = -min(dx, dy, dz)
                else:
                    sigma[idx] = (dx*dx + dy*dy + dz*dz) ** 0.5
                idx += 1
    # Normalize: solid sign = -1 means sigma < 0 in solid
    # Our convention: positive outside, negative inside → matches solidSign=-1
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"{v:.16e}" for v in sigma)
    path.write_text(
        "FoamFile\n{ version 2.0; format ascii; class scalarList; "
        'location "constant/hfdibGeometry"; object signedDistance; }\n\n'
        f"{sigma.size}\n(\n{body}\n)\n"
    )

def foam_internal(path: Path, vector: bool) -> np.ndarray:
    source = path.read_text()
    match = re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;", source, re.S)
    body = match.group(2)
    values = np.array([tuple(map(float, row.split())) for row in re.findall(r"\(([^()]+)\)", body)]) if vector else np.fromstring(body, sep=" ")
    return values

def run_case(output: Path, obstacle_bounds=(0.045, 0.045, 0.004, 0.075, 0.083, 0.012)):
    from dafoam_bridge import DAFoamResidualBridge
    from mpi4py import MPI
    output.mkdir(parents=True, exist_ok=True)
    case_dir = output / "case"
    case_dir.mkdir(parents=True, exist_ok=True)
    # Copy template case
    import shutil
    template = Path("cases/four_port_64x64")
    if template.exists():
        for item in ["0", "constant", "system"]:
            src = template / item
            dst = case_dir / item
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)
    # Write 3D blockMesh
    (case_dir / "system/blockMeshDict").write_text(blockmesh_3d())
    # Write signed distance
    write_3d_signed_distance(case_dir / "constant/hfdibGeometry/signedDistance", obstacle_bounds)
    # Run blockMesh
    subprocess.run(["blockMesh", "-case", str(case_dir)], check=True, capture_output=True)
    # Solve
    os.chdir(str(case_dir))
    bridge = DAFoamResidualBridge(str(case_dir), hfdib_signed_distance_options(
        str(case_dir), inlet_patches=["inletLower", "inletUpper"],
        outlet_patches=["outletLower", "outletUpper"]), comm=MPI.COMM_SELF)
    t0 = time.perf_counter()
    bridge.solver()
    elapsed = time.perf_counter() - t0
    w_star = np.ascontiguousarray(bridge.solver.getStates().copy(), dtype=np.float64)
    n_cells = NX * NY * NZ
    n_u = 3 * n_cells
    u = w_star[:n_u].reshape(n_cells, 3)
    p = w_star[n_u:n_u + n_cells]
    speed = np.linalg.norm(u, axis=1)
    # 3D metrics
    u_reshaped = u.reshape(NZ, NY, NX, 3)
    mid = NZ // 2
    u_mid = u_reshaped[mid]
    u_wall_near = u_reshaped[0]
    depth_var = float(np.linalg.norm(u_wall_near[:, :, :2] - u_mid[:, :, :2]) * 2 /
                      (np.linalg.norm(u_mid[:, :, :2]) + 1e-30))
    uz_l2 = float(np.linalg.norm(u[:, 2]))
    u_total_l2 = float(np.linalg.norm(u))
    solid_mask = np.zeros(n_cells, dtype=bool)
    idx = 0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                cx, cy, cz = (i+0.5)*H, (j+0.5)*H, (k+0.5)*H
                x0, y0, z0, x1, y1, z1 = obstacle_bounds
                if x0 <= cx <= x1 and y0 <= cy <= y1 and z0 <= cz <= z1:
                    solid_mask[idx] = True
                idx += 1
    max_solid_speed = float(speed[solid_mask].max()) if solid_mask.any() else 0.0
    metrics = {
        "mesh": f"{NX}x{NY}x{NZ}", "cells": n_cells,
        "obstacle_bounds": list(obstacle_bounds),
        "elapsed_seconds": elapsed,
        "u_max": float(speed.max()), "u_mean": float(speed.mean()),
        "uz_l2": uz_l2, "uz_fraction": uz_l2 / (u_total_l2 + 1e-30),
        "depth_variation_metric": depth_var,
        "max_solid_speed": max_solid_speed,
        "immersed_wall_check": "PASS" if max_solid_speed < 0.01 else "FAIL",
        "genuine_3d": depth_var > 0.01,
        "delta_p": float(p.max() - p.min()),
    }
    # Save fields
    np.savez_compressed(output / "fields_3d.npz",
        u=u, p=p, speed=speed, solid_mask=solid_mask,
        u_reshaped=u_reshaped, p_reshaped=p.reshape(NZ, NY, NX))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--obstacle", nargs=6, type=float,
                   default=[0.045, 0.045, 0.004, 0.075, 0.083, 0.012])
    a = p.parse_args()
    run_case(a.output, tuple(a.obstacle))

if __name__ == "__main__":
    main()
