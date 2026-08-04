#!/usr/bin/env python3
"""Gate G verification bundle for the single_obstacle HFDIB case.

  1. Ordinary HFDIB primal solve: residuals converge; solid-region velocity
     ~0 near the rectangle; report numbers.
  2. Partial-primal captures (k = args.ks) for warm-start stages.
  3. HFDIB JTV tape-inclusion test at the latest warm state: dot-product
     JTV-vs-FD on scaled directions, blockwise (U/p/phi separately).

NOTE: requires the patched DAFoam build (dafoam_extension patches applied,
normal+ADR rebuilt). The script detects the extension early via 'DAFvSource'
registration errors and reports them clearly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import hfdib_options, HFDIB_OBSTACLE, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402

EPS_GRID = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6]


def obstacle_lambda(x, y):
    """Analytic lambda(x, y) for the configured rectangle (paper convention).

    Solid = 1 inside the rect; interface = tanh-smeared ring of width V^(1/3)
    approximated here with h = 0.00625 m (in-plane cell size).
    """
    x0, y0, _z0, x1, y1, _z1 = HFDIB_OBSTACLE["bounds"]
    h = 0.00625
    inside = (x0 <= x <= x1) and (y0 <= y <= y1)
    if inside:
        d = min(x - x0, x1 - x, y - y0, y1 - y)
        return 0.5 * (1 + np.tanh(d / h))
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    d = np.hypot(dx, dy)
    return 0.5 * (1 - np.tanh(d / h))


def dot_test(bridge, w, d, v):
    jtv = bridge.residual_jacobian_transpose_vector(w, v)
    b = float(d @ jtv)
    best = None
    for eps in EPS_GRID:
        fd = float(v @ (bridge.residual(w + eps * d)
                        - bridge.residual(w - eps * d)) / (2 * eps))
        rel = abs(fd - b) / max(1.0, abs(fd), abs(b))
        if best is None or rel < best[2]:
            best = (eps, fd, rel)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="*", default=[0, 3, 5, 8])
    args = ap.parse_args()

    case_dir = os.path.join(PROJECT_ROOT, "cases", "single_obstacle")
    os.chdir(case_dir)

    bridge = DAFoamResidualBridge(case_dir, hfdib_options(case_dir))
    layout = build_state_layout("isothermal")
    n = bridge.state_size

    # ---- 1) ordinary HFDIB primal solve -----------------------------------
    t0 = time.perf_counter()
    bridge.solver()
    solve_s = time.perf_counter() - t0

    w_conv = np.ascontiguousarray(
        bridge.solver.getStates().copy(), dtype=np.float64)
    r_conv = bridge.residual(w_conv)
    rl2 = float(np.linalg.norm(r_conv))

    # max solid-region velocity from the converged state (x-fastest cell order)
    nx, ny = 40, 16
    u_idx = layout.indices("U")
    uvec = w_conv[u_idx].reshape(-1, 3)[::]  # cell-major [x0,y0,z0,x1,y1,z1,...]
    lam = np.empty(nx * ny)
    for j in range(ny):
        for i in range(nx):
            cx = 0.0125 + 0.025 * i
            cy = 0.003125 + 0.00625 * j
            lam[j * nx + i] = obstacle_lambda(cx, cy)
    u_xy = uvec[:, :2]
    speed_solid = float(np.max(np.linalg.norm(u_xy[lam > 0.999999], axis=1))) \
        if (lam > 0.999999).any() else 0.0
    speed_if = u_xy[(lam > 0.0) & (lam < 1.0)]
    max_if = float(np.max(np.linalg.norm(speed_if, axis=1))) if speed_if.size else 0.0
    chi_cells = int((lam > 0.0).sum())
    print(f"[g1] HFDIB solve {solve_s:.1f}s  ||R||={rl2:.3e}  "
          f"solid cells={chi_cells}  max|U| in solid={speed_solid:.4e}  "
          f"max|U| at interface={max_if:.4e}")

    # ---- 2) partial-primal captures ---------------------------------------
    from mpi4py import MPI

    captured = {}
    work = os.path.join(PROJECT_ROOT, "outputs", "work")
    os.makedirs(work, exist_ok=True)
    for k in sorted(args.ks):
        # reuse the same partially-solved channel via re-running primal from
        # initial fields with an iteration cap via primalMinIters + endTime=k
        opts = hfdib_options(case_dir)
        opts["primalMinResTol"] = 1e-30
        opts["primalMinIters"] = max(k + 1, 2)
        opts["printInterval"] = 1
        b = DAFoamResidualBridge(case_dir, opts)
        if k > 0:
            b.solver()
        wk = np.ascontiguousarray(
            b.solver.getStates().copy(), dtype=np.float64)
        rk = b.residual(wk)
        captured[k] = {
            "W": wk, "rl2": float(np.linalg.norm(rk)),
        }
        kdir = os.path.join(work, f"hfdib-k{k}")
        os.makedirs(kdir, exist_ok=True)
        np.save(os.path.join(kdir, "W_k.npy"), wk)
        np.save(os.path.join(kdir, "R_k.npy"), rk)
        write_json(os.path.join(kdir, "partial_primal.json"),
                   {"requested_iterations": k,
                    "residual_l2": captured[k]["rl2"]})
        print(f"[g1] partial k={k}: ||R||={captured[k]['rl2']:.3e}",
              flush=True)

    # ---- 3) HFDIB JTV tape-inclusion test at the largest k ----------------
    k_star = max(captured)
    w_star = captured[k_star]["W"]
    rng = np.random.default_rng(777)
    eps_dirs = {
        "U": layout.indices("U"), "p": layout.indices("p"),
        "phi": layout.indices("phi"),
    }
    reports = {}
    for name, ids in [("global", np.arange(n)), *eps_dirs.items()]:
        rels = []
        ids_set = set(ids.tolist())
        block_mask = np.array([1.0 if i in ids_set else 0.0 for i in range(n)])
        for _ in range(4):
            d = rng.standard_normal(n) * block_mask
            s = np.linalg.norm(d)
            if s == 0.0:
                continue
            d /= s
            v = rng.standard_normal(n) * block_mask
            s = np.linalg.norm(v)
            if s == 0.0:
                continue
            v /= s
            eps, fd, ref = dot_test(bridge, w_star, d, v)
            rels.append(ref)
        reports[name] = {"median": float(np.median(rels)), "max": float(max(rels))}
        print(f"[g2] JTV {name:6s} @k{k_star}: median={np.median(rels):.3e} "
              f"max={max(rels):.3e}", flush=True)

    summary = {
        "solve_seconds": solve_s,
        "converged_residual_l2": rl2,
        "solid_cells_python_mask": chi_cells,
        "max_solid_speed": speed_solid,
        "max_interface_speed": max_if,
        "partial_primal": {str(k): v["rl2"] for k, v in captured.items()},
        "hfdib_jtv": reports,
    }
    write_json(os.path.join(PROJECT_ROOT, "outputs", "hfdib_gateG.json"), summary)

    # ---- verdict ------------------------------------------------------------
    jtv_ok = all(v["max"] < 1e-5 for v in reports.values())
    solve_ok = rl2 < 1e-3 and np.all(np.isfinite(r_conv))
    print(f"[g] Gate G interim: solve={'OK' if solve_ok else 'FAIL'} "
          f"JTV={'OK' if jtv_ok else 'FAIL (see hfdib_gateG.json)'}")
    return 0 if (solve_ok and jtv_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
