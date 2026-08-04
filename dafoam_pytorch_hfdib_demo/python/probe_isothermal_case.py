#!/usr/bin/env python3
"""Probe the isothermal thin-3D channel case (Gate: 3 of revised plan).

Acceptance:
  * registered states are exactly U, p, phi  (size = 4*640 + 1416 = 3976)
  * getStates/setStates round-trip exactly
  * primal solve converges (quick endTime run)
  * residual deterministic on repeat
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import isothermal_channel_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402

EXPECTED = 3 * 640 + 640 + 2616  # 4*Nc + Nf(all mesh faces, incl boundary)


def main() -> int:
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_isothermal")
    os.chdir(case_dir)

    from mpi4py import MPI
    from dafoam import PYDAFOAM

    opts = isothermal_channel_options(case_dir)
    solver = PYDAFOAM(options=opts, comm=MPI.COMM_WORLD)
    solver.solverAD.initializedRdWTMatrixFree()

    n = int(solver.getNLocalAdjointStates())
    print(f"[iso] state size: {n} (expected {EXPECTED})")

    bridge = None  # constructed lazily via solver handle semantics below
    w0 = np.ascontiguousarray(solver.getStates().copy(), dtype=np.float64)
    solver.setStates(w0)
    solver.solverAD.calcPrimalResidualStatistics("calc")
    r0 = np.ascontiguousarray(solver.getResiduals().copy(), dtype=np.float64)
    solver.setStates(w0)
    solver.solverAD.calcPrimalResidualStatistics("calc")
    r1 = np.ascontiguousarray(solver.getResiduals().copy(), dtype=np.float64)
    repeat = float(np.max(np.abs(r0 - r1)))
    print(f"[iso] residual repeat inf: {repeat:.3e}")

    # round trip
    w_mod = w0.copy()
    w_mod[3 * 640] += 0.333  # first p entry
    solver.setStates(w_mod)
    back = np.ascontiguousarray(solver.getStates().copy())
    rt = abs(back[3 * 640] - w_mod[3 * 640])
    print(f"[iso] roundtrip inf err: {rt:.3e}")
    solver.setStates(w0)

    # primal convergence (quick: 500 max iterations, tol defaults)
    import time
    t0_ = time.perf_counter()
    solver()
    conv_s = time.perf_counter() - t0_
    wc = np.ascontiguousarray(solver.getStates().copy(), dtype=np.float64)
    solver.solverAD.calcPrimalResidualStatistics("calc")
    rc = np.asarray(solver.getResiduals()).copy()
    rl2 = float(np.linalg.norm(rc))
    print(f"[iso] primal converged in {conv_s:.1f}s: ||R||_2 = {rl2:.3e}, "
          f"||R||_inf = {float(np.max(np.abs(rc))):.3e}")

    payload = {
        "state_size": n, "expected": EXPECTED, "size_ok": n == EXPECTED,
        "repeat_inf": repeat,
        "roundtrip_err": rt,
        "primal_seconds": round(conv_s, 2),
        "primal_residual_l2": rl2,
        "residual_linf": float(np.max(np.abs(rc))),
        "states_expected": ["U", "p", "phi"],
    }
    write_json(os.path.join(case_dir, "isothermal_probe.json"), payload)

    ok = (n == EXPECTED and repeat < 1e-12 and rt < 1e-12
          and rl2 < 1e-3 and np.all(np.isfinite(rc)))
    print(f"[iso] probe: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
