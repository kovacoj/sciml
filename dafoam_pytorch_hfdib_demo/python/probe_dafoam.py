#!/usr/bin/env python3
"""Gate B probe: arbitrary-state residual evaluation (no primal solve).

Initializes PYDAFOAM over the channel baseline case, then:
  1. prints local state / residual sizes and writes state_layout_report.json,
  2. captures states w0,
  3. sets w0 back, computes residual r0,
  4. sets w0 again, computes residual r1,
  5. asserts ||r0 - r1||_inf < 1e-12 (deterministic serial residual).

Run inside the pinned container:  mpirun -np 1 python python/probe_dafoam.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
from mpi4py import MPI

from dafoam import PYDAFOAM

from common import channel_baseline_options, write_json, PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        default=os.path.join(PROJECT_ROOT, "cases", "channel_baseline"),
    )
    args = parser.parse_args()
    case_dir = os.path.abspath(args.case)

    da_options = channel_baseline_options(case_dir)

    # PYDAFOAM resolves the case from CWD (see common.py note).
    os.chdir(case_dir)

    t0 = time.perf_counter()
    solver = PYDAFOAM(options=da_options, comm=MPI.COMM_WORLD)
    solver.solverAD.initializedRdWTMatrixFree()

    n_states = solver.getNLocalAdjointStates()
    print(f"[probe] local state size:    {n_states}")

    w0 = np.ascontiguousarray(solver.getStates().copy(), dtype=np.float64)
    print(f"[probe] getStates() shape:    {w0.shape} dtype: {w0.dtype}")

    solver.setStates(w0)
    solver.solverAD.calcPrimalResidualStatistics("calc")
    r0 = np.ascontiguousarray(solver.getResiduals().copy(), dtype=np.float64)
    print(f"[probe] getResiduals() shape: {r0.shape} dtype: {r0.dtype}")
    if r0.shape[0] != n_states:
        raise RuntimeError(
            f"residual size {r0.shape[0]} != state size {n_states}"
        )

    solver.setStates(w0)
    solver.solverAD.calcPrimalResidualStatistics("calc")
    r1 = np.ascontiguousarray(solver.getResiduals().copy(), dtype=np.float64)

    diff = float(np.max(np.abs(r0 - r1)))
    print(f"[probe] ||r0 - r1||_inf =      {diff:.3e}")
    print(f"[probe] ||r0||_inf =           {float(np.max(np.abs(r0))):.3e}")
    print(f"[probe] finite check: {np.all(np.isfinite(r0))}")

    write_json(
        os.path.join(case_dir, "state_layout_report.json"),
        {
            "n_local_states": int(n_states),
            "n_local_residuals": int(r0.shape[0]),
            "solver": "DASimpleFoam",
            "case": os.path.relpath(case_dir, PROJECT_ROOT),
            "registered_fields": ["p (cells)", "U (cells)", "phi (faces)",
                                  "(model states if any)"],
            "ordering_source": "opaque from getStates(); decode pending "
                               "(see references/DAFOAM_API_NOTES.md)",
            "n_mesh_cells_hint": "blockMeshDict (40,16,1) -> 640",
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        },
    )

    ok = diff < 1e-12
    print(f"[probe] Gate B residual round-trip: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
