#!/usr/bin/env python3
"""Prepare a single topology: warm start + residual + loss config.

Runs in an isolated process to avoid DAFoam process-global state issues.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options, write_json  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from mpi4py import MPI  # noqa: E402


def edit_end_time(path: str, k: int) -> None:
    with open(path) as f:
        s = f.read()
    s = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {max(k, 1)};", s)
    with open(path, "w") as f:
        f.write(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--warm-state", required=True)
    ap.add_argument("--residual", required=True)
    ap.add_argument("--loss-config", required=True)
    args = ap.parse_args()

    case_dir = args.case
    k = args.k

    edit_end_time(os.path.join(case_dir, "system", "controlDict"), k)

    options = hfdib_signed_distance_options(case_dir)
    options["primalMinResTol"] = 1e-30
    options["primalMinIters"] = max(k + 1, 2)
    options["printInterval"] = 1

    os.chdir(case_dir)
    bridge = DAFoamResidualBridge(case_dir, options, comm=MPI.COMM_SELF)

    if k > 0:
        bridge.solver()

    warm_state = np.ascontiguousarray(
        bridge.solver.getStates().copy(), dtype=np.float64)
    residual = bridge.residual(warm_state)

    np.save(args.warm_state, warm_state)
    np.save(args.residual, residual)

    layout = build_state_layout("isothermal")
    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")

    lu = 0.5 * float((residual[u_ids] * residual[u_ids]).sum())
    lp = 0.5 * float((residual[p_ids] * residual[p_ids]).sum())
    lphi = 0.5 * float((residual[phi_ids] * residual[phi_ids]).sum())

    write_json(args.loss_config, {
        "gamma_u": 1.0 / (lu + 1e-30),
        "gamma_p": 1.0 / (lp + 1e-30),
        "gamma_phi": 1.0 / (lphi + 1e-30),
    })

    print(f"[prepare-worker] k={k}: ||R||={np.linalg.norm(residual):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
