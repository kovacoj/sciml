#!/usr/bin/env python3
"""Physical worker: primal solve or residual eval (one bridge per process).

Usage:
  python -m hfdib.physical_worker --mode solve --case <dir> --options <name> --out <dir>
  python -m hfdib.physical_worker --mode residual --case <dir> --options <name> --state <W.npy> --out <R.npy>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import (isothermal_channel_options, hfdib_options,  # noqa: E402
                     write_json, PROJECT_ROOT)
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["solve", "residual"])
    ap.add_argument("--case", required=True)
    ap.add_argument("--options", required=True, choices=["isothermal", "hfdib"])
    ap.add_argument("--state", type=str, default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    opts_factory = {
        "isothermal": isothermal_channel_options,
        "hfdib": hfdib_options,
    }[args.options]

    os.chdir(args.case)
    bridge = DAFoamResidualBridge(args.case, opts_factory(args.case))

    if args.mode == "solve":
        result = bridge.solver()
        # PYDAFOAM __call__ returns None on success, or a fail flag;
        # check bridge.solver.primalFail instead
        fail = getattr(bridge.solver, 'primalFail', 0)
        if fail:
            print(f"[phys-worker] primal solve FAILED (primalFail={fail})",
                  file=sys.stderr)
            return 1
        w = np.ascontiguousarray(
            bridge.solver.getStates().copy(), dtype=np.float64)
        r = bridge.residual(w)
        np.save(os.path.join(args.out, "W.npy"), w)
        np.save(os.path.join(args.out, "R.npy"), r)
        print(f"[phys-worker] solve {args.options}: ||R||={np.linalg.norm(r):.3e}")
    elif args.mode == "residual":
        w = np.load(args.state)
        r = bridge.residual(w)
        np.save(args.out, r)
        print(f"[phys-worker] residual {args.options} at {args.state}: "
              f"||R||={np.linalg.norm(r):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
