#!/usr/bin/env python3
"""Subprocess worker: evaluate residual and JTV at a given state.

Usage: python -m hfdib.derivative_worker \
    --case <dir> --options <isothermal|hfdib> \
    --state <W.npy> --seed <v.npy> --out <result.npz>

Writes: result.npz with keys: jtv (1d array), residual (1d array)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import isothermal_channel_options, hfdib_options  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--options", required=True,
                    choices=["isothermal", "hfdib"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    opts_factory = {"isothermal": isothermal_channel_options,
                    "hfdib": hfdib_options}[args.options]

    os.chdir(args.case)
    bridge = DAFoamResidualBridge(args.case, opts_factory(args.case))

    w = np.load(args.state)
    v = np.load(args.seed)

    bridge.residual(w)  # refresh state
    jtv = bridge.residual_jacobian_transpose_vector(w, v)
    r = bridge.residual(w)

    np.savez(args.out, jtv=jtv, residual=r)
    print(f"[worker] wrote {args.out}: jtv {jtv.shape}, residual {r.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
