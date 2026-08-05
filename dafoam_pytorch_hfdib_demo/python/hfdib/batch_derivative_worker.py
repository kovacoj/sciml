#!/usr/bin/env python3
"""Batch derivative worker: one DAFoam process per (case, k).

Saves ALL epsilon-dependent FD values and a direction hash so the parent
can compute same-epsilon delta-JTV comparisons.

Usage:
  python -m hfdib.batch_derivative_worker \
      --case <dir> --options <isothermal|hfdib> \
      --state <W_k.npy> --k <int> \
      --n-dirs 10 --out <result.json> [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import isothermal_channel_options, hfdib_options  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402

EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
N_DIRS = 10


def _scaled_dir(rng, n, ids):
    d = np.zeros(n)
    d[ids] = rng.standard_normal(ids.size)
    norm = np.linalg.norm(d)
    return d / norm if norm else d


def _dir_hash(d, v):
    h = hashlib.sha256()
    h.update(d.tobytes())
    h.update(v.tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--options", required=True, choices=["isothermal", "hfdib"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--n-dirs", type=int, default=N_DIRS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    opts_factory = {
        "isothermal": isothermal_channel_options,
        "hfdib": hfdib_options,
    }[args.options]

    os.chdir(args.case)
    bridge = DAFoamResidualBridge(args.case, opts_factory(args.case))
    layout = build_state_layout("isothermal")
    n = bridge.state_size

    w = np.load(args.state)
    bridge.residual(w)

    blocks = {"U": layout.indices("U"),
              "p": layout.indices("p"),
              "phi": layout.indices("phi")}
    rng = np.random.default_rng(42 + args.k)

    if args.smoke:
        block_pairs = [("U", "U"), ("U", "p"), ("U", "phi")]
        n_dirs = 1
    else:
        block_pairs = [(v, d) for v in blocks for d in blocks]
        n_dirs = args.n_dirs

    results = []
    for v_name, d_name in block_pairs:
        v_ids = blocks[v_name]
        d_ids = blocks[d_name]
        for dir_i in range(n_dirs):
            d = _scaled_dir(rng, n, d_ids)
            v = _scaled_dir(rng, n, v_ids)

            jtv = bridge.residual_jacobian_transpose_vector(w, v)
            ad = float(d @ jtv)

            fd_vals = []
            for eps in EPS_GRID:
                rp = bridge.residual(w + eps * d)
                rm = bridge.residual(w - eps * d)
                fd = float(v @ (rp - rm) / (2 * eps))
                fd_vals.append(fd)

            results.append({
                "k": args.k,
                "v_block": v_name,
                "d_block": d_name,
                "direction_id": dir_i,
                "direction_hash": _dir_hash(d, v),
                "epsilons": EPS_GRID,
                "fd_values": fd_vals,
                "ad_value": ad,
            })
            print(f"[batch-d {args.options} k={args.k}] "
                  f"v={v_name} d={d_name} dir={dir_i} "
                  f"ad={ad:.6e}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[batch-d {args.options} k={args.k}] wrote {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
