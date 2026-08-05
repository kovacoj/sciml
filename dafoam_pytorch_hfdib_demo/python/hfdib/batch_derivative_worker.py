#!/usr/bin/env python3
"""Batch derivative worker: one DAFoam process per (case, k).

Loads a state, generates all directions internally, evaluates all JTV
and FD in one process, writes one compressed npz result file.

Usage:
  python -m hfdib.batch_derivative_worker \
      --case <dir> --options <isothermal|hfdib> \
      --state <W_k.npy> --k <int> \
      --n-dirs 10 --out <result.npz>
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
from state_layout import build_state_layout  # noqa: E402

EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
N_DIRS = 10


def _scaled_dir(rng, n, ids):
    d = np.zeros(n)
    d[ids] = rng.standard_normal(ids.size)
    norm = np.linalg.norm(d)
    return d / norm if norm else d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--options", required=True, choices=["isothermal", "hfdib"])
    ap.add_argument("--state", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--n-dirs", type=int, default=N_DIRS)
    ap.add_argument("--out", required=True)
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
    bridge.residual(w)  # refresh

    blocks = {"U": layout.indices("U"),
              "p": layout.indices("p"),
              "phi": layout.indices("phi")}
    rng = np.random.default_rng(42 + args.k)

    # Collect all results
    results = []
    for v_name, v_ids in blocks.items():
        for d_name, d_ids in blocks.items():
            for dir_i in range(args.n_dirs):
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

                # pick best
                best_eps = None
                best_rel = None
                best_abs = None
                best_fd = None
                for eps, fd in zip(EPS_GRID, fd_vals):
                    denom = max(abs(fd), abs(ad), 1e-12)
                    rel = abs(fd - ad) / denom
                    if best_rel is None or rel < best_rel:
                        best_eps = eps
                        best_rel = rel
                        best_abs = abs(fd - ad)
                        best_fd = fd

                results.append({
                    "k": args.k,
                    "v_block": v_name,
                    "d_block": d_name,
                    "direction_id": dir_i,
                    "best_eps": best_eps,
                    "relative_error": best_rel,
                    "absolute_error": best_abs,
                    "fd_value": best_fd,
                    "ad_value": ad,
                })
                print(f"[batch-d {args.options} k={args.k}] "
                      f"v={v_name} d={d_name} dir={dir_i} "
                      f"rel={best_rel:.2e}", flush=True)

    # Save as structured array
    import json
    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"[batch-d {args.options} k={args.k}] wrote {len(results)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
