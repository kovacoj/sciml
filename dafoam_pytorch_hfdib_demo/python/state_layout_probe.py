#!/usr/bin/env python3
"""Decode the opaque DAFoam state layout for ConvergentChannel.

Hypothesis (sums exactly 343+1029+343+1176 = 2891 = getNLocalAdjointStates):
    [ p : 343 ][ Ux,Uy,Uz : 1029 ][ nuTilda : 343 ][ phi : 1176 ]

Strategy: per-block JTV-vs-FD dot-product test from Gate C (restrict d and v
to one block at a time).  This both labels blocks (phi/nuTilda perturbations
have characteristic residual coupling) and localises the Gate-D mismatch.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import channel_baseline_options, write_json, PROJECT_ROOT
from dafoam_bridge import DAFoamResidualBridge


def main() -> int:
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))

    w = bridge.initial_state()
    n = w.size
    print(f"[layout] n={n}  (343 cells hypothesis: |p|=343 |U|=1029 |nuT|=343 |phi|=1176)")

    blocks = [("p", 0, 343), ("U", 343, 1372), ("nuTilda(?) ", 1372, 1715),
              ("phi(?)   ", 1715, 2891)]

    # 1) value statistics at the initial state
    for name, a, b in blocks:
        blk = w[a:b]
        print(f"[layout] {name}[{a}:{b}] n={b-a:5d} min={blk.min():+12.6g} "
              f"max={blk.max():+12.6g} l2={np.linalg.norm(blk):.6g}")

    # 2) single-block JTV vs FD (eps sweep, best rel err)
    rng = np.random.default_rng(555)
    eps_grid = [3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
    rows = []
    for name, a, b in blocks:
        d = np.zeros(n); v = np.zeros(n)
        d[a:b] = rng.standard_normal(b - a); d /= np.linalg.norm(d)
        v[a:b] = rng.standard_normal(b - a); v /= np.linalg.norm(v)

        jtv = bridge.residual_jacobian_transpose_vector(w, v)
        bb = float(d @ jtv)

        best = None
        for eps in eps_grid:
            fd = float(v @ (bridge.residual(w + eps * d)
                            - bridge.residual(w - eps * d)) / (2 * eps))
            rel = abs(fd - bb) / max(1.0, abs(fd), abs(bb))
            if best is None or rel < best[1]:
                best = (eps, fd, bb, rel)
        rows.append({"block": name, "eps": best[0], "fd": best[1],
                     "jtv": best[2], "best_rel": best[3]})
        print(f"[layout] {name}: fd={best[1]:+.8e} jtv={best[2]:+.8e} "
              f"best_rel={best[3]:.3e} @eps={best[0]:g}")

    write_json(os.path.join(case_dir, "state_layout_decode.json"),
               {"hypothesis": "p|Ux,Uy,Uz|nuTilda|phi 343|1029|343|1176",
                "block_stats": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
