#!/usr/bin/env python3
"""Characterize JTV-vs-FD error as a function of distance from convergence.

States: W(t) = W_conv + t (W_init - W_conv) for t in {1, 0.3, 0.1, 0.03}.
At each state, redo the p/U-block JTV-vs-FD dot test from Gate C.  This maps
the error-vs->state distance scaling and tells whether the Gate-E gradient is
directionally usable mid-trajectory.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import channel_baseline_options, PROJECT_ROOT
from dafoam_bridge import DAFoamResidualBridge


def jtv_err(bridge, w, d, v):
    jtv = bridge.residual_jacobian_transpose_vector(w, v)
    bb = float(jtv @ d)
    best = None
    for eps in [3e-4, 1e-4, 3e-5, 1e-5, 1e-6]:
        fd = float(v @ (bridge.residual(w + eps * d)
                        - bridge.residual(w - eps * d)) / (2 * eps))
        rel = abs(fd - bb) / max(1.0, abs(fd), abs(bb))
        if best is None or rel < best:
            best = rel
    return best


def main() -> int:
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))

    w_init = bridge.initial_state()
    bridge.solver()  # converge
    w_conv = np.ascontiguousarray(
        bridge.solver.getStates().copy(), dtype=np.float64)
    delta = w_init - w_conv

    n = w_init.size
    rng = np.random.default_rng(777)
    d = rng.standard_normal(n); d /= np.linalg.norm(d)
    v = rng.standard_normal(n); v /= np.linalg.norm(v)

    print(f"[scale] t, ||R||_2, jtv_vs_fd_best_rel")
    for t in [1.0, 0.3, 0.1, 0.03, 0.01]:
        w = np.ascontiguousarray(w_conv + t * delta, dtype=np.float64)
        rnorm = float(np.linalg.norm(bridge.residual(w)))
        err = jtv_err(bridge, w, d, v)
        print(f"[scale] t={t:5.2f}  ||R||={rnorm:9.3e}  best_rel={err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
