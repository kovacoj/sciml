#!/usr/bin/env python3
"""JTV consistency AT the converged primal solution vs at initial state.

Runs a quick primal solve first, then repeats the per-block JTV-vs-FD test.
If p/U-block error collapses near the converged point, the reverse JTV's
residual derivative is only trustworthy near convergence, and the training
walker must account for that (Gate E/Gate F design consequence).
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


def main() -> int:
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    opts = channel_baseline_options(case_dir)
    bridge = DAFoamResidualBridge(case_dir, opts)

    # quick primal solve to a converged state (this case converges easily)
    bridge.solver()
    w = np.ascontiguousarray(
        bridge.solver.getStates().copy(), dtype=np.float64)
    r_norm = np.linalg.norm(bridge.residual(w))
    print(f"[conv] converged-state residual_l2 = {r_norm:.3e}")

    n = w.size
    rng = np.random.default_rng(555)
    blocks = [("p", 0, 343), ("U", 343, 1372), ("nuTilda", 1372, 1715),
              ("phi", 1715, 2891)]
    for name, a, b in blocks:
        d = np.zeros(n); v = np.zeros(n)
        d[a:b] = rng.standard_normal(b - a); d /= np.linalg.norm(d)
        v[a:b] = rng.standard_normal(b - a); v /= np.linalg.norm(v)
        jtv = bridge.residual_jacobian_transpose_vector(w, v)
        bb = float(jtv @ d)
        best = None
        for eps in [3e-4, 1e-4, 3e-5, 1e-5, 1e-6]:
            fd = float(v @ (bridge.residual(w + eps * d)
                            - bridge.residual(w - eps * d)) / (2 * eps))
            rel = abs(fd - bb) / max(1.0, abs(fd), abs(bb))
            if best is None or rel < best[1]:
                best = (eps, fd, bb, rel)
        print(f"[conv] {name:8s}: fd={best[1]:+.8e} jtv={best[2]:+.8e} "
              f"best_rel={best[3]:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
