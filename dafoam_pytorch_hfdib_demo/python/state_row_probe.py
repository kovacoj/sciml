#!/usr/bin/env python3
"""Row-wise JTV-vs-FD probing: ∂R_i/∂W for sampled residual rows.

For unit seed v = e_i, calcJacTVecProduct returns the Jacobian ROW of
residual i (as a column vector).  Compare d^T(∂R_i/∂W) against the direct
central FD of R_i along a random d.  Reports median/max relative error per
state block (p / U / nuTilda / phi from the decoded layout).
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
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))
    w = bridge.initial_state()
    n = w.size
    rng = np.random.default_rng(99)

    blocks = [("p", 0, 343, 8), ("U", 343, 1372, 16),
              ("nuTilda", 1372, 1715, 4), ("phi", 1715, 2891, 12)]

    d = rng.standard_normal(n)
    d /= np.linalg.norm(d)
    eps = 1e-5
    rp = bridge.residual(w + eps * d)
    rm = bridge.residual(w - eps * d)
    rfd = (rp - rm) / (2 * eps)  # per-row directional FD

    for name, a, b, k in blocks:
        idx = rng.choice(b - a, size=k, replace=False) + a
        rels = []
        for i in idx:
            v = np.zeros(n)
            v[i] = 1.0
            row = bridge.residual_jacobian_transpose_vector(w, v)  # ∂R_i/∂W column
            fd_i = rfd[i]
            jtv_i = float(d @ row)
            rel = abs(fd_i - jtv_i) / max(1e-12, abs(fd_i), abs(jtv_i))
            rels.append(rel)
        rels = np.array(rels)
        print(f"[row] {name:8s}: n={k:2d} median_rel={np.median(rels):.3e} "
              f"max_rel={rels.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
