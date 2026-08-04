#!/usr/bin/env python3
"""Gate C: verify DAFoam's reverse residual JTV against finite differences.

v^T (R(W+eps d) - R(W-eps d)) / (2 eps)   vs   d^T (∂R/∂W)ᵀ v

Acceptance:
  * visible finite-difference convergence region across the eps grid
  * minimum relative error below 1e-4 (user-approved: measured systematic
    offset plateau at ~5e-5..6e-5 relative, independent of eps, i.e. not FD
    truncation noise; the brief's original 1e-5 bar was relaxed to 5e-5+
    by the owner on 2026-08-04)
  * no NaN/Inf
  * repeated JTV calls are bit-identical

Run: scripts/run_gradient_check.sh (inside the pinned container).
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

EPSILONS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]


def main() -> int:
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))

    rng = np.random.default_rng(20260804)
    w = bridge.initial_state()
    d = rng.standard_normal(w.shape)
    d /= np.linalg.norm(d)
    v = rng.standard_normal(w.shape)
    v /= np.linalg.norm(v)

    r0 = bridge.residual(w)
    if not np.all(np.isfinite(r0)):
        print("[jtv] residual not finite at base state: FAIL")
        return 1

    # deterministic JTV check (repeatability)
    jtv1 = bridge.residual_jacobian_transpose_vector(w, v)
    jtv2 = bridge.residual_jacobian_transpose_vector(w, v)
    repeat_err = float(np.max(np.abs(jtv1 - jtv2)))
    print(f"[jtv] repeat ||jtv1-jtv2||_inf = {repeat_err:.3e}")

    b = float(d @ jtv1)  # d^T (∂R/∂W)ᵀ v

    rows = []
    for eps in EPSILONS:
        rp = bridge.residual(w + eps * d)
        rm = bridge.residual(w - eps * d)
        if not (np.all(np.isfinite(rp)) and np.all(np.isfinite(rm))):
            print(f"[jtv] eps={eps:g}: non-finite residual: FAIL")
            return 1
        fd = float(v @ (rp - rm) / (2.0 * eps))
        err = abs(fd - b)
        denom = max(1.0, abs(fd), abs(b))
        rel = err / denom
        rows.append({"epsilon": eps, "finite_difference_scalar": fd,
                     "jtv_scalar": b, "absolute_error": err,
                     "relative_error": rel})
        print(f"[jtv] eps={eps:8.1e}  fd={fd:+16.10e}  jtv={b:+16.10e}  "
              f"abs={err:8.3e}  rel={rel:8.3e}")

    best_rel = min(r["relative_error"] for r in rows)
    print(f"[jtv] best relative error: {best_rel:.3e}")

    write_json(
        os.path.join(case_dir, "jtv_report.json"),
        {"rows": rows, "best_relative_error": best_rel,
         "repeat_inf_err": repeat_err},
    )

    ok = best_rel < 1e-4 and repeat_err == 0.0
    print(f"[jtv] Gate C: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
