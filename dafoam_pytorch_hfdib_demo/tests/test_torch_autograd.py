#!/usr/bin/env python3
"""Gate D: end-to-end PyTorch backward through the DAFoam residual.

Parameterization: W(theta) = W0 + B theta, theta in R^5, B fixed random.
Compare grad_theta of L = 1/2||R(W)||^2 (custom autograd via DAFoam JTV)
against centred finite differences of the full scalar loss.

Run: ./scripts/run_in_container.sh "mpirun -np 1 python tests/test_torch_autograd.py"
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import torch

from common import channel_baseline_options, write_json, PROJECT_ROOT
from dafoam_bridge import DAFoamResidualBridge
from residual_autograd import dafoam_residual_loss

EPSILONS = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]


def main() -> int:
    torch.set_default_dtype(torch.float64)
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))

    rng = np.random.default_rng(31337)
    n = bridge.state_size
    w0 = torch.from_numpy(bridge.initial_state())
    B = torch.from_numpy(rng.standard_normal((n, 5)) * 0.01)

    theta = torch.zeros(5, dtype=torch.float64, requires_grad=True)

    def loss_at(t: torch.Tensor) -> torch.Tensor:
        return dafoam_residual_loss(w0 + B @ t, bridge)

    loss = loss_at(theta)
    loss.backward()
    ad = theta.grad.detach().clone()
    print(f"[autograd] loss = {loss.item():.10e}")
    print(f"[autograd] AD grad: {ad.numpy()}")

    best_rel = 0.0
    rows = []
    for j in range(5):
        per_eps = []
        for eps in EPSILONS:
            tp = theta.detach().clone(); tp[j] += eps
            tm = theta.detach().clone(); tm[j] -= eps
            lp = loss_at(tp).item()
            lm = loss_at(tm).item()
            fd = (lp - lm) / (2.0 * eps)
            denom = max(1.0, abs(fd), abs(ad[j].item()))
            rel = abs(fd - ad[j].item()) / denom
            per_eps.append((eps, fd, rel))
        best = min(per_eps, key=lambda t: t[2])
        best_rel = max(best_rel, best[2])
        rows.append({
            "param": j, "ad": ad[j].item(),
            "per_eps": [{"epsilon": e, "fd": f, "relative_error": r}
                        for (e, f, r) in per_eps],
            "best_fd": best[1], "best_relative_error": best[2],
        })
        print(f"[autograd] theta[{j}]: ad={ad[j].item():+16.10e} "
              f"best_fd={best[1]:+16.10e} best_rel={best[2]:8.3e}")

    write_json(
        os.path.join(case_dir, "autograd_report.json"),
        {"params": rows, "worst_best_relative_error": best_rel},
    )
    ok = bool(np.all(np.isfinite(ad.numpy()))) and best_rel < 1e-4
    print(f"[autograd] Gate D: {'PASS' if ok else 'FAIL'} "
          f"(worst best rel err {best_rel:.3e}, target < 1e-4)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
