#!/usr/bin/env python3
"""Test the residual-returning autograd op on the isothermal case.

  * residual tensor equals the NumPy residual bit-exactly
  * weighted block losses backprop at all (finite grads)
  * two backward passes with retain_graph=True match
  * dtype/device preservation
  * FD check of the full loss gradient along a few directions (loose,
    cold-start band — the certified D2 harness lives in
    tests/test_d2_warm_gradient.py)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import torch  # noqa: E402

from common import isothermal_channel_options, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from dafoam_residual_function import dafoam_residual  # noqa: E402
from state_layout import build_state_layout  # noqa: E402


def main() -> int:
    torch.set_default_dtype(torch.float64)
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_isothermal")
    bridge = DAFoamResidualBridge(case_dir, isothermal_channel_options(case_dir))
    layout = build_state_layout("isothermal")

    fail = []
    w0 = bridge.initial_state()

    state = torch.from_numpy(w0.copy()).requires_grad_(True)
    res = dafoam_residual(state, bridge)

    # bit-exact forward equality with the numpy residual
    r_np = bridge.residual(w0)
    if not np.array_equal(res.detach().numpy(), r_np):
        fail.append("residual tensor differs from numpy residual")
    if res.dtype != torch.float64 or res.device.type != "cpu":
        fail.append("dtype/device not preserved")

    # block-weighted loss + retain_graph double backward
    wts = {
        "U": torch.zeros(bridge.state_size).index_fill_(
            0, torch.from_numpy(layout.indices("U")), 1.0),
        "p": torch.zeros(bridge.state_size).index_fill_(
            0, torch.from_numpy(layout.indices("p")), 2.0),
        "phi": torch.zeros(bridge.state_size).index_fill_(
            0, torch.from_numpy(layout.indices("phi")), 0.5),
    }
    weights = wts["U"] + wts["p"] + wts["phi"]
    loss_a = 0.5 * (weights * res).square().sum()
    loss_a.backward(retain_graph=True)
    g1 = state.grad.clone()
    state.grad.zero_()
    loss_b = 0.5 * (weights * res).square().sum()
    loss_b.backward()
    g2 = state.grad.clone()
    if not torch.allclose(g1, g2, rtol=0, atol=0):
        fail.append("retain_graph second backward differs")
    if not torch.all(torch.isfinite(g1)):
        fail.append("gradient contains NaN/Inf")

    # loose FD sanity along three random directions (cold-start noise band)
    rng = np.random.default_rng(1)
    def L(x):
        r = bridge.residual(x)
        w_np = weights.numpy()
        rw = w_np * r
        return 0.5 * float(rw @ rw)
    ok_fd = True
    for _ in range(3):
        d = rng.standard_normal(bridge.state_size)
        d /= np.linalg.norm(d)
        fd = (L(w0 + 1e-5 * d) - L(w0 - 1e-5 * d)) / 2e-5
        ad = float(d @ g1.numpy())
        rel = abs(fd - ad) / max(1.0, abs(fd), abs(ad))
        ok_fd = ok_fd and rel < 5e-2  # cold-state band; certified band at W_8
        print(f"[rt-autograd] dir rel err {rel:.2e}")

    if not ok_fd:
        fail.append("FD gradient sanity outside cold-state band")

    print(f"[rt-autograd] {'FAIL: ' + '; '.join(fail) if fail else 'PASS'}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
