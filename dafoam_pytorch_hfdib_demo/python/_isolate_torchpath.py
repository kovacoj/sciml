#!/usr/bin/env python3
"""Isolates the Gate-D noise source: numpy JTV path vs torch autograd path.

For the same warm state and the same direction, compare:
  A) g_np  = d . (J^T R) computed directly with numpy
  B) g_th  = theta.grad through DAFoamResidualLoss (custom autograd)
and FD of the scalar loss along d.  If A == B, the torch autograd path is
faithful and Gate-D errors are direction-statistics; if A != B the torch
wrapper has a bug.
"""
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "python"))

import torch

from common import channel_baseline_options, PROJECT_ROOT
from dafoam_bridge import DAFoamResidualBridge
from residual_autograd import dafoam_residual_loss
from state_layout import build_state_layout

case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))

layout = build_state_layout().indices_by_name
n = bridge.state_size

# reproduce sweep k=20 mixed direction exactly: rng(1000+20), 40 global draws
# (20 pairs x 2), then 4 block draws, then the 5th = mixed
BLOCK_SCALES = {"U": 10.0, "p": 50.0, "T": 300.0, "phi": 1.0}


def scaled_unit(rng, block=None):
    d = np.zeros(n, dtype=np.float64)
    if block is None:
        for name, ids in layout.items():
            d[ids] = rng.standard_normal(ids.size) * BLOCK_SCALES[name]
    else:
        ids = layout[block]
        d[ids] = rng.standard_normal(ids.size) * BLOCK_SCALES[block]
    return d / np.linalg.norm(d)


for tag, k in [("k0", 0), ("k20", 20)]:
    wk = np.load(
        os.path.join(PROJECT_ROOT, "outputs", "work",
                     f"partial-k{k}", "W_k.npy"))
    rng = np.random.default_rng(1000 + k)
    for _ in range(20 * 2):  # 4.1 consumed 20 d + 20 v
        scaled_unit(rng)
    for blk in ["U", "p", "T", "phi"]:
        scaled_unit(rng, blk)
    direction = scaled_unit(rng, None)  # the mixed 5th = sweep's 4.2 mixed

    # A) numpy path
    g = bridge.residual_jacobian_transpose_vector(wk, bridge.residual(wk))
    g_np = float(direction @ g)

    # B) torch autograd path (same direction, same state)
    dt = torch.from_numpy(direction)
    w0t = torch.from_numpy(wk)
    theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
    loss = dafoam_residual_loss(w0t + dt * theta, bridge)
    loss.backward()
    g_th = theta.grad.item()

    # FD truth
    r = bridge.residual
    l_p = 0.5 * float(r(wk + 1e-5 * direction) @ r(wk + 1e-5 * direction))
    l_m = 0.5 * float(r(wk - 1e-5 * direction) @ r(wk - 1e-5 * direction))
    fd = (l_p - l_m) / 2e-5

    print(f"[iso] {tag}: g_np={g_np:+.10e} g_th={g_th:+.10e} "
          f"np_vs_th={abs(g_np - g_th) / max(1.0, abs(g_np), abs(g_th)):.2e} "
          f"fd={fd:+.10e} np_vs_fd={abs(g_np - fd) / max(1.0, abs(g_np), abs(fd)):.2e}")
