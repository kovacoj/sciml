#!/usr/bin/env python3
"""Gate E: free-state residual minimization (label-free).

Optimizes the complete DAFoam state directly with Adam on the residual loss;
no neural network, no DAFoam primal solution used as a label, no LBFGS.
Logs per-step norms; acceptance: loss falls by >= 1e2, no NaN/Inf, and the
final state round-trips through setStates.

Run: ./scripts/run_free_state.sh
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import torch

from common import channel_baseline_options, write_json, PROJECT_ROOT
from dafoam_bridge import DAFoamResidualBridge
from residual_autograd import dafoam_residual_loss

STEPS = 1500
LR = 1e-4
GRAD_CLIP = 0.1


def main() -> int:
    torch.set_default_dtype(torch.float64)
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))

    state_np = bridge.initial_state()
    init_res_norm = float(np.linalg.norm(bridge.residual(
        bridge.initial_state())))
    state = torch.nn.Parameter(torch.from_numpy(bridge.initial_state()))
    opt = torch.optim.Adam([state], lr=LR)

    t0 = time.perf_counter()
    history = []
    print(f"[free] initial loss residual_l2={init_res_norm:.6e}")
    for step in range(STEPS):
        opt.zero_grad()
        loss = dafoam_residual_loss(state, bridge)
        if not torch.isfinite(loss):
            print(f"[free] step {step}: non-finite loss {loss.item()}: FAIL")
            return 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_([state], max_norm=GRAD_CLIP)
        opt.step()

        if step % 10 == 0 or step == STEPS - 1:
            with torch.no_grad():
                g = state.grad
                history.append({
                    "step": step,
                    "loss": float(loss.item()),
                    "residual_l2": float(2.0 * loss.item()) ** 0.5,
                    "gradient_l2": float(g.norm().item()),
                    "gradient_linf": float(g.abs().max().item()),
                    "state_ltwo": float(state.norm().item()),
                    "elapsed_seconds": round(time.perf_counter() - t0, 2),
                })
                print(f"[free] step {step:4d} loss={loss.item():.6e} "
                      f"grad_linf={g.abs().max().item():.3e}")

    final_loss = history[-1]["loss"]
    first_loss = history[0]["loss"]
    if history[0]["step"] != 0:
        raise AssertionError

    ratio = first_loss / max(final_loss, 1e-300)
    # round-trip: must be writable back and finite
    end_state = state.detach().numpy().copy()
    bridge.set_state(end_state)
    back = bridge.initial_state()
    rt_err = float(np.max(np.abs(back - end_state)))

    write_json(
        os.path.join(case_dir, "free_state_training.json"),
        {"history": history, "loss_ratio": ratio,
         "roundtrip_inf_err": rt_err},
    )
    print(f"[free] loss {first_loss:.6e} -> {final_loss:.6e}  ratio {ratio:.3e}")
    print(f"[free] setStates round-trip inf err {rt_err:.3e}")
    ok = np.isfinite(final_loss) and ratio >= 1e2 and rt_err == 0.0
    print(f"[free] Gate E: {'PASS' if ok else 'FAIL'} (target ratio >= 1e2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
