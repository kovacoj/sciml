#!/usr/bin/env python3
"""Gate D2 sweep: reverse-JTV accuracy measured on ACTUAL partial-primal
iterates W_k (produced by run_partial_primal.py).

For every captured W_k:
  4.1 global dot-product test on N_PAIRS deterministic (d, v) pairs, both
      dimensionless-scaled per certified layout blocks
      (S: U 10, p 50, T 300, phi 1), eps grid 1e-2 .. 1e-6;
  4.2 five torch directions (U-only, p-only, T-only, phi-only, mixed)
      through the custom autograd bridge, central FD comparison;
  4.3 loss-gradient dot test on 10 random directions: d^T (J^T R) vs FD L.

Writes outputs/warm_start_sweep/summary.{csv,json}.
Entry criterion per revised plan:
    global median best_rel < 1e-5 AND global max best_rel < 1e-4
    AND all torch params < 1e-4 AND all loss dirs < 1e-4.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import torch  # noqa: E402

from common import channel_baseline_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from residual_autograd import dafoam_residual_loss  # noqa: E402
from state_layout import build_state_layout  # noqa: E402

EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
N_PAIRS = 20
N_LOSS_DIRS = 10
BLOCK_SCALES = {"U": 10.0, "p": 50.0, "T": 300.0, "phi": 1.0}

MEDIAN_TOL = 1e-5
MAX_TOL = 1e-4


def scaled_unit(rng: np.random.Generator, layout, size: int,
                block: str | None = None) -> np.ndarray:
    """Unit-norm direction in dimensionless space: d_i = N(0,1) / scale_block,
    so an eps step along d changes every block by ~eps of its natural scale.

    block=None -> mixed across all blocks; otherwise restricted to one
    state's index set.
    """
    d = np.zeros(size, dtype=np.float64)
    if block is None:
        for name, ids in layout.indices_by_name.items():
            d[ids] = rng.standard_normal(ids.size) / BLOCK_SCALES[name]
    else:
        ids = layout.indices_by_name[block]
        d[ids] = rng.standard_normal(ids.size) / BLOCK_SCALES[block]
    norm = np.linalg.norm(d)
    return d / norm if norm else d


def dot_test(bridge, w, d, v, eps_grid=EPS_GRID):
    """best rel error of v^T FD-R against d^T J^T v over the eps grid."""
    jtv = bridge.residual_jacobian_transpose_vector(w, v)
    b = float(np.dot(d, jtv))
    best = None
    for eps in eps_grid:
        fd = float(v @ (bridge.residual(w + eps * d)
                        - bridge.residual(w - eps * d)) / (2.0 * eps))
        rel = abs(fd - b) / max(1.0, abs(fd), abs(b))
        if best is None or rel < best[3]:
            best = (eps, fd, b, rel)
    return best  # (eps, fd, jtv_scalar, rel)


def loss_dot(bridge, w, d, eps_grid=EPS_GRID):
    """g_AD = J^T R via the torch bridge path semantics, compare d^T g vs FD L."""
    g = bridge.residual_jacobian_transpose_vector(w, bridge.residual(w))
    ad = float(np.dot(d, g))

    def loss(x):
        r = bridge.residual(x)
        return 0.5 * float(r @ r)

    best = None
    for eps in eps_grid:
        fd = (loss(w + eps * d) - loss(w - eps * d)) / (2.0 * eps)
        rel = abs(fd - ad) / max(1.0, abs(fd), abs(ad))
        if best is None or rel < best[1]:
            best = (eps, fd, ad, rel)
    return best


def torch_param_err(bridge, w0, direction, rng, eps_grid=EPS_GRID):
    """One θ-scalar direction through the AUTGRAD bridge: FD vs backward().

    W(θ) = w0 + θ·direction; gradient through dafoam_residual_loss vs
    central FD of the actual loss. Returns best rel error over eps grid.
    """
    dt = torch.from_numpy(direction)
    w0t = torch.from_numpy(w0)

    theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
    loss = dafoam_residual_loss(w0t + dt * theta, bridge)
    loss.backward()
    ad = theta.grad.item()

    def l_raw(th):
        r = bridge.residual(w0 + th * direction)
        return 0.5 * float(r @ r)

    best = None
    for eps in eps_grid:
        fd = (l_raw(eps) - l_raw(-eps)) / (2 * eps)
        rel = abs(fd - ad) / max(1.0, abs(fd), abs(ad))
        if best is None or rel < best[1]:
            best = (eps, fd, ad, rel)
    return best


def main() -> int:
    torch.set_default_dtype(torch.float64)
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))
    layout = build_state_layout()
    n = bridge.state_size

    work_root = os.path.join(PROJECT_ROOT, "outputs", "work")
    ks = sorted(int(d.split("k", 1)[1]) for d in os.listdir(work_root)
                if d.startswith("partial-k") and os.path.isdir(
                    os.path.join(work_root, d)))
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "warm_start_sweep")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    summary_rows = []
    for k in ks:
        wk = np.load(os.path.join(work_root, f"partial-k{k}", "W_k.npy"))
        meta = json.load(open(os.path.join(work_root, f"partial-k{k}", "partial_primal.json")))

        rng = np.random.default_rng(1000 + k)

        # 4.1 global dot test on scaled (d, v) pairs
        rels = []
        for _ in range(N_PAIRS):
            d = scaled_unit(rng, layout, n)
            v = scaled_unit(rng, layout, n)
            rels.append(dot_test(bridge, wk, d, v)[3])
        rels = np.array(rels)
        med = float(np.median(rels))
        mx = float(rels.max())

        # 4.2 torch parameter directions (block-restricted + mixed)
        torch_rels = []
        for block in ["U", "p", "T", "phi", None]:
            direction = scaled_unit(rng, layout, n, block=block)
            best = torch_param_err(bridge, wk, direction, rng)
            torch_rels.append(best[3])
        torch_max = max(torch_rels)

        # 4.3 loss-gradient directional test
        loss_rels = []
        for _ in range(N_LOSS_DIRS):
            direction = scaled_unit(rng, layout, n)
            loss_rels.append(loss_dot(bridge, wk, direction)[3])
        loss_max = max(loss_rels)

        eligible = (med < MEDIAN_TOL and mx < MAX_TOL
                    and torch_max < MAX_TOL and loss_max < MAX_TOL)
        rows.append({
            "k": k,
            "actual_iterations": meta["actual_iterations"],
            "residual_l2": meta["residual_l2"],
            "partial_wall_seconds": meta["wall_seconds"],
            "solve_seconds": meta["solve_seconds"],
            "global_median_jtv_error": med,
            "global_max_jtv_error": mx,
            "torch_max_gradient_error": torch_max,
            "loss_direction_max_error": loss_max,
            "eligible": bool(eligible),
        })
        print(f"[sweep] k={k:2d} med={med:.2e} max={mx:.2e} "
              f"torch={torch_max:.2e} lossdir={loss_max:.2e} elig={eligible}",
              flush=True)

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    write_json(os.path.join(out_dir, "summary.json"), {"rows": rows})
    print(f"[sweep] wrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
