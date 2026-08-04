#!/usr/bin/env python3
"""Gate D2: derivative verification at the actual warm state W_k.

Full state, full residual (no masking). Direction sets:
  * 20 global scaled state directions
  * 10 U-only, 10 p-only, 10 phi-only directions
  * 10 loss-gradient directions:  dᵀ J(W_k)ᵀ R(W_k)  vs  FD of L = 1/2||R||²

eps sweep 1e-2 .. 1e-6 (central FD, best-relative-error per direction).
Acceptance: median best rel err < 1e-5 AND max best rel err < 1e-4.
Fallback k order is supplied by the caller: --k 8 (default), 10, 12, 15.

Writes outputs/d2_warm_gradient/k{k}/summary.json + directions.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import channel_baseline_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402

EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
BLOCK_SCALES = {"U": 10.0, "p": 50.0, "T": 300.0, "phi": 1.0}
MEDIAN_TOL = 1e-5
MAX_TOL = 1e-4


def scaled_unit(rng, layout, size, block=None):
    d = np.zeros(size, dtype=np.float64)
    blocks = [block] if block else list(BLOCK_SCALES)
    for name in blocks:
        ids = layout.indices_by_name[name]
        d[ids] = rng.standard_normal(ids.size) / BLOCK_SCALES[name]
    nrm = np.linalg.norm(d)
    return d / nrm if nrm else d


def best_dot_err(bridge, w, d, v):
    jtv = bridge.residual_jacobian_transpose_vector(w, v)
    b = float(d @ jtv)
    best = None
    for eps in EPS_GRID:
        fd = float(v @ (bridge.residual(w + eps * d)
                        - bridge.residual(w - eps * d)) / (2 * eps))
        rel = abs(fd - b) / max(1.0, abs(fd), abs(b))
        if best is None or rel < best[2]:
            best = (eps, fd, rel, b)
    return best


def best_loss_err(bridge, w, d):
    g = bridge.residual_jacobian_transpose_vector(w, bridge.residual(w))
    ad = float(d @ g)

    def L(x):
        r = bridge.residual(x)
        return 0.5 * float(r @ r)

    best = None
    for eps in EPS_GRID:
        fd = (L(w + eps * d) - L(w - eps * d)) / (2 * eps)
        rel = abs(fd - ad) / max(1.0, abs(fd), abs(ad))
        if best is None or rel < best[2]:
            best = (eps, fd, rel, ad)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))
    layout = build_state_layout()
    n = bridge.state_size

    w = np.load(os.path.join(PROJECT_ROOT, "outputs", "work",
                             f"partial-k{args.k}", "W_k.npy"))

    rng = np.random.default_rng(20260_04_00 + args.k)
    rows = []

    def add_row(kind, eps, fd, rel, other):
        rows.append({"kind": kind, "best_eps": eps, "fd": fd,
                     "best_rel_err": rel, "ad_or_jtv": other})

    for i in range(20):
        d = scaled_unit(rng, layout, n)
        v = scaled_unit(rng, layout, n)
        eps, fd, rel, other = best_dot_err(bridge, w, d, v)
        add_row("global", eps, fd, rel, other)
    for block in ["U", "p", "phi"]:
        for i in range(10):
            d = scaled_unit(rng, layout, n, block=block)
            v = scaled_unit(rng, layout, n, block=block)
            eps, fd, rel, other = best_dot_err(bridge, w, d, v)
            add_row(f"{block}-only", eps, fd, rel, other)
    for i in range(10):
        d = scaled_unit(rng, layout, n)
        eps, fd, rel, other = best_loss_err(bridge, w, d)
        add_row("loss-gradient", eps, fd, rel, other)

    rels = np.array([r["best_rel_err"] for r in rows])
    med = float(np.median(rels))
    mx = float(rels.max())

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "d2_warm_gradient",
                           f"k{args.k}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "directions.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    write_json(os.path.join(out_dir, "summary.json"), {
        "k": args.k, "n_directions": len(rows),
        "median_best_rel": med, "max_best_rel": mx,
        "per_kind": {kind: {
            "median": float(np.median([r["best_rel_err"] for r in rows
                                       if r["kind"] == kind])),
            "max": float(max(r["best_rel_err"] for r in rows
                             if r["kind"] == kind)),
        } for kind in {r["kind"] for r in rows}},
        "acceptance": {"median_lt": MEDIAN_TOL, "max_lt": MAX_TOL,
                       "pass": bool(med < MEDIAN_TOL and mx < MAX_TOL)},
    })

    print(f"[d2] k={args.k} n_dirs={len(rows)} median={med:.3e} max={mx:.3e}")
    for kind in sorted({r["kind"] for r in rows}):
        ks_ = [r["best_rel_err"] for r in rows if r["kind"] == kind]
        print(f"[d2]   {kind:14s} median={np.median(ks_):.3e} "
              f"max={max(ks_):.3e}")
    ok = med < MEDIAN_TOL and mx < MAX_TOL
    print(f"[d2] Gate D2 @ W_{args.k}: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
