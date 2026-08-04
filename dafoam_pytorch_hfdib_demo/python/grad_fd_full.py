#!/usr/bin/env python3
"""Full per-entry gradient FD at one warm state: g_i = dL/dW_i central FD.

Compares per layout block: g_tape (JTV with seed R) vs g_fd (per-entry
central FD of L = 1/2 R^T R). Prints medians/maxes and emits the top-20
worst entries per block as JSON. N=2891 -> ~6k residual evals; minutes.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import channel_baseline_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402

SCALE = {"U": 10.0, "p": 50.0, "T": 300.0, "phi": 1.0}
BLOCKS = [("U", 0, 1029), ("p", 1029, 1372), ("T", 1372, 1715), ("phi", 1715, 2891)]


def main() -> int:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    eps_rel = 1e-5

    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))
    w = np.load(os.path.join(PROJECT_ROOT, "outputs", "work",
                             f"partial-k{k}", "W_k.npy"))

    def loss(x):
        r = bridge.residual(x)
        return 0.5 * float(r @ r)

    g_tape = bridge.residual_jacobian_transpose_vector(w, bridge.residual(w))

    scale = np.ones(w.size)
    for name, a, b in BLOCKS:
        scale[a:b] = SCALE[name]

    g_fd = np.zeros(w.size)
    t0 = time.perf_counter()
    ei = np.zeros(w.size)
    for i in range(w.size):
        e = eps_rel / scale[i]
        ei[i] = e
        lp = loss(w + ei)
        lm = loss(w - ei)
        ei[i] = 0.0
        g_fd[i] = (lp - lm) / (2 * e)
        if i % 500 == 0:
            print(f"[gra] {i}/{w.size} {time.perf_counter()-t0:.1f}s", flush=True)

    out_dir = os.path.join(PROJECT_ROOT, "outputs")
    np.save(os.path.join(out_dir, f"grad_fd_k{k}.npy"), g_fd)
    np.save(os.path.join(out_dir, f"grad_tape_k{k}.npy"), g_tape)
    print(f"[gra] k={k} full FD in {time.perf_counter()-t0:.1f}s")

    report = {}
    for name, a, b in BLOCKS:
        ga = g_tape[a:b]
        gf = g_fd[a:b]
        denom = np.maximum(1.0, np.maximum(np.abs(ga), np.abs(gf)))
        rel = np.abs(ga - gf) / denom
        order = np.argsort(-rel)[:20]
        report[name] = {
            "median_rel": float(np.median(rel)),
            "max_rel": float(rel.max()),
            "frac_gt_1e-4": float((rel > 1e-4).mean()),
            "worst": [{"index": int(a + i), "g_tape": float(ga[i]),
                       "g_fd": float(gf[i]), "rel": float(rel[i])}
                      for i in order],
        }
        print(f"[gra] {name:4s} median={np.median(rel):.3e} "
              f"max={rel.max():.3e} frac>1e-4={(rel > 1e-4).mean():.3%}", flush=True)
    write_json(os.path.join(out_dir, f"grad_diff_summary_k{k}.json"), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
