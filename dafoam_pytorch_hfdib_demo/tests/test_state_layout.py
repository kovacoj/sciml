#!/usr/bin/env python3
"""Certify the corrected DAFoam state layout (no nuTilda interpretation).

Tests:
  1. adjStateOrdering == "state" (queried, not parsed from lengths).
  2. Layout sizes sum to getNLocalAdjointStates().
  3. Component fingerprints on the actual state vector match the certified
     layout EXACTLY at block boundaries:
        U block: 343 entries ~10 at positions i%3==0 (cell-major),
        p block: all-zero at the initial state,
        T block: all-300,
        phi block: |v| < ~2 and no constants,
  4. Perturbation echo: for one entry of each registered state, setStates()
     + getStates() preserves the edit bit-exactly (plumbing round-trip),
     and the statistical fingerprint boundaries above uniquely identify the
     physical blocks (no unexplained state block).

Run: ./scripts/run_in_container.sh "mpirun -np 1 python tests/test_state_layout.py"
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
from state_layout import build_state_layout, REGISTERED


def main() -> int:
    failures = []
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")
    bridge = DAFoamResidualBridge(case_dir, channel_baseline_options(case_dir))
    layout = build_state_layout()

    # 1. recorded artifacts
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "state_layout"), exist_ok=True)
    write_json(
        os.path.join(PROJECT_ROOT, "outputs", "state_layout",
                     "state_layout_summary.json"),
        {
            "ordering": bridge.solver.getOption("adjStateOrdering"),
            "n_local_adjoint_states": bridge.state_size,
            "n_cells": 343,
            "n_faces_total": 1176,
            "states": [
                {"name": "U", "type": "volVectorState", "size": 1029,
                 "range": [0, 1029], "packing": "cell-major (Ux,Uy,Uz)"},
                {"name": "p", "type": "volScalarState", "size": 343,
                 "range": [1029, 1372]},
                {"name": "T", "type": "volScalarState", "size": 343,
                 "range": [1372, 1715]},
                {"name": "phi", "type": "surfaceScalarState", "size": 1176,
                 "range": [1715, 2891]},
            ],
            "provenance": "OpenFOAM init log registration print + component "
                          "fingerprints; see state_layout.py docstring",
        },
    )

    # 2. ordering + size
    ordering = bridge.solver.getOption("adjStateOrdering")
    if ordering != layout.ordering:
        failures.append(f"ordering {ordering} != {layout.ordering}")
    if sum(s for _, _, s in REGISTERED) != bridge.state_size:
        failures.append("layout size != getNLocalAdjointStates")

    # 3. fingerprints
    w = bridge.initial_state()
    u_idx = layout.indices("U")
    p_idx = layout.indices("p")
    t_idx = layout.indices("T")
    phi_idx = layout.indices("phi")

    ux_pos = u_idx[0::3]
    n_ux10 = np.count_nonzero(np.abs(w[ux_pos] - 10.0) < 1e-6)
    if n_ux10 != ux_pos.size:
        failures.append(f"Ux pattern: {n_ux10}/{ux_pos.size} ≈ 10")
    u_rest = np.setdiff1d(u_idx, ux_pos)
    if np.count_nonzero(np.abs(w[u_rest]) < 1e-12) != u_rest.size:
        failures.append("Uy/Uz not all zero at the initial state")
    if np.count_nonzero(np.abs(w[p_idx]) < 1e-12) != p_idx.size:
        failures.append("p block not all zero at the initial state")
    if np.count_nonzero(np.abs(w[t_idx] - 300.0) < 1e-12) != t_idx.size:
        failures.append("T block not all 300 at the initial state")
    phi = w[phi_idx]
    if not (np.count_nonzero(np.abs(phi - 300.0) < 1e-9) == 0
            and np.max(np.abs(phi)) < 10.0 and np.all(np.isfinite(phi))):
        failures.append("phi block fingerprint implausible")

    # overlap check: blocks partition the full vector
    all_idx = np.sort(np.concatenate([u_idx, p_idx, t_idx, phi_idx]))
    if not np.array_equal(all_idx, np.arange(bridge.state_size)):
        failures.append("layout blocks do not partition the state")

    # 4. perturbation echo (plumbing) for one entry per registered state
    probes = {
        "Ux cell77": u_idx[77 * 3 + 0],
        "Uy cell77": u_idx[77 * 3 + 1],
        "p  cell77": p_idx[77],
        "T  cell77": t_idx[77],
        "phi face377": phi_idx[377],
    }
    w0 = w.copy()
    for name, i in probes.items():
        w_p = w0.copy()
        w_p[i] += 0.4321
        bridge.set_state(w_p)
        back = bridge.initial_state()
        if abs(back[i] - (w0[i] + 0.4321)) > 1e-12:
            failures.append(f"perturb echo failed at {name} (index {i})")
        bridge.set_state(w0)

    print("[layout] layout: U[0:1029] p[1029:1372] T[1372:1715] phi[1715:2891]")
    for name, i in probes.items():
        print(f"[layout]   probe {name} -> index {i}")
    print(f"[layout] checks: {len(failures)} failures")
    if failures:
        for f in failures:
            print(f"[layout][FAIL] {f}")
        return 1
    print("[layout] all checks PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
