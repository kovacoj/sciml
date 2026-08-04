#!/usr/bin/env python3
"""Layout test for the isothermal channel: exactly U, p, phi states."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import isothermal_channel_options, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402


def main() -> int:
    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_isothermal")
    bridge = DAFoamResidualBridge(case_dir, isothermal_channel_options(case_dir))
    layout = build_state_layout("isothermal")

    fail = []
    print(f"[iso-layout] bridge states={bridge.state_size} "
          f"layout size={layout.size} blocks="
          f"{ {k: len(v) for k, v in layout.indices_by_name.items()} }")

    if bridge.state_size != layout.size:
        fail.append(f"size {bridge.state_size} != layout {layout.size}")
    all_idx = np.sort(np.concatenate(
        [layout.indices(s) for s in ("U", "p", "phi")]))
    if not np.array_equal(all_idx, np.arange(layout.size)):
        fail.append("iso layout does not partition the state")
    if set(layout.indices_by_name) != {"U", "p", "phi"}:
        fail.append("expected exactly [U, p, phi] state names")

    w = bridge.initial_state()
    ux = w[layout.indices("U")[0::3]]
    print(f"[iso-layout] Ux stats: min={ux.min():+.3g} max={ux.max():+.3g} "
          f"(inlet 0.2, walls 0, internal start 0.5)")
    if not np.all(np.isfinite(w)):
        fail.append("initial state not finite")

    print(f"[iso-layout] {'FAIL: ' + '; '.join(fail) if fail else 'PASS'}")
    return 0 if not fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
