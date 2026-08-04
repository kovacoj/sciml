"""Shared helpers: case paths, daOptions for the channel baseline, JSON IO."""
from __future__ import annotations

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def channel_baseline_options(case_dir: str) -> dict:
    """daOptions for the serial ConvergentChannel baseline.

    Key patterns are copied from the v5-era test scripts shipped with the
    pinned image (reg_test examples) — no invented keys.  PYDAFOAM resolves
    the OpenFOAM case from the CURRENT WORKING DIRECTORY (this is how the
    test scripts select their case) — callers must os.chdir(case_dir) first.
    """
    return {
        "solverName": "DASimpleFoam",
        "discipline": "aero",
        "useAD": {"mode": "reverse"},
        "printDAOptions": False,
        "primalMinResTol": 1.0e-10,
        "primalMinResTolDiff": 1e12,
        "primalBC": {
            "U0": {"variable": "U", "patches": ["inlet"], "value": [10.0, 0.0, 0.0]},
            "p0": {"variable": "p", "patches": ["outlet"], "value": [0.0]},
            "useWallFunction": False,
        },
        "function": {},
        # NOTE: no "normalizeStates" — Gate C showed the AD JTV and
        # getResiduals disagree in normalized coordinates (measured 2026-08-04);
        # residual scaling will be applied torch-side instead (brief section 26).
    }


def isothermal_channel_options(case_dir: str) -> dict:
    """daOptions for the thin-3-D ISOTHERMAL channel (states [U, p, phi])."""
    return {
        "solverName": "DASimpleFoam",
        "discipline": "aero",
        "useAD": {"mode": "reverse"},
        "printDAOptions": False,
        "primalMinResTol": 1.0e-10,
        "primalMinResTolDiff": 1e12,
        "primalBC": {
            "U0": {"variable": "U", "patches": ["inlet"], "value": [0.2, 0.0, 0.0]},
            "p0": {"variable": "p", "patches": ["outlet"], "value": [0.0]},
            "useWallFunction": False,
        },
        "function": {},
        # no "normalizeStates" (see channel_baseline_options note)
    }


def write_json(path: str, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
