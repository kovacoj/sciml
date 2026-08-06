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


# Static HFDIB obstacle definition (single rectangle) used by the
# single_obstacle case and Gate G experiments. Kept here so the fvSource
# wiring lives in exactly one place.
HFDIB_OBSTACLE = {
    "type": "hfdibStaticRect",
    "bounds": [0.45, 0.03, 0.0, 0.55, 0.07, 0.005],
    "d1Factor": 1.5,
}


def hfdib_options(case_dir: str) -> dict:
    """daOptions for channel_isothermal geometry + one static HFDIB rectangle.

    Identical physics to isothermal_channel_options except for the taped
    fvSource carrying the immersed-boundary forcing (the SAME equations are
    used for the partial-primal warm start and the JTV-differentiated
    residual, as required).
    """
    opts = isothermal_channel_options(case_dir)
    opts["fvSource"] = {"obstacle": dict(HFDIB_OBSTACLE)}
    return opts


def hfdib_signed_distance_options(
    case_dir: str,
    geometry_file: str = "constant/hfdibGeometry/signedDistance",
) -> dict:
    """daOptions for topology-conditioned HFDIB using a signed-distance field."""
    opts = isothermal_channel_options(case_dir)
    opts["fvSource"] = {
        "obstacle": {
            "type": "hfdibSignedDistance",
            "geometryFile": geometry_file,
            "solidSign": -1,
            "d1Factor": 1.5,
        }
    }
    return opts


def write_json(path: str, payload) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
