import numpy as np
import pytest


@pytest.fixture
def synthetic_geometry_path(tmp_path):
    n = 24
    spacing = 1.0 / n
    axis = (np.arange(n) + 0.5) * spacing
    xx, yy = np.meshgrid(axis, axis)
    sigma = xx - 0.5 + 0.04 * np.sin(2.0 * np.pi * yy)
    lam = 0.5 * (1.0 - np.tanh(sigma / spacing))
    lam[sigma >= 4.0 * spacing] = 0.0
    lam[sigma <= -4.0 * spacing] = 1.0
    path = tmp_path / "synthetic_geometry.npz"
    np.savez(path, inputs=lam[None, None])
    return path, spacing


@pytest.fixture
def synthetic_domain_data():
    return {
        "classification": "RECONSTRUCTED_TPFM_DOMAIN",
        "dx": 1.0,
        "roi": {
            "nx": 4,
            "ny": 3,
            "bounds": {"xmin": 0.0, "ymin": 10.0, "xmax": 4.0, "ymax": 13.0},
        },
        "full_domain": {
            "nx": 7,
            "ny": 3,
            "bounds": {"xmin": -2.0, "ymin": 10.0, "xmax": 5.0, "ymax": 13.0},
        },
        "left_extension_cells": 2,
        "right_extension_cells": 1,
        "patches": {
            "inlet": [{
                "name": "inlet", "marker": 101, "side": "left",
                "intervals": [{"min": 10.0, "max": 13.0}],
            }],
            "outlet": [{
                "name": "outlet", "marker": 102, "side": "right",
                "intervals": [{"min": 10.0, "max": 13.0}],
            }],
            "wall": [
                {
                    "name": "bottom", "marker": 103, "side": "bottom",
                    "intervals": [{"min": -2.0, "max": 5.0}],
                },
                {
                    "name": "top", "marker": 104, "side": "top",
                    "intervals": [{"min": -2.0, "max": 5.0}],
                },
            ],
        },
        "uin": 0.1,
        "pout": 0.0,
        "nu": 0.01,
        "roi_cell_indices": [
            [2, 3, 4, 5],
            [9, 10, 11, 12],
            [16, 17, 18, 19],
        ],
    }


@pytest.fixture
def controlled_setup():
    from pathlib import Path

    from src.benchmark_setup import load_config_geometry

    root = Path(__file__).resolve().parents[1]
    return load_config_geometry(root / "configs/controlled_geometry_smoke.json")
