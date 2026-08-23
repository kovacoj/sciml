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
