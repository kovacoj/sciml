"""Build CNN feature tensors from prepared topology data."""
from __future__ import annotations

import numpy as np


def build_features(
    cell_centres: np.ndarray,
    cell_to_grid: np.ndarray,
    grid_shape: tuple[int, int],
    psi: np.ndarray,
    lam: np.ndarray,
    chi: np.ndarray,
    interface: np.ndarray,
    warm_state: np.ndarray,
    u_ref: float = 0.2,
    p_ref: float | None = None,
    domain_bounds: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.1),
) -> np.ndarray:
    """Build 9-channel CNN input [C, H, W].

    Channels: x_hat, y_hat, psi/h, lambda, chi, interface,
              Ux8/Uref, Uy8/Uref, p8/pref
    """
    if p_ref is None:
        p_ref = max(u_ref ** 2, 1e-8)

    nx, ny = grid_shape[1], grid_shape[0]
    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny
    h = np.sqrt(dx * dy)

    n_cells = cell_centres.shape[0]
    n_u = 3 * n_cells
    n_p = n_cells

    feat = np.zeros((9, ny, nx), dtype=np.float64)
    for cell_id in range(n_cells):
        j, i = cell_to_grid[cell_id]
        feat[0, j, i] = 2 * cell_centres[cell_id, 0] / (domain_bounds[1] - domain_bounds[0]) - 1
        feat[1, j, i] = 2 * cell_centres[cell_id, 1] / (domain_bounds[3] - domain_bounds[2]) - 1
        feat[2, j, i] = psi[cell_id] / h
        feat[3, j, i] = lam[cell_id]
        feat[4, j, i] = chi[cell_id]
        feat[5, j, i] = interface[cell_id]
        feat[6, j, i] = warm_state[3 * cell_id] / u_ref      # Ux
        feat[7, j, i] = warm_state[3 * cell_id + 1] / u_ref   # Uy
        feat[8, j, i] = warm_state[n_u + cell_id] / p_ref     # p

    return feat
