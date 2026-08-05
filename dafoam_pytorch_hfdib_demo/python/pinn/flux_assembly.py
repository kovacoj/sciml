"""Differentiable flux assembly: delta_phi = A_phi * delta_U.

For internal faces: delta_phi_f = [alpha_f * dU_owner + (1-alpha_f) * dU_neighbour] . Sf
For boundary faces: delta_phi_f = 0 (fixed-flux BCs).

Uses indexed operations (owner/neighbour gather) — no dense matrices.
"""
from __future__ import annotations

import torch
import numpy as np


class FluxAssembler:
    """Precompute sparse interpolation indices from mesh connectivity."""

    def __init__(self, owners: np.ndarray, neighbours: np.ndarray,
                 sf_vec: np.ndarray, n_cells: int):
        """owners/neighbours: internal face owner/neighbour cell IDs.
        sf_vec: [n_internal_faces, 2] face area vectors (x, y).
        """
        self.n_internal = len(owners)
        self.n_cells = n_cells

        # linear interpolation weights (alpha = 0.5 for uniform orthogonal mesh)
        alpha = 0.5
        self.owners = torch.from_numpy(owners.astype(np.int64))
        self.neighbours = torch.from_numpy(neighbours.astype(np.int64))
        self.sf = torch.from_numpy(sf_vec.astype(np.float64))  # [n_int, 2]
        self.alpha = alpha

    def __call__(self, delta_u: torch.Tensor) -> torch.Tensor:
        """Compute delta_phi from delta_u.

        delta_u: [n_cells, 2] (dUx, dUy per cell)
        returns: [n_internal_faces] delta_phi values
        """
        u_own = delta_u.index_select(0, self.owners)   # [n_int, 2]
        u_nei = delta_u.index_select(0, self.neighbours) # [n_int, 2]
        u_face = self.alpha * u_own + (1 - self.alpha) * u_nei
        phi = (u_face * self.sf).sum(dim=1)  # dot product -> [n_int]
        return phi

    def to(self, device):
        self.owners = self.owners.to(device)
        self.neighbours = self.neighbours.to(device)
        self.sf = self.sf.to(device)
        return self
