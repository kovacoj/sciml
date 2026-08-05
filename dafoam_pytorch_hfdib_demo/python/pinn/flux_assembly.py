"""Differentiable flux assembly: delta_phi = A_phi * delta_U.

For internal faces: delta_phi_f = [w_f * dU_owner + (1-w_f) * dU_neighbour] . Sf
For boundary faces: delta_phi_f = 0 (fixed-flux BCs).

Uses indexed operations (owner/neighbour gather) — no dense matrices.
"""
from __future__ import annotations

import torch
import numpy as np


class FluxAssembler:
    """Precompute sparse interpolation indices from mesh metadata."""

    def __init__(self, owners: np.ndarray, neighbours: np.ndarray,
                 sf_vec: np.ndarray, owner_weights: np.ndarray,
                 n_cells: int, n_faces: int):
        """owners: [n_faces] all face owners.
        neighbours: [n_internal] internal face neighbours.
        sf_vec: [n_internal, 2] face area vectors (x, y only — 2D).
        owner_weights: [n_internal] interpolation weights.
        """
        self.n_internal = len(neighbours)
        self.n_cells = n_cells
        self.n_faces = n_faces

        self.owners = torch.from_numpy(owners[:self.n_internal].astype(np.int64))
        self.neighbours = torch.from_numpy(neighbours.astype(np.int64))
        self.sf = torch.from_numpy(sf_vec[:, :2].astype(np.float64))  # [n_int, 2]
        self.weights = torch.from_numpy(owner_weights.astype(np.float64))

    def __call__(self, delta_u: torch.Tensor) -> torch.Tensor:
        """Compute delta_phi_internal from delta_u.

        delta_u: [n_cells, 2] (dUx, dUy per cell)
        returns: [n_internal] delta_phi values
        """
        u_own = delta_u.index_select(0, self.owners)
        u_nei = delta_u.index_select(0, self.neighbours)
        u_face = self.weights.unsqueeze(1) * u_own + \
                 (1.0 - self.weights.unsqueeze(1)) * u_nei
        phi = (u_face * self.sf).sum(dim=1)
        return phi

    def to(self, device):
        self.owners = self.owners.to(device)
        self.neighbours = self.neighbours.to(device)
        self.sf = self.sf.to(device)
        self.weights = self.weights.to(device)
        return self
