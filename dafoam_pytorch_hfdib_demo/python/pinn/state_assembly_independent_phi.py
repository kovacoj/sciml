"""State assembler with independent phi corrections.

Unlike StateAssembler which constrains phi = phi0 + A_phi * dU,
this assembler allows independent phi corrections on trainable faces:

  U = U0 + dU
  p = p0 + dp
  phi = phi0 + A_phi * dU + dphi_corr

where dphi_corr is predicted independently by the network on
trainable face indices (internal + outlet faces).

At zero network output, the state is exactly W0 (same as old assembler).
"""
from __future__ import annotations

import torch
import numpy as np

from .flux_assembly import FluxAssembler
from state_layout import StateLayout


class IndependentPhiStateAssembler:
    """Assemble DAFoam state with independent phi corrections."""

    def __init__(self, base_state: torch.Tensor,
                 layout: StateLayout,
                 flux_assembler: FluxAssembler,
                 n_cells: int,
                 n_faces: int,
                 phi_trainable_indices: np.ndarray):
        n_u = len(layout.indices("U"))
        n_p = len(layout.indices("p"))
        n_phi = len(layout.indices("phi"))

        assert base_state.shape[0] == n_u + n_p + n_phi

        self.base = base_state.detach().clone()
        self.flux_asm = flux_assembler
        self.n_u = n_u
        self.n_p = n_p
        self.n_phi = n_phi
        self.n_cells = n_cells
        self.n_faces = n_faces

        self.phi_trainable_indices = torch.from_numpy(
            phi_trainable_indices.astype(np.int64))
        self.n_phi_trainable = len(phi_trainable_indices)

        # Physical scales for output normalization
        self.U_SCALE = 0.1
        self.P_SCALE = 0.01
        self.PHI_SCALE = 0.1 * 0.002 * 0.002  # U_scale * face_area = 4e-7

    def assemble(self, cell_corrections: torch.Tensor,
                 phi_corrections: torch.Tensor) -> torch.Tensor:
        """Assemble full state from network outputs.

        Parameters
        ----------
        cell_corrections : [n_cells, 3] = (dUx, dUy, dp) raw (dimensionless)
        phi_corrections : [n_phi_trainable] raw (dimensionless)

        Returns
        -------
        state : [N_state] physical state
        """
        assert cell_corrections.shape == (self.n_cells, 3)
        assert phi_corrections.shape == (self.n_phi_trainable,)

        # Scale to physical units
        delta_u = self.U_SCALE * cell_corrections[:, :2]
        delta_p = self.P_SCALE * cell_corrections[:, 2]

        # U block: cell-major (Ux0, Uy0, Uz0, Ux1, Uy1, Uz1, ...)
        delta_u_flat = torch.zeros(self.n_u, dtype=cell_corrections.dtype,
                                    device=cell_corrections.device)
        delta_u_flat[0::3] = delta_u[:, 0]
        delta_u_flat[1::3] = delta_u[:, 1]

        # phi block: interpolation from U + independent correction
        delta_phi_interpolated = self.flux_asm(delta_u)
        delta_phi = torch.zeros(self.n_phi, dtype=cell_corrections.dtype,
                                 device=cell_corrections.device)
        n_int = delta_phi_interpolated.shape[0]
        delta_phi[:n_int] = delta_phi_interpolated

        # Add independent phi correction on trainable faces
        delta_phi[self.phi_trainable_indices] += (
            self.PHI_SCALE * phi_corrections)

        # Assemble
        state = self.base.clone()
        state[:self.n_u] = state[:self.n_u] + delta_u_flat
        state[self.n_u:self.n_u + self.n_p] = \
            state[self.n_u:self.n_u + self.n_p] + delta_p
        state[self.n_u + self.n_p:] = \
            state[self.n_u + self.n_p:] + delta_phi

        return state
