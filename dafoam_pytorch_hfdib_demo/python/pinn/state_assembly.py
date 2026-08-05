"""Assemble the complete DAFoam state [U, p, phi] from network output.

state = [U_theta, p_theta, phi_theta]
  U_theta = U_k + delta_U_theta       (cell-major: [Ux0,Uy0,Uz0, Ux1,...])
  p_theta = p_k + delta_p_theta
  phi_theta = phi_k + A_phi * delta_U_theta

Block order matches certified isothermal layout: [U:n_u][p:n_p][phi:n_phi].
"""
from __future__ import annotations

import torch
from .flux_assembly import FluxAssembler
from state_layout import StateLayout


class StateAssembler:
    """Assemble DAFoam state from base state + neural corrections."""

    def __init__(self, base_state: torch.Tensor,
                 layout: StateLayout,
                 flux_assembler: FluxAssembler,
                 n_cells: int,
                 n_faces: int):
        """base_state: [N_state] flat tensor (will be detached).
        layout: StateLayout for block indices.
        flux_assembler: precomputed from mesh metadata.
        n_cells: number of cells.
        n_faces: total number of faces (for phi block size).
        """
        n_u = len(layout.indices("U"))
        n_p = len(layout.indices("p"))
        n_phi = len(layout.indices("phi"))

        assert base_state.dim() == 1, "base_state must be rank 1"
        assert base_state.shape[0] == n_u + n_p + n_phi, \
            f"base_state length {base_state.shape[0]} != " \
            f"layout size {n_u + n_p + n_phi}"
        assert n_u == 3 * n_cells, \
            f"U block {n_u} != 3 * n_cells {3 * n_cells}"
        assert n_phi == n_faces, \
            f"phi block {n_phi} != n_faces {n_faces}"

        self.base = base_state.detach().clone()
        self.flux_asm = flux_assembler
        self.n_u = n_u
        self.n_p = n_p
        self.n_phi = n_phi
        self.n_cells = n_cells

    def assemble(self, cell_corrections: torch.Tensor) -> torch.Tensor:
        """cell_corrections: [n_cells, 3] = (dUx, dUy, dp).

        Returns: [N_state] assembled state.
        """
        assert cell_corrections.shape == (self.n_cells, 3), \
            f"expected ({self.n_cells}, 3), got {cell_corrections.shape}"

        delta_u = cell_corrections[:, :2]
        delta_p = cell_corrections[:, 2]

        # U block: cell-major (Ux0, Uy0, Uz0, Ux1, Uy1, Uz1, ...)
        delta_u_flat = torch.zeros(self.n_u, dtype=cell_corrections.dtype,
                                   device=cell_corrections.device)
        delta_u_flat[0::3] = delta_u[:, 0]
        delta_u_flat[1::3] = delta_u[:, 1]
        # Uz = 0 (pseudo-2D)

        # phi block: internal faces from flux assembler, boundary = 0
        delta_phi_internal = self.flux_asm(delta_u)
        delta_phi = torch.zeros(self.n_phi, dtype=cell_corrections.dtype,
                                device=cell_corrections.device)
        n_int = delta_phi_internal.shape[0]
        delta_phi[:n_int] = delta_phi_internal

        # assemble
        state = self.base.clone()
        state[:self.n_u] = state[:self.n_u] + delta_u_flat
        state[self.n_u:self.n_u + self.n_p] = \
            state[self.n_u:self.n_u + self.n_p] + delta_p
        state[self.n_u + self.n_p:] = \
            state[self.n_u + self.n_p:] + delta_phi

        return state
