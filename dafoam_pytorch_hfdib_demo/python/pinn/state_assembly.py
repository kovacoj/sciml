"""Assemble the complete DAFoam state [U, p, phi] from network output.

state = [U_theta, p_theta, phi_theta]
  U_theta = U_k + delta_U_theta       (cell-major: [Ux0,Uy0,Uz0, Ux1,...])
  p_theta = p_k + delta_p_theta
  phi_theta = phi_k + A_phi * delta_U_theta

Block order matches certified isothermal layout: [U:1920][p:640][phi:2616].
"""
from __future__ import annotations

import torch
from .flux_assembly import FluxAssembler


class StateAssembler:
    """Assemble DAFoam state from base state + neural corrections."""

    def __init__(self, base_state: torch.Tensor,
                 flux_assembler: FluxAssembler,
                 n_u: int = 1920, n_p: int = 640, n_phi: int = 2616):
        """base_state: [N_state] flat tensor (will be detached).
        flux_assembler: precomputed from mesh connectivity.
        """
        self.base = base_state.detach().clone()
        self.flux_asm = flux_assembler
        self.n_u = n_u
        self.n_p = n_p
        self.n_phi = n_phi
        self.n_cells = n_u // 3  # cell-major Ux,Uy,Uz

    def assemble(self, cell_corrections: torch.Tensor) -> torch.Tensor:
        """cell_corrections: [n_cells, 3] = (dUx, dUy, dp).

        Returns: [N_state] assembled state.
        """
        n = self.n_cells
        delta_u = cell_corrections[:, :2]   # [n_cells, 2]
        delta_p = cell_corrections[:, 2]   # [n_cells]

        # U block: cell-major (Ux0, Uy0, Uz0, Ux1, Uy1, Uz1, ...)
        delta_u_flat = torch.zeros(self.n_u, dtype=cell_corrections.dtype,
                                   device=cell_corrections.device)
        delta_u_flat[0::3] = delta_u[:, 0]  # Ux
        delta_u_flat[1::3] = delta_u[:, 1]  # Uy
        # Uz = 0 (pseudo-2D)

        # phi block: assemble from delta_u
        delta_phi_internal = self.flux_asm(delta_u)  # [n_internal_faces]
        # boundary faces: delta_phi = 0 (fixed flux BCs)
        delta_phi = torch.zeros(self.n_phi, dtype=cell_corrections.dtype,
                                device=cell_corrections.device)
        n_internal = delta_phi_internal.shape[0]
        delta_phi[:n_internal] = delta_phi_internal

        # assemble full state
        state = self.base.clone()
        state[:self.n_u] = state[:self.n_u] + delta_u_flat
        state[self.n_u:self.n_u + self.n_p] = state[self.n_u:self.n_u + self.n_p] + delta_p
        state[self.n_u + self.n_p:] = state[self.n_u + self.n_p:] + delta_phi

        return state
