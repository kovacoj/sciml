"""Loss functions for DAFoam-HFDIB PINN training.

Uses the residual-returning autograd op and block-weighted losses.
No CFD solution labels. Both torch and numpy paths implement the SAME
weighted objective so gradient checks are consistent.
"""
from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from dafoam_residual_function import dafoam_residual
from dafoam_bridge import DAFoamResidualBridge
from state_layout import StateLayout


@dataclass(frozen=True)
class ResidualLossConfig:
    gamma_u: float
    gamma_p: float
    gamma_phi: float


def weighted_residual_loss_torch(
    residual: torch.Tensor,
    u_ids: torch.Tensor,
    p_ids: torch.Tensor,
    phi_ids: torch.Tensor,
    config: ResidualLossConfig,
) -> Tuple[torch.Tensor, dict]:
    """Compute weighted L = gamma_U*0.5*||R_U||^2 + ... (torch path)."""
    r_u = residual.index_select(0, u_ids)
    r_p = residual.index_select(0, p_ids)
    r_phi = residual.index_select(0, phi_ids)

    loss_u = 0.5 * (r_u * r_u).sum()
    loss_p = 0.5 * (r_p * r_p).sum()
    loss_phi = 0.5 * (r_phi * r_phi).sum()

    total = config.gamma_u * loss_u + config.gamma_p * loss_p + config.gamma_phi * loss_phi

    return total, {
        "total": total,
        "momentum": loss_u,
        "pressure": loss_p,
        "flux": loss_phi,
    }


def weighted_residual_loss_numpy(
    residual: np.ndarray,
    u_ids: np.ndarray,
    p_ids: np.ndarray,
    phi_ids: np.ndarray,
    config: ResidualLossConfig,
) -> float:
    """Compute the SAME weighted objective (numpy path for FD checks)."""
    r_u = residual[u_ids]
    r_p = residual[p_ids]
    r_phi = residual[phi_ids]

    loss_u = 0.5 * float((r_u * r_u).sum())
    loss_p = 0.5 * float((r_p * r_p).sum())
    loss_phi = 0.5 * float((r_phi * r_phi).sum())

    return config.gamma_u * loss_u + config.gamma_p * loss_p + config.gamma_phi * loss_phi


def compute_initial_weights(
    state: torch.Tensor,
    bridge: DAFoamResidualBridge,
    u_ids: np.ndarray,
    p_ids: np.ndarray,
    phi_ids: np.ndarray,
) -> ResidualLossConfig:
    """Compute block weights so each term has comparable magnitude at init."""
    with torch.no_grad():
        r = bridge.residual(state.detach().numpy())
        lu = 0.5 * float((r[u_ids] * r[u_ids]).sum())
        lp = 0.5 * float((r[p_ids] * r[p_ids]).sum())
        lphi = 0.5 * float((r[phi_ids] * r[phi_ids]).sum())

    return ResidualLossConfig(
        gamma_u=1.0 / (lu + 1e-30),
        gamma_p=1.0 / (lp + 1e-30),
        gamma_phi=1.0 / (lphi + 1e-30),
    )
