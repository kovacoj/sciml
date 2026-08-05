"""Loss functions for DAFoam-HFDIB PINN training.

Uses the residual-returning autograd op (dafoam_residual_function) and
block-weighted losses. No CFD solution labels.
"""
from __future__ import annotations

import torch
from typing import Optional

from dafoam_residual_function import dafoam_residual
from dafoam_bridge import DAFoamResidualBridge


def physics_loss(state: torch.Tensor, bridge: DAFoamResidualBridge,
                 u_ids: torch.Tensor, p_ids: torch.Tensor,
                 phi_ids: torch.Tensor,
                 gamma_u: float = 1.0, gamma_p: float = 1.0,
                 gamma_phi: float = 1.0) -> tuple:
    """Compute block-weighted physics residual loss.

    Returns: (total_loss, loss_dict)
    """
    residual = dafoam_residual(state, bridge)

    r_u = residual.index_select(0, u_ids)
    r_p = residual.index_select(0, p_ids)
    r_phi = residual.index_select(0, phi_ids)

    loss_u = 0.5 * (r_u * r_u).sum()
    loss_p = 0.5 * (r_p * r_p).sum()
    loss_phi = 0.5 * (r_phi * r_phi).sum()

    total = gamma_u * loss_u + gamma_p * loss_p + gamma_phi * loss_phi

    return total, {
        "total": total.item(),
        "momentum": loss_u.item(),
        "pressure": loss_p.item(),
        "flux": loss_phi.item(),
    }


def compute_initial_weights(state: torch.Tensor, bridge: DAFoamResidualBridge,
                            u_ids, p_ids, phi_ids) -> tuple:
    """Compute block weights so each term has comparable magnitude at init."""
    with torch.no_grad():
        r = bridge.residual(state.detach().numpy())
        r_u = r[u_ids.numpy()]
        r_p = r[p_ids.numpy()]
        r_phi = r[phi_ids.numpy()]

        lu = 0.5 * float((r_u * r_u).sum())
        lp = 0.5 * float((r_p * r_p).sum())
        lphi = 0.5 * float((r_phi * r_phi).sum())

        gu = 1.0 / (lu + 1e-30)
        gp = 1.0 / (lp + 1e-30)
        gphi = 1.0 / (lphi + 1e-30)

    return gu, gp, gphi
