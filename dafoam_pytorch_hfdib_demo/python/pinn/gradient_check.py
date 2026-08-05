"""Network-parameter gradient check for the DAFoam-HFDIB PINN.

Tests: theta -> W_theta -> R(W_theta) -> L -> dL/dtheta
against central finite differences. This is the decisive derivative test
because the network produces coupled U-phi perturbations (not arbitrary
phi-only directions that may hit the known baseline phi-Jacobian issue).
"""
from __future__ import annotations

import torch
import numpy as np
from typing import List

from .losses import physics_loss


def gradient_check(model, features, bridge, state_assembler,
                   u_ids, p_ids, phi_ids,
                   gamma_u, gamma_p, gamma_phi,
                   n_params: int = 20,
                   epsilons: List[float] = None) -> dict:
    """Check dL/dtheta_j against central FD for n_params parameters.

    Returns dict with median/max rel error and per-parameter details.
    """
    if epsilons is None:
        epsilons = [1e-4, 1e-5, 1e-6]

    # collect all parameters
    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)

    # select spread-out parameter indices
    indices = np.linspace(0, total_params - 1, n_params, dtype=int)

    # compute AD gradients
    model.zero_grad()
    cell_corr = model(features)
    state = state_assembler.assemble(cell_corr)
    loss, loss_info = physics_loss(state, bridge, u_ids, p_ids, phi_ids,
                                   gamma_u, gamma_p, gamma_phi)
    loss.backward()
    ad_grads = []
    for p in params:
        if p.grad is not None:
            ad_grads.append(p.grad.detach().clone().view(-1))
        else:
            ad_grads.append(torch.zeros(p.numel(), dtype=p.dtype))
    ad_flat = torch.cat(ad_grads)

    # FD check
    results = []
    for idx in indices:
        # find which parameter and offset
        cumulative = 0
        for pi, p in enumerate(params):
            if idx < cumulative + p.numel():
                local_idx = idx - cumulative
                break
            cumulative += p.numel()

        best_rel = None
        best_eps = None
        best_fd = None
        best_ad = ad_flat[idx].item()

        for eps in epsilons:
            # perturb +eps
            orig = p.data.view(-1)[local_idx].item()
            p.data.view(-1)[local_idx] = orig + eps
            with torch.no_grad():
                cell_corr_p = model(features)
                state_p = state_assembler.assemble(cell_corr_p)
                r_p = bridge.residual(state_p.detach().numpy())
                loss_p = 0.5 * float((r_p * r_p).sum())

            # perturb -eps
            p.data.view(-1)[local_idx] = orig - eps
            with torch.no_grad():
                cell_corr_m = model(features)
                state_m = state_assembler.assemble(cell_corr_m)
                r_m = bridge.residual(state_m.detach().numpy())
                loss_m = 0.5 * float((r_m * r_m).sum())

            # restore
            p.data.view(-1)[local_idx] = orig

            fd = (loss_p - loss_m) / (2 * eps)
            denom = max(abs(fd), abs(best_ad), 1e-12)
            rel = abs(fd - best_ad) / denom

            if best_rel is None or rel < best_rel:
                best_rel = rel
                best_eps = eps
                best_fd = fd

        results.append({
            "param_index": int(idx),
            "ad_value": best_ad,
            "fd_value": best_fd,
            "best_eps": best_eps,
            "relative_error": best_rel,
        })

    rels = np.array([r["relative_error"] for r in results])
    return {
        "median_rel": float(np.median(rels)),
        "max_rel": float(np.max(rels)),
        "pass": bool(np.median(rels) < 1e-5 and np.max(rels) < 1e-4),
        "details": results,
    }
