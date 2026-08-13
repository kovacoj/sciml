"""Network-parameter gradient check via directional derivatives.

Tests: theta -> W_theta -> R(W_theta) -> L -> dL/dtheta
against central finite differences of the SAME weighted objective.

Uses directional derivatives (not individual parameters) because the
zero-initialized final layer makes most hidden-layer gradients exactly zero.
"""
from __future__ import annotations

import copy
import hashlib
import numpy as np
import torch
from typing import List

from .losses import (
    ResidualLossConfig,
    weighted_residual_loss_torch,
    weighted_residual_loss_numpy,
)
from dafoam_residual_function import dafoam_residual


EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]


def _param_direction(model, rng, scale=1.0, output_layer_only=False):
    """Generate a random normalized parameter direction."""
    direction = []
    params = list(model.parameters())
    if output_layer_only and hasattr(model, 'output_layer'):
        # include both weight and bias of the output layer
        output_params = list(model.output_layer.parameters())
        output_ids = set()
        for i, p in enumerate(params):
            for op in output_params:
                if p is op:
                    output_ids.add(i)
                    break
        for i, p in enumerate(params):
            if i in output_ids:
                d = rng.standard_normal(p.shape) * scale
                direction.append(torch.from_numpy(d).to(dtype=p.dtype, device=p.device))
            else:
                direction.append(torch.zeros_like(p))
    else:
        for p in params:
            d = rng.standard_normal(p.shape) * scale
            direction.append(torch.from_numpy(d).to(dtype=p.dtype, device=p.device))
    # normalize
    norm = sum(torch.sum(d * d).item() for d in direction) ** 0.5
    if norm < 1e-30:
        return direction
    return [d / norm for d in direction]


def _apply_direction(model, direction, scale):
    saved = []
    for p, d in zip(model.parameters(), direction):
        saved.append(p.data.clone())
        p.data.add_(d * scale)
    return saved


def _restore_parameters(model, saved):
    for p, s in zip(model.parameters(), saved):
        p.data.copy_(s)


def _direction_hash(direction):
    h = hashlib.sha256()
    for d in direction:
        h.update(d.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def gradient_check(
    model,
    features,
    bridge,
    state_assembler,
    u_ids_t,
    p_ids_t,
    phi_ids_t,
    config: ResidualLossConfig,
    n_directions: int = 10,
    epsilons: list = None,
    output_layer_only: bool = False,
) -> dict:
    """Directional parameter-gradient check."""
    if epsilons is None:
        epsilons = EPS_GRID

    rng = np.random.default_rng(42)

    # compute AD gradient
    model.zero_grad()
    cell_corr = model(features)
    state = state_assembler.assemble(cell_corr)
    residual = dafoam_residual(state, bridge)
    loss, loss_info = weighted_residual_loss_torch(
        residual, u_ids_t, p_ids_t, phi_ids_t, config)
    loss.backward()

    ad_grads = [p.grad.clone() if p.grad is not None
                else torch.zeros_like(p) for p in model.parameters()]

    results = []
    for dir_idx in range(n_directions):
        direction = _param_direction(model, rng, scale=1e-4,
                                      output_layer_only=output_layer_only)
        dir_hash = _direction_hash(direction)

        # AD directional derivative
        ad_val = sum(
            torch.sum(g * d).item()
            for g, d in zip(ad_grads, direction)
        )

        # FD directional derivative
        best_eps = best_rel = best_abs = best_fd = None
        for eps in epsilons:
            saved = _apply_direction(model, direction, eps)
            try:
                with torch.no_grad():
                    cell_corr_p = model(features)
                    state_p = state_assembler.assemble(cell_corr_p)
                    r_p = bridge.residual(state_p.detach().numpy())
                    loss_p = weighted_residual_loss_numpy(
                        r_p, u_ids_t.numpy(), p_ids_t.numpy(),
                        phi_ids_t.numpy(), config)
            finally:
                _restore_parameters(model, saved)

            saved = _apply_direction(model, direction, -eps)
            try:
                with torch.no_grad():
                    cell_corr_m = model(features)
                    state_m = state_assembler.assemble(cell_corr_m)
                    r_m = bridge.residual(state_m.detach().numpy())
                    loss_m = weighted_residual_loss_numpy(
                        r_m, u_ids_t.numpy(), p_ids_t.numpy(),
                        phi_ids_t.numpy(), config)
            finally:
                _restore_parameters(model, saved)

            fd = (loss_p - loss_m) / (2 * eps)
            denom = max(abs(fd), abs(ad_val), 1e-12)
            rel = abs(fd - ad_val) / denom
            if best_rel is None or rel < best_rel:
                best_eps = eps
                best_rel = rel
                best_abs = abs(fd - ad_val)
                best_fd = fd

        results.append({
            "direction_id": dir_idx,
            "direction_hash": dir_hash,
            "ad_value": ad_val,
            "best_eps": best_eps,
            "fd_value": best_fd,
            "absolute_error": best_abs,
            "relative_error": best_rel,
        })
        print(f"  dir {dir_idx}: ad={ad_val:.6e} fd={best_fd:.6e} "
              f"rel={best_rel:.2e} eps={best_eps:.1e}", flush=True)

    rels = np.array([r["relative_error"] for r in results])
    return {
        "median_rel": float(np.median(rels)),
        "max_rel": float(np.max(rels)),
        "pass": bool(np.median(rels) < 1e-5 and np.max(rels) < 1e-4),
        "all_finite": bool(np.all(np.isfinite(rels))),
        "details": results,
        "output_layer_only": output_layer_only,
    }


def create_probe_model(model, scale=1e-4):
    """Clone model and perturb output-layer WEIGHT (not just bias)."""
    probe = copy.deepcopy(model)
    with torch.no_grad():
        # perturb the output-layer weight tensor (not just bias)
        output_weight = probe.output_layer.weight
        output_weight.copy_(scale * torch.randn_like(output_weight))
    return probe


def verify_hidden_gradients(probe_model, features, bridge, state_assembler,
                            u_ids_t, p_ids_t, phi_ids_t, config):
    """Verify that the probe model activates hidden-layer gradients."""
    probe_model.zero_grad()
    cell_corr = probe_model(features)
    state = state_assembler.assemble(cell_corr)
    residual = dafoam_residual(state, bridge)
    loss, _ = weighted_residual_loss_torch(
        residual, u_ids_t, p_ids_t, phi_ids_t, config)
    loss.backward()

    hidden_grad_norm = 0.0
    output_grad_norm = 0.0
    for name, p in probe_model.named_parameters():
        if p.grad is not None:
            n = p.grad.norm().item()
            if "output_layer" in name:
                output_grad_norm += n
            else:
                hidden_grad_norm += n

    return {
        "hidden_grad_norm": hidden_grad_norm,
        "output_grad_norm": output_grad_norm,
        "hidden_activated": hidden_grad_norm > 1e-14,
    }
