"""PyTorch custom autograd bridge for the DAFoam residual loss (CPU float64).

    L(W)    = 1/2 * R(W)^T R(W)          (optionally D^2-weighted)
    grad_W  = (dR/dW)^T * seed,  seed = R  (or D^2 R)

This deliberately uses a residual JTV — NOT the design adjoint solve.
"""
from __future__ import annotations

import numpy as np
import torch

from dafoam_bridge import DAFoamResidualBridge


class DAFoamResidualLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, bridge: DAFoamResidualBridge, residual_weights):
        if state.device.type != "cpu":
            raise ValueError("Initial DAFoam bridge supports CPU tensors only.")
        if state.dtype != torch.float64:
            raise ValueError("DAFoam bridge requires torch.float64.")

        state_np = np.ascontiguousarray(
            state.detach().numpy(), dtype=np.float64
        )
        residual_np = bridge.residual(state_np)

        residual = torch.from_numpy(residual_np.copy()).to(dtype=state.dtype)

        if residual_weights is None:
            weighted_residual = residual
            seed = residual
        else:
            weights = residual_weights.detach().cpu()
            weighted_residual = weights * residual
            # L = 1/2 ||D r||^2  ->  dL/dr = D^2 r
            seed = weights.square() * residual

        loss = 0.5 * torch.dot(weighted_residual, weighted_residual)

        ctx.bridge = bridge
        ctx.state_np = state_np.copy()
        ctx.seed_np = np.ascontiguousarray(
            seed.detach().numpy(), dtype=np.float64
        )
        ctx.state_dtype = state.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        gradient_np = ctx.bridge.residual_jacobian_transpose_vector(
            ctx.state_np,
            ctx.seed_np,
        )
        gradient = torch.from_numpy(gradient_np.copy()).to(dtype=ctx.state_dtype)
        return grad_output * gradient, None, None


def dafoam_residual_loss(state, bridge, residual_weights=None):
    return DAFoamResidualLoss.apply(state, bridge, residual_weights)
