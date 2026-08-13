"""Full-residual returning DAFoam autograd op (CPU float64).

    forward:   W (torch)  -> R(W) (torch)        [via getResiduals]
    backward:  g_res     -> (dR/dW)^T g_res       [via calcJacTVecProduct]

This exposes the residual itself so losses can weight state blocks
(U / p / phi) individually.  The scalar helper dafoam_residual_loss in
residual_autograd.py remains for regression comparison.
"""
from __future__ import annotations

import numpy as np
import torch

from dafoam_bridge import DAFoamResidualBridge


class DAFoamResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, bridge: DAFoamResidualBridge):
        state_np = (
            state.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .numpy()
            .copy()
        )
        residual_np = bridge.residual(state_np)

        ctx.bridge = bridge
        ctx.state_np = state_np
        ctx.state_device = state.device
        ctx.state_dtype = state.dtype

        return torch.from_numpy(residual_np.copy()).to(
            device=state.device, dtype=state.dtype
        )

    @staticmethod
    def backward(ctx, grad_residual):
        seed_np = (
            grad_residual.detach()
            .to(device="cpu", dtype=torch.float64)
            .contiguous()
            .numpy()
            .copy()
        )
        grad_state_np = ctx.bridge.residual_jacobian_transpose_vector(
            ctx.state_np,
            seed_np,
        )
        grad_state = torch.from_numpy(grad_state_np.copy()).to(
            device=ctx.state_device, dtype=ctx.state_dtype
        )
        return grad_state, None


def dafoam_residual(state, bridge: DAFoamResidualBridge):
    return DAFoamResidualFunction.apply(state, bridge)
