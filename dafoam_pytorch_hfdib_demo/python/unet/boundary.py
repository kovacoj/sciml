"""Boundary condition enforcement for U-Net flow predictions.

Applies hard BCs to the network output so that inlet, wall and outlet
conditions are exactly satisfied without requiring a warm state.
"""
from __future__ import annotations

import torch
import numpy as np


class BoundaryEnforcer:
    """Applies hard boundary conditions to predicted (ux, uy, p) fields.

    For the 64x64 four-port case:
    - Inlet (left edge, two port bands): ux = U_inlet, uy = 0, p unchanged
    - Outlet (right edge, two port bands): p = 0, ux/uy unchanged
    - Walls (top/bottom edges + port gaps): ux = 0, uy = 0
    - Solid cells (lambda > 0.5): ux = 0, uy = 0
    """

    def __init__(self, grid_h: int = 64, grid_w: int = 64,
                 inlet_bands: list[tuple[int, int]] = None,
                 outlet_bands: list[tuple[int, int]] = None,
                 u_inlet: float = 0.1):
        self.h = grid_h
        self.w = grid_w
        self.u_inlet = u_inlet

        # Default port bands for four-port case (rows 8-15 and 48-55)
        if inlet_bands is None:
            inlet_bands = [(8, 16), (48, 56)]
        if outlet_bands is None:
            outlet_bands = [(8, 16), (48, 56)]

        # Build masks
        self.inlet_mask = torch.zeros(grid_h, dtype=torch.bool)
        for s, e in inlet_bands:
            self.inlet_mask[s:e] = True

        self.outlet_mask = torch.zeros(grid_h, dtype=torch.bool)
        for s, e in outlet_bands:
            self.outlet_mask[s:e] = True

        # Wall mask: everything on left/right edges that isn't inlet/outlet
        self.wall_left = ~self.inlet_mask
        self.wall_right = ~self.outlet_mask

    def apply(self, pred: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Apply hard BCs to predicted fields.

        pred: [B, 3, H, W] (ux, uy, p)
        lam:  [B, 1, H, W] (lambda field, 0=fluid, 1=solid)
        """
        B = pred.shape[0]
        out = pred.clone()

        # Inlet (left column, x=0): ux = U_inlet, uy = 0
        inlet_idx = self.inlet_mask.to(pred.device)
        out[:, 0, inlet_idx, 0] = self.u_inlet  # ux
        out[:, 1, inlet_idx, 0] = 0.0            # uy

        # Wall on left edge (non-inlet): u = 0
        wall_left_idx = self.wall_left.to(pred.device)
        out[:, 0, wall_left_idx, 0] = 0.0
        out[:, 1, wall_left_idx, 0] = 0.0

        # Wall on right edge (non-outlet): u = 0
        wall_right_idx = self.wall_right.to(pred.device)
        out[:, 0, wall_right_idx, -1] = 0.0
        out[:, 1, wall_right_idx, -1] = 0.0

        # Top and bottom walls: u = 0
        out[:, 0, 0, :] = 0.0
        out[:, 1, 0, :] = 0.0
        out[:, 0, -1, :] = 0.0
        out[:, 1, -1, :] = 0.0

        # Outlet (right column): p = 0
        outlet_idx = self.outlet_mask.to(pred.device)
        out[:, 2, outlet_idx, -1] = 0.0

        # Solid cells (lambda > 0.5): u = 0
        solid = (lam[:, 0] > 0.5)
        out[:, 0][solid] = 0.0
        out[:, 1][solid] = 0.0

        return out

    def to(self, device):
        self.inlet_mask = self.inlet_mask.to(device)
        self.outlet_mask = self.outlet_mask.to(device)
        self.wall_left = self.wall_left.to(device)
        self.wall_right = self.wall_right.to(device)
        return self
