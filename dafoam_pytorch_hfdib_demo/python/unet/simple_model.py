"""Simple two-convolution physics network for HFDIB residual training.

Input:  [B, 1, 64, 64] lambda field
Output: (cell_corrections, phi_corrections)
  cell_corrections: [B, 3, 64, 64] (dux, duy, dp) scaled to physical units
  phi_corrections: [B, n_phi_trainable] independent flux corrections

Two convolutions + shared latent + two linear heads.
Zero-initialized output so W_theta_0 = W_0 (shared base state).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleFlowNet(nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        conv_channels: tuple[int, int] = (8, 16),
        latent_dim: int = 128,
        velocity_scale: float = 0.1,
        pressure_scale: float = 0.01,
        phi_scale: float = 4e-7,
        n_phi_trainable: int = 8080,
    ):
        super().__init__()
        c1, c2 = conv_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        reduced = input_size // 4
        flat = c2 * reduced * reduced
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, latent_dim),
            nn.SiLU(),
        )

        # Cell head: 3 * 64 * 64 = 12288 outputs (dux, duy, dp)
        self.cell_head = nn.Linear(latent_dim, 3 * input_size * input_size)

        # Phi head: n_phi_trainable independent flux corrections
        self.phi_head = nn.Linear(latent_dim, n_phi_trainable)

        self.input_size = input_size
        self.n_phi_trainable = n_phi_trainable

        self.register_buffer(
            "cell_scales",
            torch.tensor([velocity_scale, velocity_scale, pressure_scale],
                         dtype=torch.float64).view(1, 3, 1, 1),
        )
        self.phi_scale = phi_scale

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.shared[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.shared[1].bias)
        # Zero-init both output heads so initial output = 0 => W = W0
        nn.init.zeros_(self.cell_head.weight)
        nn.init.zeros_(self.cell_head.bias)
        nn.init.zeros_(self.phi_head.weight)
        nn.init.zeros_(self.phi_head.bias)
        self.to(torch.float64)

    def forward(self, lam: torch.Tensor):
        """Returns (cell_corrections [B,3,H,W], phi_corrections [B,n_phi_trainable])."""
        enc = self.features(lam)
        latent = self.shared(enc)

        cell_raw = self.cell_head(latent).view(
            lam.shape[0], 3, self.input_size, self.input_size)
        cell_out = cell_raw * self.cell_scales

        phi_out = self.phi_scale * self.phi_head(latent)

        return cell_out, phi_out
