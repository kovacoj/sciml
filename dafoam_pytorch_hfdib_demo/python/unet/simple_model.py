"""Simple two-convolution physics network for HFDIB residual training.

Input:  [B, 1, 64, 64] lambda field
Output: [B, 3, 64, 64] corrections (dux, duy, dp)

Only two convolutions + two linear layers. No pooling, skip, decoder, or normalization.
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
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 3 * input_size * input_size),
        )

        self.input_size = input_size
        self.register_buffer(
            "output_scales",
            torch.tensor([velocity_scale, velocity_scale, pressure_scale],
                        dtype=torch.float64).view(1, 3, 1, 1),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.regressor[1].weight, nonlinearity="relu")
        nn.init.zeros_(self.regressor[1].bias)
        nn.init.zeros_(self.regressor[3].weight)
        nn.init.zeros_(self.regressor[3].bias)
        self.to(torch.float64)

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        enc = self.features(lam)
        raw = self.regressor(enc).view(lam.shape[0], 3, self.input_size, self.input_size)
        return raw * self.output_scales
