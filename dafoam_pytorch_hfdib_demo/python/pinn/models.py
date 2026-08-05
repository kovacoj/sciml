"""Architecture-independent neural field models for DAFoam-HFDIB PINN.

All models produce cell-aligned corrections [n_cells, 3] = (dUx, dUy, dp).
The final layer is zero-initialized so W_theta_0 = W_k (warm state).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuralFieldModel(nn.Module):
    """Base class: features -> [n_cells, 3] cell corrections."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CoordinateMLP(NeuralFieldModel):
    """Pointwise MLP: (x, y, lambda, sigma/h, ...) -> (dUx, dUy, dp)."""

    def __init__(self, input_size: int = 4, width: int = 32, hidden_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(input_size, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)
        self._zero_init_output()

    def _zero_init_output(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [n_cells, input_size]
        return self.net(features)


class CompactDilatedCNN(NeuralFieldModel):
    """Same-resolution dilated CNN for structured 40x16 grid.

    Input: [batch, C, 16, 40] (H=16, W=40)
    Output: [batch, 3, 16, 40] -> reshaped to [n_cells, 3]

    Dilations 1,2,4,8 give receptive field 31 — covers most of the domain.
    Final 1x1 conv zero-initialized.
    """

    def __init__(self, in_channels: int = 6, width: int = 24,
                 dilations: tuple = (1, 2, 4, 8)):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for d in dilations:
            padding = d  # for kernel_size=3, padding=dilation maintains size
            layers.extend([
                nn.Conv2d(prev_ch, width, kernel_size=3, padding=padding, dilation=d),
                nn.Tanh(),
            ])
            prev_ch = width
        layers.append(nn.Conv2d(width, 3, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self._zero_init_output()

    def _zero_init_output(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [batch, C, H, W] or [C, H, W]
        if features.dim() == 3:
            features = features.unsqueeze(0)
        out = self.net(features)  # [batch, 3, H, W]
        # reshape to [n_cells, 3] (batch=1 case)
        b, c, h, w = out.shape
        return out.permute(0, 2, 3, 1).reshape(-1, 3) if b == 1 \
            else out.permute(0, 2, 3, 1).reshape(b, -1, 3)
