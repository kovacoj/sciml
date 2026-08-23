"""Coordinate neural field."""

from __future__ import annotations

import torch
from torch import nn


class CoordinateMLP(nn.Module):
    """Float64 tanh MLP mapping `(x, y)` to `(u_x, u_y, p)`."""

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 3,
        width: int = 64,
        depth: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(in_features, width), nn.Tanh()))
            in_features = width
        final = nn.Linear(in_features, output_dim)
        nn.init.uniform_(final.weight, -1.0e-5, 1.0e-5)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.network = nn.Sequential(*layers)
        self.to(dtype=torch.float64)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        return self.network(coordinates)
