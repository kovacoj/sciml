"""Pure-PyTorch HFDIB interpolation and geometry point search."""

from __future__ import annotations

import numpy as np
import torch

from .geometry import TPFMGeometry


def _distance_like(distance: torch.Tensor | float, value: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(distance, dtype=value.dtype, device=value.device)
    while result.ndim < value.ndim:
        result = result.unsqueeze(-1)
    return result


def first_order(
    boundary_value: torch.Tensor | float,
    first_value: torch.Tensor,
    signed_distance: torch.Tensor | float,
    first_distance: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate the line through values at normal coordinates 0 and d1."""
    x = _distance_like(signed_distance, first_value)
    d1 = _distance_like(first_distance, first_value)
    boundary = torch.as_tensor(
        boundary_value, dtype=first_value.dtype, device=first_value.device
    )
    return (1.0 - x / d1) * boundary + (x / d1) * first_value


def second_order(
    boundary_value: torch.Tensor | float,
    first_value: torch.Tensor,
    second_value: torch.Tensor,
    signed_distance: torch.Tensor | float,
    first_distance: torch.Tensor | float,
    second_distance: torch.Tensor | float,
) -> torch.Tensor:
    """Evaluate the quadratic at samples located at 0, d1, and d1 + d2."""
    x = _distance_like(signed_distance, first_value)
    d1 = _distance_like(first_distance, first_value)
    d12 = d1 + _distance_like(second_distance, first_value)
    boundary = torch.as_tensor(
        boundary_value, dtype=first_value.dtype, device=first_value.device
    )
    l0 = (x - d1) * (x - d12) / (d1 * d12)
    l1 = x * (x - d12) / (d1 * (d1 - d12))
    l2 = x * (x - d1) / (d12 * (d12 - d1))
    return l0 * boundary + l1 * first_value + l2 * second_value


class HFDIB:
    """Locate fluid interpolation points along reconstructed outward normals."""

    def __init__(self, geometry: TPFMGeometry) -> None:
        self.geometry = geometry

    def outward_points(
        self,
        boundary_points: torch.Tensor,
        distances: tuple[float, ...] | list[float],
        *,
        normals: torch.Tensor | None = None,
        search_step: float | None = None,
        max_steps: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find fluid points at or beyond requested outward normal distances."""
        points_np = boundary_points.detach().cpu().numpy().astype(np.float64)
        if normals is None:
            normals_np = self.geometry.interpolate(points_np, "normals")
        else:
            normals_np = normals.detach().cpu().numpy().astype(np.float64)
        lengths = np.linalg.norm(normals_np, axis=-1, keepdims=True)
        if np.any(lengths == 0.0):
            raise ValueError("outward normals must be nonzero")
        normals_np = normals_np / lengths

        step = self.geometry.spacing if search_step is None else float(search_step)
        found_points = []
        found_distances = []
        for requested_distance in distances:
            actual = np.full(points_np.shape[:-1], float(requested_distance))
            candidate = points_np + actual[..., None] * normals_np
            for _ in range(max_steps + 1):
                in_fluid = self.geometry.interpolate(
                    candidate, "signed_distance"
                ) > 0.0
                if np.all(in_fluid):
                    break
                actual = np.where(in_fluid, actual, actual + step)
                candidate = points_np + actual[..., None] * normals_np
            else:
                raise RuntimeError("outward fluid-point search did not converge")
            found_points.append(candidate)
            found_distances.append(actual)

        output_points = torch.as_tensor(
            np.stack(found_points, axis=-2),
            dtype=boundary_points.dtype,
            device=boundary_points.device,
        )
        output_distances = torch.as_tensor(
            np.stack(found_distances, axis=-1),
            dtype=boundary_points.dtype,
            device=boundary_points.device,
        )
        return output_points, output_distances
