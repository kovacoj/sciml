"""TPFM diffuse-interface geometry reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


class TPFMGeometry:
    """Reconstruct geometry fields from one `inputs` sample in mixer_64.npz."""

    def __init__(
        self,
        dataset: str | Path,
        sample_index: int = 0,
        *,
        spacing: float = 0.002,
        interface_tolerance: float = 1.0e-10,
    ) -> None:
        with np.load(Path(dataset)) as archive:
            inputs = archive["inputs"]
            if inputs.ndim != 4 or inputs.shape[1] != 1:
                raise ValueError("inputs must have shape (samples, 1, ny, nx)")
            self.lambda_field = np.asarray(
                inputs[sample_index, 0], dtype=np.float64
            ).copy()

        self.sample_index = sample_index
        self.spacing = float(spacing)
        self.ny, self.nx = self.lambda_field.shape
        self.x = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.spacing
        self.y = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.spacing

        solid = self.lambda_field > 0.5
        sigma = (
            distance_transform_edt(~solid) - distance_transform_edt(solid)
        ) * self.spacing
        interface = (
            (self.lambda_field > interface_tolerance)
            & (self.lambda_field < 1.0 - interface_tolerance)
        )
        sigma[interface] = self.spacing * np.arctanh(
            1.0 - 2.0 * self.lambda_field[interface]
        )
        self.signed_distance = sigma
        self.interface = interface

        grad_y, grad_x = np.gradient(sigma, self.spacing, self.spacing)
        magnitude = np.hypot(grad_x, grad_y)
        safe_magnitude = np.where(magnitude > 0.0, magnitude, 1.0)
        self.normals = np.stack(
            (grad_x / safe_magnitude, grad_y / safe_magnitude), axis=-1
        )

        reconstructed = 0.5 * (1.0 - np.tanh(sigma / self.spacing))
        difference = reconstructed - self.lambda_field
        self.reconstruction_error = {
            "relative_l2": float(
                np.linalg.norm(difference)
                / (np.linalg.norm(self.lambda_field) + 1.0e-30)
            ),
            "max_abs": float(np.max(np.abs(difference))),
        }

    def interpolate(
        self, coordinates: np.ndarray, field: str = "signed_distance"
    ) -> np.ndarray:
        """Linearly interpolate a geometry field at `(..., 2)` x-y points."""
        fields = {
            "lambda": self.lambda_field,
            "signed_distance": self.signed_distance,
            "normals": self.normals,
        }
        if field not in fields:
            raise ValueError(f"unknown geometry field: {field}")

        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 2:
            raise ValueError("coordinates must have shape (..., 2)")
        original_shape = points.shape[:-1]
        # RegularGridInterpolator indexes arrays as (y, x).
        query = points.reshape(-1, 2)[:, ::-1]
        interpolator = RegularGridInterpolator(
            (self.y, self.x), fields[field], bounds_error=True
        )
        result = interpolator(query)
        trailing_shape = fields[field].shape[2:]
        return result.reshape(original_shape + trailing_shape)
