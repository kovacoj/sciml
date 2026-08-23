"""TPFM diffuse-interface geometry reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from .domain import (
    Bounds,
    DomainSpec,
    Interval,
    MANUFACTURED_CIRCULAR_HFDIB,
    MANUFACTURED_EMPTY_CHANNEL,
    Patch,
)


class ManufacturedDomainMetadata:
    """Physical values and complete outer-boundary partition for a rectangle."""

    def __init__(self, classification: str, bounds: Bounds) -> None:
        self.classification = classification
        self.uin = 0.1
        self.pout = 0.0
        self.nu = 0.01
        self.article_reproduction = False
        vertical = (Interval(bounds.ymin, bounds.ymax),)
        horizontal = (Interval(bounds.xmin, bounds.xmax),)
        self.inlet = (Patch("left", vertical, marker=1, name="inlet"),)
        self.outlet = (Patch("right", vertical, marker=2, name="outlet"),)
        self.wall = (
            Patch("bottom", horizontal, marker=3, name="bottom"),
            Patch("top", horizontal, marker=4, name="top"),
        )


class EmptyChannelGeometry:
    """Analytic obstacle-free manufactured rectangular channel."""

    def __init__(self, *, spacing: float = 0.002) -> None:
        self.spacing = float(spacing)
        self.xmin = self.ymin = 0.0
        self.xmax, self.ymax = 0.256, 0.128
        self.nx = int(round((self.xmax - self.xmin) / self.spacing))
        self.ny = int(round((self.ymax - self.ymin) / self.spacing))
        if self.nx != 128 or self.ny != 64:
            raise ValueError("manufactured raster spacing must produce a 128 by 64 grid")
        self.x = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.spacing
        self.y = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.spacing
        self.classification = MANUFACTURED_EMPTY_CHANNEL
        self.analytic_geometry = True
        self.spec = ManufacturedDomainMetadata(
            self.classification, Bounds(self.xmin, self.ymin, self.xmax, self.ymax)
        )
        self.roi_bounds = None
        self.lambda_field = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.signed_distance = np.full_like(self.lambda_field, self.xmax + self.ymax)
        self.normals = np.zeros((*self.lambda_field.shape, 2), dtype=np.float64)
        self.interface = np.zeros_like(self.lambda_field, dtype=bool)
        self.reconstruction_error = {"relative_l2": 0.0, "max_abs": 0.0}

    def interpolate(self, coordinates: np.ndarray, field: str = "signed_distance") -> np.ndarray:
        points = _coordinate_array(coordinates)
        shape = points.shape[:-1]
        if field == "lambda":
            return np.zeros(shape, dtype=np.float64)
        if field == "signed_distance":
            return np.full(shape, self.xmax + self.ymax, dtype=np.float64)
        if field == "normals":
            return np.zeros(shape + (2,), dtype=np.float64)
        raise ValueError(f"unknown geometry field: {field}")


class CircularObstacleGeometry(EmptyChannelGeometry):
    """Analytic manufactured diffuse circular obstacle in the channel."""

    center = np.array((0.128, 0.064), dtype=np.float64)
    radius = 0.024

    def __init__(self, *, spacing: float = 0.002) -> None:
        super().__init__(spacing=spacing)
        self.classification = MANUFACTURED_CIRCULAR_HFDIB
        self.spec = ManufacturedDomainMetadata(
            self.classification, Bounds(self.xmin, self.ymin, self.xmax, self.ymax)
        )
        xx, yy = np.meshgrid(self.x, self.y)
        points = np.stack((xx, yy), axis=-1)
        self.signed_distance = self._sigma(points)
        self.normals = self._normals(points)
        self.lambda_field = 0.5 * (
            1.0 - np.tanh(self.signed_distance / self.spacing)
        )
        self.interface = (self.lambda_field > 1.0e-10) & (
            self.lambda_field < 1.0 - 1.0e-10
        )
        self.reconstruction_error = {"relative_l2": 0.0, "max_abs": 0.0}

    def _sigma(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points - self.center, axis=-1) - self.radius

    def _normals(self, points: np.ndarray) -> np.ndarray:
        displacement = points - self.center
        distance = np.linalg.norm(displacement, axis=-1)
        return np.divide(
            displacement,
            distance[..., None],
            out=np.zeros_like(displacement),
            where=distance[..., None] > 0.0,
        )

    def interpolate(self, coordinates: np.ndarray, field: str = "signed_distance") -> np.ndarray:
        points = _coordinate_array(coordinates)
        sigma = self._sigma(points)
        if field == "signed_distance":
            return sigma
        if field == "normals":
            return self._normals(points)
        if field == "lambda":
            return 0.5 * (1.0 - np.tanh(sigma / self.spacing))
        raise ValueError(f"unknown geometry field: {field}")


def _coordinate_array(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    if points.ndim == 0 or points.shape[-1] != 2:
        raise ValueError("coordinates must have shape (..., 2)")
    return points


def _geometry_fields(lambda_field: np.ndarray, spacing: float, tolerance: float):
    solid = lambda_field > 0.5
    sigma = (
        distance_transform_edt(~solid) - distance_transform_edt(solid)
    ) * spacing
    interface = (lambda_field > tolerance) & (lambda_field < 1.0 - tolerance)
    sigma[interface] = spacing * np.arctanh(1.0 - 2.0 * lambda_field[interface])
    grad_y = (
        np.gradient(sigma, spacing, axis=0)
        if sigma.shape[0] > 1 else np.zeros_like(sigma)
    )
    grad_x = (
        np.gradient(sigma, spacing, axis=1)
        if sigma.shape[1] > 1 else np.zeros_like(sigma)
    )
    magnitude = np.hypot(grad_x, grad_y)
    safe_magnitude = np.where(magnitude > 0.0, magnitude, 1.0)
    normals = np.stack((grad_x / safe_magnitude, grad_y / safe_magnitude), axis=-1)
    return sigma, interface, normals


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
        self.xmin = self.ymin = 0.0
        self.xmax = self.nx * self.spacing
        self.ymax = self.ny * self.spacing
        self.classification = "CROPPED_TPFM_ROI_DIAGNOSTIC"
        self.x = (np.arange(self.nx, dtype=np.float64) + 0.5) * self.spacing
        self.y = (np.arange(self.ny, dtype=np.float64) + 0.5) * self.spacing

        sigma, interface, normals = _geometry_fields(
            self.lambda_field, self.spacing, interface_tolerance
        )
        self.signed_distance = sigma
        self.interface = interface
        self.normals = normals

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


class FullDomainGeometry:
    """Full rectangular geometry reconstructed from an ROI and a domain spec."""

    def __init__(
        self,
        roi_geometry: TPFMGeometry,
        spec: DomainSpec,
        *,
        interface_tolerance: float = 1.0e-10,
    ) -> None:
        if roi_geometry.lambda_field.shape != (spec.roi.ny, spec.roi.nx):
            raise ValueError(
                "ROI lambda shape does not match domain spec roi.ny/roi.nx"
            )
        if not np.isclose(roi_geometry.spacing, spec.dx):
            raise ValueError("ROI geometry spacing does not match domain spec dx")
        expected = np.arange(spec.full_domain.nx * spec.full_domain.ny).reshape(
            spec.full_domain.ny, spec.full_domain.nx
        )[:, spec.left_extension_cells:spec.left_extension_cells + spec.roi.nx]
        declared = np.asarray(spec.roi_cell_indices, dtype=np.int64)
        if not np.array_equal(declared, expected):
            raise ValueError(
                "roi_cell_indices conflicts with rectangular extension construction; "
                "arbitrary mappings are not supported yet"
            )
        for patch in (*spec.inlet, *spec.outlet):
            if patch.side not in {"left", "right"}:
                raise ValueError(
                    "rectangular horizontal extension construction supports inlet/outlet "
                    "patches only on left or right sides"
                )

        self.spec = spec
        self.classification = spec.classification
        self.spacing = spec.dx
        self.nx, self.ny = spec.full_domain.nx, spec.full_domain.ny
        self.xmin, self.ymin = spec.full_domain.bounds.xmin, spec.full_domain.bounds.ymin
        self.xmax, self.ymax = spec.full_domain.bounds.xmax, spec.full_domain.bounds.ymax
        self.roi_bounds = spec.roi.bounds
        self.full_bounds = spec.full_domain.bounds
        self.left_extension_cells = spec.left_extension_cells
        self.right_extension_cells = spec.right_extension_cells
        self.roi_cell_indices = declared.ravel().copy()
        self.x = self.xmin + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.spacing
        self.y = self.ymin + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.spacing

        self.lambda_field = np.ones((self.ny, self.nx), dtype=np.float64)
        roi_slice = slice(self.left_extension_cells, self.left_extension_cells + spec.roi.nx)
        self.lambda_field[:, roi_slice] = roi_geometry.lambda_field
        for patch in (*spec.inlet, *spec.outlet):
            columns = (
                slice(0, self.left_extension_cells) if patch.side == "left"
                else slice(self.left_extension_cells + spec.roi.nx, self.nx)
            )
            for interval in patch.intervals:
                tolerance = max(1.0e-12, self.spacing * 1.0e-10)
                below_upper = self.y < interval.maximum - tolerance
                if np.isclose(
                    interval.maximum, self.ymax, atol=tolerance, rtol=0.0
                ):
                    below_upper = self.y <= interval.maximum + tolerance
                rows = (self.y >= interval.minimum - tolerance) & below_upper
                self.lambda_field[rows, columns] = 0.0

        self.signed_distance, self.interface, self.normals = _geometry_fields(
            self.lambda_field, self.spacing, interface_tolerance
        )
        reconstructed = 0.5 * (1.0 - np.tanh(self.signed_distance / self.spacing))
        difference = reconstructed - self.lambda_field
        self.reconstruction_error = {
            "relative_l2": float(
                np.linalg.norm(difference) / (np.linalg.norm(self.lambda_field) + 1.0e-30)
            ),
            "max_abs": float(np.max(np.abs(difference))),
        }

    def interpolate(
        self, coordinates: np.ndarray, field: str = "signed_distance"
    ) -> np.ndarray:
        return TPFMGeometry.interpolate(self, coordinates, field)

    def extract_roi(self, array: np.ndarray) -> np.ndarray:
        """Extract an ROI-shaped cell array using the declared flat mapping."""
        values = np.asarray(array)
        if values.shape != (self.ny, self.nx):
            raise ValueError(
                f"cell array must have shape ({self.ny}, {self.nx}), got {values.shape}"
            )
        return values.ravel()[self.roi_cell_indices].reshape(
            self.spec.roi.ny, self.spec.roi.nx
        )
