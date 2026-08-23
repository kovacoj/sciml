"""Differentiable neural fields at Firedrake coefficient locations."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import torch
from firedrake import Function, SpatialCoordinate

from .domain import RECONSTRUCTED_TPFM_DOMAIN
from .geometry import TPFMGeometry
from .hfdib import second_order


U_SCALE = 0.1
P_SCALE = 0.01


@dataclass
class NeuralFields:
    ux: torch.Tensor
    uy: torch.Tensor
    p: torch.Tensor
    uibx: torch.Tensor
    uiby: torch.Tensor


class FEFieldMapper:
    """Evaluate a coordinate network at exact finite-element DOF coordinates."""

    def __init__(
        self,
        context,
        geometry: TPFMGeometry,
        *,
        u_scale: float = U_SCALE,
        p_scale: float = P_SCALE,
        interface_tolerance: float = 1.0e-12,
    ) -> None:
        self.context = context
        self.geometry = geometry
        self.u_scale = float(u_scale)
        self.p_scale = float(p_scale)
        self.interface_tolerance = float(interface_tolerance)
        self.s_coords = self._coordinates(context.S)
        self.q_coords = self._coordinates(context.Q)
        self.dg0_coords = self._coordinates(context.DG0)

        lam = self._sample(self.dg0_coords, "lambda")
        chi = (lam > self.interface_tolerance).astype(np.float64)
        assert np.all((chi == 0.0) | (chi == 1.0)), "chi must be binary"
        context.lam.dat.data[:] = lam
        context.chi.dat.data[:] = chi

        self.s_features = self._feature_array(self.s_coords)
        self.q_features = self._feature_array(self.q_coords)
        self.s_lambda = self._sample(self.s_coords, "lambda")
        self.s_sigma = self._sample(self.s_coords, "signed_distance")
        self.s_normals = self._sample(self.s_coords, "normals")
        inlet_nodes = context.inlet_velocity_nodes
        prescribed_nonzero = inlet_nodes[
            np.abs(context.ux_lift[inlet_nodes]) > 0.0
        ]
        self.inlet_geometry_conflicts = int(np.count_nonzero(
            self.s_lambda[prescribed_nonzero] > self.interface_tolerance
        ))
        if (
            self.inlet_geometry_conflicts
            and geometry.classification != RECONSTRUCTED_TPFM_DOMAIN
        ):
            warnings.warn(
                f"hard inlet overlaps solid/interface geometry at "
                f"{self.inlet_geometry_conflicts} velocity DOFs; the cropped "
                "TPFM area of interest does not define a compatible full-side inlet",
                RuntimeWarning,
                stacklevel=2,
            )
        boundary = self.s_coords - self.s_sigma[:, None] * self.s_normals
        self.d1 = np.full(self.s_sigma.shape, geometry.spacing)
        self.d2 = np.full(self.s_sigma.shape, geometry.spacing)
        self.first_points = boundary + self.d1[:, None] * self.s_normals
        self.second_points = boundary + (self.d1 + self.d2)[:, None] * self.s_normals
        self.first_features = self._feature_array(self.first_points)
        self.second_features = self._feature_array(self.second_points)
        self.first_velocity_mask, self.first_ux_lift = self._velocity_constraints(
            self.first_points
        )
        self.second_velocity_mask, self.second_ux_lift = self._velocity_constraints(
            self.second_points
        )

    @staticmethod
    def _coordinates(space) -> np.ndarray:
        x = SpatialCoordinate(space.mesh())
        return np.column_stack([
            Function(space).interpolate(x[i]).dat.data_ro.copy() for i in range(2)
        ])

    def _sample(self, points: np.ndarray, field: str) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64).copy()
        query[..., 0] = np.clip(query[..., 0], self.geometry.x[0], self.geometry.x[-1])
        query[..., 1] = np.clip(query[..., 1], self.geometry.y[0], self.geometry.y[-1])
        return np.asarray(self.geometry.interpolate(query, field), dtype=np.float64)

    def _feature_array(self, points: np.ndarray) -> np.ndarray:
        c = self.context
        return np.column_stack((
            2.0 * (points[:, 0] - c.xmin) / c.Lx - 1.0,
            2.0 * (points[:, 1] - c.ymin) / c.Ly - 1.0,
            self._sample(points, "lambda"),
            self._sample(points, "signed_distance") / self.geometry.spacing,
        ))

    def _velocity_constraints(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.context.velocity_constraints_at(points)

    def _velocity(
        self,
        model,
        points: np.ndarray,
        features: np.ndarray,
        mask: np.ndarray,
        ux_lift: np.ndarray,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw = model(torch.as_tensor(features, dtype=dtype))
        hard_mask = torch.as_tensor(mask, dtype=dtype)
        lift = torch.as_tensor(ux_lift, dtype=dtype)
        return torch.column_stack((
            lift + self.u_scale * hard_mask * raw[:, 0],
            self.u_scale * hard_mask * raw[:, 1],
        ))

    def evaluate(self, model) -> NeuralFields:
        """Return graph-connected fields without assigning Firedrake coefficients."""
        dtype = next(model.parameters()).dtype
        velocity = self._velocity(
            model,
            self.s_coords,
            self.s_features,
            self.context.velocity_mask,
            self.context.ux_lift,
            dtype,
        )
        q_raw = model(torch.as_tensor(self.q_features, dtype=dtype))[:, 2]
        pressure = self.context.pout + self.p_scale * torch.as_tensor(
            self.context.pressure_mask, dtype=dtype
        ) * q_raw

        first_velocity = self._velocity(
            model,
            self.first_points,
            self.first_features,
            self.first_velocity_mask,
            self.first_ux_lift,
            dtype,
        )
        second_velocity = self._velocity(
            model,
            self.second_points,
            self.second_features,
            self.second_velocity_mask,
            self.second_ux_lift,
            dtype,
        )
        uib = second_order(
            0.0,
            first_velocity,
            second_velocity,
            torch.as_tensor(self.s_sigma, dtype=dtype),
            torch.as_tensor(self.d1, dtype=dtype),
            torch.as_tensor(self.d2, dtype=dtype),
        )
        interface = torch.as_tensor(
            (self.s_lambda > self.interface_tolerance)
            & (self.s_lambda < 1.0 - self.interface_tolerance)
        )
        uib = torch.where(interface[:, None], uib, torch.zeros_like(uib))
        return NeuralFields(
            velocity[:, 0], velocity[:, 1], pressure, uib[:, 0], uib[:, 1]
        )


def enforce_inlet_geometry_compatibility(mapper: FEFieldMapper) -> None:
    """Hard-gate incompatible prescribed inlet flow on a reconstructed domain."""
    if (
        mapper.geometry.classification == RECONSTRUCTED_TPFM_DOMAIN
        and mapper.inlet_geometry_conflicts
    ):
        raise RuntimeError(
            f"reconstructed full domain has {mapper.inlet_geometry_conflicts} nonzero "
            "inlet velocity DOFs where lambda exceeds chi_eps"
        )
