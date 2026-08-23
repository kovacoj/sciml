"""Numerical bridge between graph-connected fields and Firedrake residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import NeuralFields


@dataclass(frozen=True)
class ResidualMetrics:
    loss: float
    loss_x: float
    loss_y: float
    loss_continuity: float
    raw_loss_x: float
    raw_loss_y: float
    raw_loss_continuity: float
    residual_x_l2: float
    residual_y_l2: float
    residual_continuity_l2: float


@dataclass(frozen=True)
class FEGradients:
    ux: np.ndarray
    uy: np.ndarray
    p: np.ndarray
    uibx: np.ndarray
    uiby: np.ndarray


class FiredrakeResidualBridge:
    """Assign fields and expose loss metrics plus coefficient gradients."""

    def __init__(self, context, gamma: float = 1.0) -> None:
        self.context = context
        self.gamma = float(gamma)
        self.Cm: float | None = None
        self.Cc: float | None = None
        context.gamma_fd.assign(self.gamma)

    def _assign(self, fields: NeuralFields) -> None:
        for coefficient, value in zip(
            (self.context.ux, self.context.uy, self.context.p,
             self.context.uibx, self.context.uiby),
            (fields.ux, fields.uy, fields.p, fields.uibx, fields.uiby),
        ):
            coefficient.dat.data[:] = value.detach().cpu().numpy()

    def initialize_normalization(self, fields: NeuralFields) -> tuple[float, float]:
        normalization = self.compute_baseline_normalization()
        self._assign(fields)
        return normalization

    def compute_baseline_normalization(self) -> tuple[float, float]:
        self.context.assign_boundary_lift()
        self.context.beta.assign(0.0)
        residuals = self.context.solve_riesz()
        Lx, Ly, Lc = self.context.raw_losses(residuals)
        self.Cm = max(Lx + Ly, 1.0e-12)
        self.Cc = max(Lc, 1.0e-12)
        self.context.inv_Cm.assign(1.0 / self.Cm)
        self.context.inv_Cc.assign(1.0 / self.Cc)
        return self.Cm, self.Cc

    def _evaluate(self, fields: NeuralFields, beta: float) -> tuple[ResidualMetrics, tuple]:
        if self.Cm is None or self.Cc is None:
            raise RuntimeError("call initialize_normalization(fields) first")
        self._assign(fields)
        self.context.beta.assign(beta)
        residuals = self.context.solve_riesz()
        Lx, Ly, Lc = self.context.raw_losses(residuals)
        loss_x, loss_y = Lx / self.Cm, Ly / self.Cm
        loss_continuity = Lc / self.Cc
        residual_l2 = tuple(float(np.linalg.norm(r.dat.data_ro)) for r in residuals)
        metrics = ResidualMetrics(
            loss_x + loss_y + self.gamma * loss_continuity,
            loss_x,
            loss_y,
            loss_continuity,
            Lx,
            Ly,
            Lc,
            *residual_l2,
        )
        return metrics, residuals

    def evaluate_loss(self, fields: NeuralFields, beta: float) -> ResidualMetrics:
        return self._evaluate(fields, beta)[0]

    def evaluate_loss_and_gradients(
        self, fields: NeuralFields, beta: float
    ) -> tuple[ResidualMetrics, FEGradients]:
        metrics, _ = self._evaluate(fields, beta)
        gradients = list(self.context.assemble_gradients())
        nodes = self.context.velocity_boundary_nodes
        gradients[0][nodes] = 0.0
        gradients[1][nodes] = 0.0
        gradients[2][self.context.pressure_outlet_nodes] = 0.0
        return metrics, FEGradients(*(
            np.asarray(value, dtype=np.float64).copy() for value in gradients
        ))
