"""Firedrake state, strong residual forms, and fixed Riesz maps."""

from __future__ import annotations

import numpy as np
from firedrake import (
    COMM_WORLD,
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    LinearSolver,
    RectangleMesh,
    TestFunction,
    TrialFunction,
    as_vector,
    assemble,
    derivative,
    div,
    ds,
    dx,
    grad,
    inner,
    outer,
    transpose,
)


class FiredrakeContext:
    """Serial flow coefficients, residuals, Riesz maps, and derivative forms."""

    def __init__(
        self,
        geometry,
        nx: int,
        ny: int,
        *,
        velocity_degree: int = 2,
        pressure_degree: int = 1,
        quadrature_degree: int = 6,
        nu: float = 0.01,
        ell: float | None = None,
        uin: float = 0.1,
        pout: float = 0.0,
        inlet_marker: int = 1,
        outlet_marker: int = 2,
        wall_markers: list[int] | tuple[int, ...] = (3, 4),
    ) -> None:
        if COMM_WORLD.size != 1:
            raise RuntimeError(
                "FiredrakeContext is intentionally serial; use experiment-level "
                "parallelism instead."
            )
        self.xmin = self.ymin = 0.0
        self.Lx = geometry.nx * geometry.spacing
        self.Ly = geometry.ny * geometry.spacing
        self.xmax, self.ymax = self.Lx, self.Ly
        self.ell = self.Ly if ell is None else float(ell)
        self.uin = float(uin)
        self.pout = float(pout)
        self.inlet_marker = int(inlet_marker)
        self.outlet_marker = int(outlet_marker)
        self.wall_markers = tuple(int(marker) for marker in wall_markers)
        self.quadrature_degree = int(quadrature_degree)
        self.mesh = RectangleMesh(nx, ny, self.Lx, self.Ly)
        self.S = FunctionSpace(self.mesh, "CG", velocity_degree)
        self.Q = FunctionSpace(self.mesh, "CG", pressure_degree)
        self.DG0 = FunctionSpace(self.mesh, "DG", 0)

        self.ux, self.uy = Function(self.S, name="ux"), Function(self.S, name="uy")
        self.p = Function(self.Q, name="p")
        self.uibx = Function(self.S, name="uibx")
        self.uiby = Function(self.S, name="uiby")
        self.lam = Function(self.DG0, name="lambda")
        self.chi = Function(self.DG0, name="chi")

        inlet_nodes = DirichletBC(self.S, 0.0, self.inlet_marker).nodes
        wall_nodes = np.unique(np.concatenate([
            DirichletBC(self.S, 0.0, marker).nodes
            for marker in self.wall_markers
        ]))
        self.velocity_boundary_nodes = np.union1d(inlet_nodes, wall_nodes)
        self.velocity_mask = np.ones(self.S.dim(), dtype=np.float64)
        self.velocity_mask[self.velocity_boundary_nodes] = 0.0
        self.ux_lift = np.zeros(self.S.dim(), dtype=np.float64)
        self.ux_lift[inlet_nodes] = self.uin
        self.ux_lift[wall_nodes] = 0.0
        self.pressure_outlet_nodes = np.asarray(
            DirichletBC(self.Q, 0.0, self.outlet_marker).nodes, dtype=np.int64
        )
        self.pressure_mask = np.ones(self.Q.dim(), dtype=np.float64)
        self.pressure_mask[self.pressure_outlet_nodes] = 0.0

        self.nu = Constant(nu)
        self.beta = Constant(0.0)
        self.inv_Cm = Constant(1.0)
        self.inv_Cc = Constant(1.0)
        self.gamma_fd = Constant(1.0)
        self.measure = dx(metadata={"quadrature_degree": self.quadrature_degree})
        self.boundary_measure = ds(
            domain=self.mesh,
            metadata={"quadrature_degree": self.quadrature_degree},
        )
        measure = self.measure
        u = as_vector((self.ux, self.uy))
        uib = as_vector((self.uibx, self.uiby))

        def momentum(w):
            return self.beta * div(outer(w, w)) - div(
                self.nu * (grad(w) + transpose(grad(w)))
            )

        self.F = momentum(u) + grad(self.p) - self.chi * (
            momentum(uib) + grad(self.p)
        )
        self.continuity = div(u)
        vs, vq = TestFunction(self.S), TestFunction(self.Q)
        self.rx_form = self.F[0] * vs * measure
        self.ry_form = self.F[1] * vs * measure
        self.rc_form = self.continuity * vq * measure

        trial_s, test_s = TrialFunction(self.S), TestFunction(self.S)
        trial_q, test_q = TrialFunction(self.Q), TestFunction(self.Q)
        self.velocity_test_bc = DirichletBC(
            self.S, 0.0, (self.inlet_marker, *self.wall_markers)
        )
        self.h1_matrix = assemble(
            (trial_s * test_s + self.ell**2 * inner(grad(trial_s), grad(test_s)))
            * measure,
            bcs=self.velocity_test_bc,
        )
        self.q_mass_matrix = assemble(trial_q * test_q * measure)
        parameters = {"ksp_type": "preonly", "pc_type": "lu"}
        self.h1_solver = LinearSolver(self.h1_matrix, solver_parameters=parameters)
        self.q_solver = LinearSolver(self.q_mass_matrix, solver_parameters=parameters)
        self.yx, self.yy, self.yc = Function(self.S), Function(self.S), Function(self.Q)

        self.weighted_objective_form = 2.0 * (
            self.inv_Cm * (self.F[0] * self.yx + self.F[1] * self.yy)
            + self.gamma_fd * self.inv_Cc * self.continuity * self.yc
        ) * measure
        coefficients = (self.ux, self.uy, self.p, self.uibx, self.uiby)
        self.derivative_forms = tuple(
            derivative(self.weighted_objective_form, coefficient)
            for coefficient in coefficients
        )

    def assign_boundary_lift(self) -> None:
        self.ux.dat.data[:] = self.ux_lift
        self.uy.assign(0.0)
        self.p.assign(self.pout)
        self.uibx.assign(0.0)
        self.uiby.assign(0.0)

    def residual_vectors(self):
        rx, ry, rc = assemble(self.rx_form), assemble(self.ry_form), assemble(self.rc_form)
        rx.dat.data[self.velocity_boundary_nodes] = 0.0
        ry.dat.data[self.velocity_boundary_nodes] = 0.0
        return rx, ry, rc

    def solve_riesz(self):
        residuals = self.residual_vectors()
        self.h1_solver.solve(self.yx, residuals[0])
        self.h1_solver.solve(self.yy, residuals[1])
        self.q_solver.solve(self.yc, residuals[2])
        return residuals

    def raw_losses(self, residuals) -> tuple[float, float, float]:
        values = []
        for y, residual in zip((self.yx, self.yy, self.yc), residuals):
            with y.dat.vec_ro as y_vec, residual.dat.vec_ro as r_vec:
                values.append(float(y_vec.dot(r_vec)))
        return tuple(values)

    def assemble_gradients(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray(assemble(form).dat.data_ro, dtype=np.float64).copy()
            for form in self.derivative_forms
        )
