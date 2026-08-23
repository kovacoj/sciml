"""Firedrake state, selectable residual forms, and fixed Riesz maps."""

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
    SpatialCoordinate,
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
from firedrake.petsc import PETSc

from .domain import MANUFACTURED_EMPTY_CHANNEL
from .weak_forms import (
    continuity_weak_action,
    momentum_weak_action,
    navier_stokes_weak_form,
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
        formulation: str = "literal_strong_hfdib",
    ) -> None:
        if COMM_WORLD.size != 1:
            raise RuntimeError(
                "FiredrakeContext is intentionally serial; use experiment-level "
                "parallelism instead."
            )
        if formulation not in {"literal_strong_hfdib", "h1_weak"}:
            raise ValueError(f"unknown residual formulation: {formulation}")
        if (
            formulation == "h1_weak"
            and geometry.classification != MANUFACTURED_EMPTY_CHANNEL
        ):
            raise ValueError(
                "h1_weak is restricted to manufactured empty geometry (chi == 0)"
            )
        self.formulation = formulation
        self.xmin = float(getattr(geometry, "xmin", 0.0))
        self.ymin = float(getattr(geometry, "ymin", 0.0))
        self.xmax = float(getattr(geometry, "xmax", geometry.nx * geometry.spacing))
        self.ymax = float(getattr(geometry, "ymax", geometry.ny * geometry.spacing))
        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin
        self.domain_spec = getattr(geometry, "spec", None)
        if self.domain_spec is not None:
            inlet_marker, outlet_marker = self._patch_side_markers(self.domain_spec)
        self.ell = self.Ly if ell is None else float(ell)
        self.uin = float(uin)
        self.pout = float(pout)
        self.inlet_marker = int(inlet_marker)
        self.outlet_marker = int(outlet_marker)
        self.wall_markers = tuple(int(marker) for marker in wall_markers)
        self.quadrature_degree = int(quadrature_degree)
        self.mesh = RectangleMesh(nx, ny, self.Lx, self.Ly)
        if self.xmin != 0.0 or self.ymin != 0.0:
            self.mesh.coordinates.dat.data[:] += (self.xmin, self.ymin)
        self.S = FunctionSpace(self.mesh, "CG", velocity_degree)
        self.Q = FunctionSpace(self.mesh, "CG", pressure_degree)
        self.DG0 = FunctionSpace(self.mesh, "DG", 0)
        self.boundary_tolerance = max(1.0e-12, geometry.spacing * 1.0e-10)

        self.ux, self.uy = Function(self.S, name="ux"), Function(self.S, name="uy")
        self.p = Function(self.Q, name="p")
        self.uibx = Function(self.S, name="uibx")
        self.uiby = Function(self.S, name="uiby")
        self.lam = Function(self.DG0, name="lambda")
        self.chi = Function(self.DG0, name="chi")

        if self.domain_spec is None:
            inlet_nodes = np.asarray(
                DirichletBC(self.S, 0.0, self.inlet_marker).nodes,
                dtype=np.int64,
            )
            wall_nodes = np.unique(np.concatenate([
                DirichletBC(self.S, 0.0, marker).nodes
                for marker in self.wall_markers
            ]))
            pressure_outlet_nodes = np.asarray(
                DirichletBC(self.Q, 0.0, self.outlet_marker).nodes,
                dtype=np.int64,
            )
        else:
            self.s_coordinates = self._coordinates(self.S)
            self.q_coordinates = self._coordinates(self.Q)
            inlet = self._patch_selector(self.s_coordinates, self.domain_spec.inlet)
            outlet = self._patch_selector(self.s_coordinates, self.domain_spec.outlet)
            if np.any(inlet & outlet):
                raise ValueError("inlet and outlet patch intervals select the same S DOF")
            external = self._external_selector(self.s_coordinates)
            declared_wall = self._patch_selector(
                self.s_coordinates, self.domain_spec.wall
            )
            assert np.all((inlet | outlet | declared_wall)[external]), (
                "validated DomainSpec boundary partition must classify every external DOF"
            )
            inlet_nodes = np.flatnonzero(inlet)
            # Explicit walls win at shared corner DOFs, matching walls-after-inlet BCs.
            wall_nodes = np.flatnonzero(declared_wall)
            pressure_outlet_nodes = np.flatnonzero(
                self._patch_selector(self.q_coordinates, self.domain_spec.outlet)
            )
        self.velocity_boundary_nodes = np.union1d(inlet_nodes, wall_nodes)
        self.inlet_velocity_nodes = inlet_nodes
        self.velocity_wall_nodes = np.asarray(wall_nodes, dtype=np.int64)
        self.velocity_mask = np.ones(self.S.dim(), dtype=np.float64)
        self.velocity_mask[self.velocity_boundary_nodes] = 0.0
        self.ux_lift = np.zeros(self.S.dim(), dtype=np.float64)
        self.ux_lift[inlet_nodes] = self.uin
        self.ux_lift[wall_nodes] = 0.0
        self.geometric_pressure_outlet_nodes = pressure_outlet_nodes
        self.pressure_outlet_nodes = (
            np.empty(0, dtype=np.int64)
            if self.formulation == "h1_weak" else pressure_outlet_nodes
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

        self.continuity = div(u)
        vs, vq = TestFunction(self.S), TestFunction(self.Q)
        if self.formulation == "literal_strong_hfdib":
            self.F = momentum(u) + grad(self.p) - self.chi * (
                momentum(uib) + grad(self.p)
            )
            self.rx_form = self.F[0] * vs * measure
            self.ry_form = self.F[1] * vs * measure
            self.rc_form = self.continuity * vq * measure
        else:
            zero = Constant(0.0)
            self.F = None
            self.rx_form = navier_stokes_weak_form(
                u, self.p, as_vector((vs, zero * vs)), zero * vs,
                self.nu, self.beta, measure,
            )
            self.ry_form = navier_stokes_weak_form(
                u, self.p, as_vector((zero * vs, vs)), zero * vs,
                self.nu, self.beta, measure,
            )
            self.rc_form = navier_stokes_weak_form(
                u, self.p, as_vector((zero * vq, zero * vq)), vq,
                self.nu, self.beta, measure,
            )

        trial_s, test_s = TrialFunction(self.S), TestFunction(self.S)
        trial_q, test_q = TrialFunction(self.Q), TestFunction(self.Q)
        h1_form = (
            trial_s * test_s + self.ell**2 * inner(grad(trial_s), grad(test_s))
        ) * measure
        if self.domain_spec is None:
            self.velocity_test_bc = DirichletBC(
                self.S, 0.0, (self.inlet_marker, *self.wall_markers)
            )
            self.h1_matrix = assemble(h1_form, bcs=self.velocity_test_bc)
            self.velocity_test_is = None
        else:
            self.velocity_test_bc = None
            self.h1_matrix = assemble(h1_form)
            indices = np.asarray(
                self.velocity_boundary_nodes, dtype=PETSc.IntType
            )
            self.velocity_test_is = PETSc.IS().createGeneral(
                indices, comm=COMM_WORLD
            )
            matrix = self.h1_matrix.petscmat
            matrix.assemble()
            matrix.zeroRowsColumns(self.velocity_test_is, diag=1.0)
            matrix.assemble()
        self.q_mass_matrix = assemble(trial_q * test_q * measure)
        parameters = {"ksp_type": "preonly", "pc_type": "lu"}
        self.h1_solver = LinearSolver(self.h1_matrix, solver_parameters=parameters)
        self.q_solver = LinearSolver(self.q_mass_matrix, solver_parameters=parameters)
        self.yx, self.yy, self.yc = Function(self.S), Function(self.S), Function(self.Q)

        if self.formulation == "literal_strong_hfdib":
            self.weighted_objective_form = 2.0 * (
                self.inv_Cm * (self.F[0] * self.yx + self.F[1] * self.yy)
                + self.gamma_fd * self.inv_Cc * self.continuity * self.yc
            ) * measure
        else:
            self.weighted_objective_form = 2.0 * (
                self.inv_Cm * momentum_weak_action(
                    u, self.p, as_vector((self.yx, self.yy)),
                    self.nu, self.beta, measure,
                )
                + self.gamma_fd * self.inv_Cc * continuity_weak_action(
                    u, self.yc, measure
                )
            )
        coefficients = (self.ux, self.uy, self.p, self.uibx, self.uiby)
        differentiated = (
            coefficients
            if self.formulation == "literal_strong_hfdib"
            else coefficients[:3]
        )
        self.derivative_forms = tuple(
            derivative(self.weighted_objective_form, coefficient)
            for coefficient in differentiated
        )

    @staticmethod
    def _patch_side_markers(spec) -> tuple[int, int]:
        side_markers = {"left": 1, "right": 2, "bottom": 3, "top": 4}

        def one_side(patches, kind: str) -> str:
            sides = {patch.side for patch in patches}
            if len(sides) != 1:
                raise ValueError(
                    f"domain {kind} patches must all occupy one external side"
                )
            return next(iter(sides))

        inlet_side = one_side(spec.inlet, "inlet")
        outlet_side = one_side(spec.outlet, "outlet")
        if inlet_side == outlet_side:
            raise ValueError("domain inlet and outlet must occupy different sides")
        return side_markers[inlet_side], side_markers[outlet_side]

    @staticmethod
    def _coordinates(space) -> np.ndarray:
        coordinates = SpatialCoordinate(space.mesh())
        return np.column_stack([
            Function(space).interpolate(coordinates[index]).dat.data_ro.copy()
            for index in range(2)
        ])

    def _external_selector(self, points: np.ndarray) -> np.ndarray:
        tolerance = self.boundary_tolerance
        return (
            np.isclose(points[:, 0], self.xmin, atol=tolerance, rtol=0.0)
            | np.isclose(points[:, 0], self.xmax, atol=tolerance, rtol=0.0)
            | np.isclose(points[:, 1], self.ymin, atol=tolerance, rtol=0.0)
            | np.isclose(points[:, 1], self.ymax, atol=tolerance, rtol=0.0)
        )

    def _patch_selector(self, points: np.ndarray, patches) -> np.ndarray:
        selected = np.zeros(len(points), dtype=bool)
        tolerance = self.boundary_tolerance
        for patch in patches:
            if patch.side in {"left", "right"}:
                normal = points[:, 0]
                tangent = points[:, 1]
                side_value = self.xmin if patch.side == "left" else self.xmax
                global_upper = self.ymax
            else:
                normal = points[:, 1]
                tangent = points[:, 0]
                side_value = self.ymin if patch.side == "bottom" else self.ymax
                global_upper = self.xmax
            on_side = np.isclose(normal, side_value, atol=tolerance, rtol=0.0)
            for interval in patch.intervals:
                below_upper = tangent < interval.maximum - tolerance
                if np.isclose(
                    interval.maximum, global_upper, atol=tolerance, rtol=0.0
                ):
                    below_upper = tangent <= interval.maximum + tolerance
                selected |= (
                    on_side
                    & (tangent >= interval.minimum - tolerance)
                    & below_upper
                )
        return selected

    def velocity_constraints_at(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return velocity free mask and x lift at arbitrary physical points."""
        coordinates = np.asarray(points, dtype=np.float64)
        if self.domain_spec is None:
            marker_masks = {
                1: np.isclose(coordinates[:, 0], self.xmin, atol=1.0e-14, rtol=0.0),
                2: np.isclose(coordinates[:, 0], self.xmax, atol=1.0e-14, rtol=0.0),
                3: np.isclose(coordinates[:, 1], self.ymin, atol=1.0e-14, rtol=0.0),
                4: np.isclose(coordinates[:, 1], self.ymax, atol=1.0e-14, rtol=0.0),
            }
            inlet = marker_masks[self.inlet_marker]
            walls = np.logical_or.reduce([
                marker_masks[marker] for marker in self.wall_markers
            ])
            essential = inlet | walls
        else:
            inlet = self._patch_selector(coordinates, self.domain_spec.inlet)
            walls = self._patch_selector(coordinates, self.domain_spec.wall)
            essential = inlet | walls
        mask = np.ones(len(coordinates), dtype=np.float64)
        lift = np.zeros(len(coordinates), dtype=np.float64)
        mask[essential] = 0.0
        lift[inlet] = self.uin
        lift[walls] = 0.0
        return mask, lift

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
        gradients = tuple(
            np.asarray(assemble(form).dat.data_ro, dtype=np.float64).copy()
            for form in self.derivative_forms
        )
        if self.formulation == "h1_weak":
            zero = np.zeros(self.S.dim(), dtype=np.float64)
            return (*gradients, zero.copy(), zero.copy())
        return gradients
