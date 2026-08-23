"""Direct Taylor-Hood Stokes solve on fitted channel Topology A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .brinkman_mesh import INLET_MARKER, OUTLET_MARKER, WALL_MARKER
from .fitted_channel_mesh import write_fitted_topology_a
from .train import relative_mass_imbalance


def run(output: Path, nominal_h: float = 0.004) -> dict:
    from firedrake import (
        Constant, DirichletBC, FacetNormal, Function, FunctionSpace, Mesh,
        NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunctions,
        VectorFunctionSpace, assemble, div, dot, ds, dx, split,
    )
    from .weak_forms import navier_stokes_weak_form

    output.mkdir(parents=True, exist_ok=True)
    mesh = Mesh(str(write_fitted_topology_a(output / "mesh.msh", nominal_h)))
    velocity_space = VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = FunctionSpace(mesh, "CG", 1)
    mixed = velocity_space * pressure_space
    state = Function(mixed, name="fitted_topology_A")
    u, p = split(state)
    v, q = TestFunctions(mixed)
    residual = navier_stokes_weak_form(
        u, p, v, q, Constant(0.01), Constant(0.0), dx
    )
    bcs = [
        DirichletBC(mixed.sub(0), Constant((0.1, 0.0)), INLET_MARKER),
        DirichletBC(mixed.sub(0), Constant((0.0, 0.0)), WALL_MARKER),
    ]
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(residual, state, bcs=bcs),
        solver_parameters={
            "snes_type": "newtonls", "snes_rtol": 1e-11, "snes_atol": 1e-12,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps", "mat_type": "aij",
        },
    )
    started = time.monotonic(); solver.solve(); elapsed = time.monotonic() - started
    velocity, pressure = state.subfunctions
    assembled_residual = assemble(residual, bcs=bcs)
    with assembled_residual.dat.vec_ro as residual_vector:
        direct_residual_l2 = float(residual_vector.norm())
    normal = FacetNormal(mesh)
    inlet_flux = float(assemble(dot(velocity, normal) * ds(INLET_MARKER)))
    outlet_flux = float(assemble(dot(velocity, normal) * ds(OUTLET_MARKER)))
    report = {
        "topology": "A_fitted_level_set_channels",
        "nominal_h": nominal_h,
        "status": "converged" if solver.snes.getConvergedReason() > 0 else "failed",
        "snes_reason": int(solver.snes.getConvergedReason()),
        "elapsed_seconds": elapsed,
        "inlet_flux": inlet_flux, "outlet_flux": outlet_flux,
        "mass_imbalance": relative_mass_imbalance(inlet_flux, outlet_flux),
        "divergence_l2": float(assemble(div(velocity) ** 2 * dx) ** 0.5),
        "assembled_weak_residual_l2": direct_residual_l2,
        "velocity_dofs": velocity_space.dim(), "pressure_dofs": pressure_space.dim(),
        "claim": "fitted level-set-derived channel domain; not article reproduction",
    }
    np.savez_compressed(
        output / "direct_fields.npz",
        velocity=velocity.dat.data_ro.copy(), pressure=pressure.dat.data_ro.copy(),
        velocity_coordinates=np.column_stack([
            Function(FunctionSpace(mesh, "CG", 2)).interpolate(
                __import__("firedrake").SpatialCoordinate(mesh)[index]
            ).dat.data_ro.copy() for index in range(2)
        ]),
        pressure_coordinates=np.column_stack([
            Function(pressure_space).interpolate(
                __import__("firedrake").SpatialCoordinate(mesh)[index]
            ).dat.data_ro.copy() for index in range(2)
        ]),
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-h", type=float, default=0.004)
    arguments = parser.parse_args()
    run(arguments.output, arguments.nominal_h)


if __name__ == "__main__":
    main()
