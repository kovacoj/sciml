"""Direct mixed-FE solve for the analytic Brinkman topology family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .brinkman_mesh import INLET_MARKER, OUTLET_MARKER, WALL_MARKER, write_segmented_square_mesh
from .brinkman_topologies import BrinkmanTopologyGeometry
from .train import relative_mass_imbalance


def run(
    topology_name: str,
    resolution: int,
    alpha: float,
    beta: float,
    output: Path,
    topology_field: str = "diffuse_lambda",
) -> dict:
    from firedrake import (
        Constant, DirichletBC, FacetNormal, Function, FunctionSpace, Mesh,
        NonlinearVariationalProblem, NonlinearVariationalSolver, SpatialCoordinate,
        TestFunctions, VectorFunctionSpace, assemble, div, dot, ds, dx, split,
    )
    from .weak_forms import navier_stokes_brinkman_weak_form

    output.mkdir(parents=True, exist_ok=True)
    mesh_path = write_segmented_square_mesh(output / "mesh.msh", resolution)
    mesh = Mesh(str(mesh_path))
    geometry = BrinkmanTopologyGeometry(topology_name)
    velocity_space = VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = FunctionSpace(mesh, "CG", 1)
    topology_space = FunctionSpace(mesh, "DG", 0)
    mixed = velocity_space * pressure_space
    state = Function(mixed, name="direct_brinkman")
    u, p = split(state)
    v, q = TestFunctions(mixed)
    lam = Function(topology_space, name="lambda")
    spatial = SpatialCoordinate(mesh)
    coordinates = np.column_stack([
        Function(topology_space).interpolate(spatial[index]).dat.data_ro.copy()
        for index in range(2)
    ])
    lambda_values = geometry.interpolate(coordinates, "lambda")
    if topology_field == "diffuse_lambda":
        lam.dat.data[:] = lambda_values
    elif topology_field == "sharp_chi":
        lam.dat.data[:] = (lambda_values > 0.5).astype(np.float64)
    else:
        raise ValueError("topology_field must be diffuse_lambda or sharp_chi")
    residual = navier_stokes_brinkman_weak_form(
        u, p, v, q, Constant(0.01), Constant(beta), Constant(alpha), lam, dx,
    )
    bcs = [
        DirichletBC(mixed.sub(0), Constant((0.1, 0.0)), INLET_MARKER),
        DirichletBC(mixed.sub(0), Constant((0.0, 0.0)), WALL_MARKER),
    ]
    problem = NonlinearVariationalProblem(residual, state, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "snes_type": "newtonls", "snes_rtol": 1.0e-11, "snes_atol": 1.0e-12,
        "snes_max_it": 40, "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps", "mat_type": "aij",
    })
    started = time.monotonic()
    solver.solve()
    elapsed = time.monotonic() - started
    velocity, pressure = state.subfunctions
    normal = FacetNormal(mesh)
    inlet_flux = float(assemble(dot(velocity, normal) * ds(INLET_MARKER)))
    outlet_flux = float(assemble(dot(velocity, normal) * ds(OUTLET_MARKER)))
    solid = float(assemble(lam * dot(velocity, velocity) * dx))
    fluid = float(assemble((1.0 - lam) * dot(velocity, velocity) * dx))
    report = {
        "topology": topology_name, "resolution": resolution,
        "alpha": alpha, "beta": beta, "topology_field": topology_field,
        "status": "converged" if solver.snes.getConvergedReason() > 0 else "failed",
        "snes_reason": int(solver.snes.getConvergedReason()),
        "newton_iterations": int(solver.snes.getIterationNumber()),
        "elapsed_seconds": elapsed,
        "inlet_flux": inlet_flux, "outlet_flux": outlet_flux,
        "mass_imbalance": relative_mass_imbalance(inlet_flux, outlet_flux),
        "divergence_l2": float(assemble(div(velocity) ** 2 * dx) ** 0.5),
        "solid_leakage": float(np.sqrt(solid) / (np.sqrt(fluid) + 1.0e-30)),
        "claim": "variational Brinkman approximation, not article HFDIB",
    }
    np.savez_compressed(
        output / "direct_fields.npz",
        velocity=velocity.dat.data_ro.copy(), pressure=pressure.dat.data_ro.copy(),
        velocity_coordinates=np.column_stack([
            Function(FunctionSpace(mesh, "CG", 2)).interpolate(spatial[index]).dat.data_ro.copy()
            for index in range(2)
        ]),
        pressure_coordinates=np.column_stack([
            Function(pressure_space).interpolate(spatial[index]).dat.data_ro.copy()
            for index in range(2)
        ]),
        lambda_dg=lam.dat.data_ro.copy(), lambda_coordinates=coordinates,
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=list("ABC"), default="A")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topology-field", choices=("diffuse_lambda", "sharp_chi"),
        default="diffuse_lambda",
    )
    arguments = parser.parse_args()
    run(
        arguments.topology, arguments.resolution, arguments.alpha, arguments.beta,
        arguments.output, arguments.topology_field,
    )


if __name__ == "__main__":
    main()
