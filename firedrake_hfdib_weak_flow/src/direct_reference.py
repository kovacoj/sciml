"""Independent conventional mixed-FE reference for manufactured benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from .benchmark_setup import load_config_geometry
from .domain import MANUFACTURED_CIRCULAR_HFDIB, MANUFACTURED_EMPTY_CHANNEL
from .train import atomic_json, relative_mass_imbalance


def run(config_path: Path, output_dir: Path) -> dict:
    config, _, _, _, geometry, context, _ = load_config_geometry(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if geometry.classification == MANUFACTURED_CIRCULAR_HFDIB:
        metrics = {
            "status": "deferred",
            "converged": False,
            "classification": geometry.classification,
            "reason": (
                "A literal strong-HFDIB reference requires a nonlocal u_ib fixed-point "
                "mapper; a conventional obstacle solve would not reference the article operator."
            ),
            "config": config,
        }
        atomic_json(output_dir / "direct_reference_metrics.json", metrics)
        return metrics
    if geometry.classification != MANUFACTURED_EMPTY_CHANNEL:
        raise ValueError("direct reference supports manufactured benchmarks only")

    from firedrake import (
        Constant, DirichletBC, FacetNormal, Function, FunctionSpace,
        NonlinearVariationalProblem, NonlinearVariationalSolver, SpatialCoordinate,
        TestFunctions, VectorFunctionSpace, as_vector, assemble, div, dot, ds, dx,
        grad, inner, outer, split, sym,
    )

    mesh = context.mesh
    velocity_degree = int(config.get("velocity_degree", 2))
    pressure_degree = int(config.get("pressure_degree", 1))
    V = VectorFunctionSpace(mesh, "CG", velocity_degree)
    Q = FunctionSpace(mesh, "CG", pressure_degree)
    W = V * Q
    solution = Function(W, name="direct_reference")
    u, p = split(solution)
    v, q = TestFunctions(W)
    normal = FacetNormal(mesh)
    nu = Constant(geometry.spec.nu)
    # Weak conservative convection retains its boundary flux; symmetric stress has
    # the natural zero-traction condition on the outlet.
    residual = (
        -inner(outer(u, u), grad(v)) * dx
        + dot(u, normal) * dot(u, v) * ds
        + 2.0 * nu * inner(sym(grad(u)), sym(grad(v))) * dx
        - p * div(v) * dx
        + q * div(u) * dx
    )
    zero = Constant((0.0, 0.0))
    bcs = [
        DirichletBC(W.sub(0), Constant((geometry.spec.uin, 0.0)), 1),
        DirichletBC(W.sub(0), zero, (3, 4)),
    ]
    x = SpatialCoordinate(mesh)
    initial_u = Function(V).interpolate(as_vector((
        6.0 * geometry.spec.uin * x[1] * (context.Ly - x[1]) / context.Ly**2,
        0.0,
    )))
    solution.sub(0).interpolate(initial_u)
    problem = NonlinearVariationalProblem(residual, solution, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters={
        "snes_type": "newtonls",
        "snes_rtol": 1.0e-11,
        "snes_atol": 1.0e-12,
        "snes_max_it": 40,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_type": "aij",
    })
    started = time.monotonic()
    solver.solve()
    elapsed = time.monotonic() - started
    reason = int(solver.snes.getConvergedReason())
    iterations = int(solver.snes.getIterationNumber())
    velocity, pressure = solution.subfunctions
    ux = Function(context.S, name="ux").interpolate(velocity[0])
    uy = Function(context.S, name="uy").interpolate(velocity[1])
    pressure_sample = Function(context.Q, name="p").interpolate(pressure)
    inlet_flux = float(assemble(dot(velocity, normal) * ds(1)))
    outlet_flux = float(assemble(dot(velocity, normal) * ds(2)))
    speed = np.hypot(ux.dat.data_ro, uy.dat.data_ro)
    metrics = {
        "status": "converged" if reason > 0 else "failed",
        "converged": reason > 0,
        "snes_reason": reason,
        "newton_iterations": iterations,
        "elapsed_seconds": elapsed,
        "classification": geometry.classification,
        "pressure_condition": "natural_zero_traction_outlet",
        "divergence_l2": float(assemble(div(velocity) ** 2 * dx) ** 0.5),
        "inlet_flux": inlet_flux,
        "outlet_flux": outlet_flux,
        "mass_imbalance": relative_mass_imbalance(inlet_flux, outlet_flux),
        "max_speed": float(speed.max()),
        "pressure_min": float(pressure_sample.dat.data_ro.min()),
        "pressure_max": float(pressure_sample.dat.data_ro.max()),
        "l2_name": "unweighted_FE_coefficient_l2",
        "config": config,
    }
    np.savez_compressed(
        output_dir / "direct_reference.npz",
        ux=ux.dat.data_ro.copy(), uy=uy.dat.data_ro.copy(),
        p=pressure_sample.dat.data_ro.copy(),
        s_coordinates=context._coordinates(context.S),
        q_coordinates=context._coordinates(context.Q),
    )
    atomic_json(output_dir / "direct_reference_metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    main()
