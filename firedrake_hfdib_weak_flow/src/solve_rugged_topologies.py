"""Sequential resumable direct CFD solves for selected rugged 64x64 masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .brinkman_mesh import INLET_MARKER, OUTLET_MARKER, WALL_MARKER
from .rugged_bitmap_mesh import H, ROI_N, STUB_CELLS, write_pixel_union_mesh
from .train import relative_mass_imbalance


def solve_case(case_dir: Path) -> dict:
    from firedrake import (
        CheckpointFile, Constant, DirichletBC, FacetNormal, Function,
        FunctionSpace, Mesh, NonlinearVariationalProblem,
        NonlinearVariationalSolver, SpatialCoordinate, TestFunctions,
        VectorFunctionSpace, assemble, div, dot, ds, dx, split,
    )
    from .weak_forms import navier_stokes_weak_form

    done = case_dir / "DONE.json"
    if done.exists():
        return json.loads((case_dir / "metrics.json").read_text())
    lam = np.load(case_dir / "lambda_64x64.npy")
    binary_fluid = lam < 0.5
    mesh_path = write_pixel_union_mesh(case_dir / "mesh.msh", binary_fluid)
    mesh = Mesh(str(mesh_path))
    velocity_space = VectorFunctionSpace(mesh, "CG", 2)
    pressure_space = FunctionSpace(mesh, "CG", 1)
    mixed = velocity_space * pressure_space
    state = Function(mixed, name=case_dir.name)
    u, p = split(state); v, q = TestFunctions(mixed)
    bcs = [
        DirichletBC(mixed.sub(0), Constant((0.1, 0.0)), INLET_MARKER),
        DirichletBC(mixed.sub(0), Constant((0.0, 0.0)), WALL_MARKER),
    ]
    stages, elapsed = [], 0.0
    for beta in (0.0, 0.25, 0.5, 1.0):
        residual = navier_stokes_weak_form(
            u, p, v, q, Constant(0.01), Constant(beta), dx
        )
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(residual, state, bcs=bcs),
            solver_parameters={
                "snes_type": "newtonls", "snes_rtol": 1e-10,
                "snes_atol": 1e-11, "snes_max_it": 35,
                "ksp_type": "preonly", "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps", "mat_type": "aij",
            },
        )
        started = time.perf_counter(); solver.solve(); seconds = time.perf_counter() - started
        reason = int(solver.snes.getConvergedReason())
        if reason <= 0:
            raise RuntimeError(f"{case_dir.name} beta={beta} failed: SNES {reason}")
        elapsed += seconds
        stages.append({
            "beta": beta, "seconds": seconds, "reason": reason,
            "nonlinear_iterations": int(solver.snes.getIterationNumber()),
            "linear_iterations": int(solver.snes.getLinearSolveIterations()),
        })
        with CheckpointFile(str(case_dir / f"beta_{str(beta).replace('.', 'p')}.h5"), "w") as checkpoint:
            checkpoint.save_mesh(mesh); checkpoint.save_function(state, name="state")
        (case_dir / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")

    velocity, pressure = state.subfunctions
    residual = navier_stokes_weak_form(
        u, p, v, q, Constant(0.01), Constant(1.0), dx
    )
    assembled = assemble(residual, bcs=bcs)
    with assembled.dat.vec_ro as vector:
        weak = float(vector.norm())
    normal = FacetNormal(mesh)
    q_in_signed = float(assemble(dot(velocity, normal) * ds(INLET_MARKER)))
    q_out_signed = float(assemble(dot(velocity, normal) * ds(OUTLET_MARKER)))
    velocity_p1 = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(velocity)
    pressure_p1 = Function(pressure_space).interpolate(pressure)
    coordinates = mesh.coordinates.dat.data_ro.copy()
    cells = mesh.coordinates.function_space().cell_node_map().values.copy()
    velocity_values = velocity_p1.dat.data_ro.copy()
    pressure_values = pressure_p1.dat.data_ro.copy()
    if not (np.all(np.isfinite(velocity_values)) and np.all(np.isfinite(pressure_values))):
        raise FloatingPointError(f"{case_dir.name} produced non-finite fields")
    np.savez_compressed(
        case_dir / "fields.npz", coordinates=coordinates, cells=cells,
        velocity=velocity_values, pressure=pressure_values,
    )
    with CheckpointFile(str(case_dir / "solution.h5"), "w") as checkpoint:
        checkpoint.save_mesh(mesh); checkpoint.save_function(velocity, name="velocity")
        checkpoint.save_function(pressure, name="pressure")

    # Rasterize only the central ROI. Firedrake point evaluation returns NaN on
    # solid pixels because they lie outside the fitted fluid mesh.
    x = (np.arange(ROI_N) + 0.5) * H
    y = (np.arange(ROI_N) + 0.5) * H
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    ux = np.full(len(points), np.nan); uy = np.full(len(points), np.nan)
    pressure_raster = np.full(len(points), np.nan)
    fluid_flat = binary_fluid.ravel()
    fluid_points = points[fluid_flat]
    evaluated_u = np.asarray(velocity.at(fluid_points, tolerance=1e-10))
    evaluated_p = np.asarray(pressure.at(fluid_points, tolerance=1e-10))
    ux[fluid_flat] = evaluated_u[:, 0]; uy[fluid_flat] = evaluated_u[:, 1]
    pressure_raster[fluid_flat] = evaluated_p
    pressure_raster[fluid_flat] -= np.nanmean(pressure_raster[fluid_flat])
    np.savez_compressed(
        case_dir / "raster_64x64.npz", lambda_field=lam,
        fluid_mask=binary_fluid, ux=ux.reshape(ROI_N, ROI_N),
        uy=uy.reshape(ROI_N, ROI_N),
        velocity_magnitude=np.hypot(ux, uy).reshape(ROI_N, ROI_N),
        pressure=pressure_raster.reshape(ROI_N, ROI_N), x=x, y=y,
    )
    speed = np.linalg.norm(velocity_values, axis=1)
    report = {
        "case": case_dir.name, "status": "CONVERGED",
        "mesh_cells": int(mesh.cell_set.size),
        "velocity_dofs": velocity_space.dim(),
        "pressure_dofs": pressure_space.dim(), "solve_time": elapsed,
        "Q_in_total": abs(q_in_signed), "Q_out_total": abs(q_out_signed),
        "mass_imbalance": relative_mass_imbalance(q_in_signed, q_out_signed),
        "divergence_L2": float(assemble(div(velocity) ** 2 * dx) ** 0.5),
        "weak_residual": weak, "u_max": float(speed.max()),
        "u_mean": float(speed.mean()), "p_min": float(pressure_values.min()),
        "p_max": float(pressure_values.max()),
        "delta_p": float(np.ptp(pressure_values)), "stages": stages,
    }
    (case_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    done.write_text(json.dumps({"status": "DONE", "metrics": "metrics.json"}, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", default="P64_A,P64_B,P64_C")
    arguments = parser.parse_args()
    reports = []
    for name in arguments.cases.split(","):
        reports.append(solve_case(arguments.root / "cases" / name.strip()))
    with (arguments.root / "metrics.csv").open("w") as stream:
        keys = ("case", "mesh_cells", "velocity_dofs", "pressure_dofs",
                "solve_time", "Q_in_total", "Q_out_total", "mass_imbalance",
                "divergence_L2", "weak_residual", "u_max", "u_mean",
                "p_min", "p_max", "delta_p")
        stream.write(",".join(keys) + "\n")
        for report in reports:
            stream.write(",".join(str(report[key]) for key in keys) + "\n")


if __name__ == "__main__":
    main()
