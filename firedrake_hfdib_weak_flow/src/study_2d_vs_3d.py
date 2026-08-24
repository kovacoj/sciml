"""Resumable 2D-to-3D confinement sweep for fitted channel topologies."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .brinkman_mesh import INLET_MARKER, OUTLET_MARKER, WALL_MARKER
from .fitted_channel_mesh import write_fitted_topology
from .train import relative_mass_imbalance

PORT_WIDTH = 0.012


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def run_case(topology: str, eta: float, output: Path, nominal_h: float, target_dz: float) -> dict:
    from firedrake import (
        CheckpointFile, Constant, DirichletBC, ExtrudedMesh, FacetNormal,
        Function, FunctionSpace, Mesh, NonlinearVariationalProblem,
        NonlinearVariationalSolver, TestFunctions, VectorFunctionSpace,
        assemble, div, dot, ds_v, dx, split,
    )
    from .weak_forms import navier_stokes_weak_form

    height = eta * PORT_WIDTH
    layers = max(2, int(round(height / target_dz)))
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "topology": topology, "eta": eta, "height": height,
        "port_width": PORT_WIDTH, "nominal_h_xy": nominal_h,
        "layers": layers, "layer_height": height / layers,
        "nu": 0.01, "inlet_velocity": [0.1, 0.0, 0.0],
        "inlet_profile": "constant vector on inherited inlet facets",
        "walls": "no-slip inherited side walls plus top and bottom",
    }
    atomic_json(output / "config.json", config)
    base = Mesh(str(write_fitted_topology(output / "base_mesh.msh", topology, nominal_h)))
    mesh = ExtrudedMesh(base, layers=layers, layer_height=height / layers)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    Z = V * Q
    state = Function(Z, name=f"topology_{topology}_eta_{eta:g}")
    u, p = split(state)
    v, q = TestFunctions(Z)
    bcs = [
        DirichletBC(Z.sub(0), Constant((0.1, 0.0, 0.0)), INLET_MARKER),
        DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), WALL_MARKER),
        DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), "top"),
        DirichletBC(Z.sub(0), Constant((0.0, 0.0, 0.0)), "bottom"),
    ]
    stages = []
    for beta, label in ((0.0, "stokes"), (0.25, "navier_stokes_0p25"),
                        (0.5, "navier_stokes_0p5"), (1.0, "navier_stokes")):
        residual = navier_stokes_weak_form(u, p, v, q, Constant(0.01), Constant(beta), dx)
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(residual, state, bcs=bcs),
            solver_parameters={
                "snes_type": "newtonls", "snes_rtol": 1e-9, "snes_atol": 1e-10,
                "snes_max_it": 30, "ksp_type": "preonly", "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps", "mat_type": "aij",
            },
        )
        started = time.perf_counter()
        try:
            solver.solve()
        except Exception as error:
            stages.append({
                "name": label, "beta": beta, "seconds": time.perf_counter() - started,
                "converged_reason": int(solver.snes.getConvergedReason()),
                "nonlinear_iterations": int(solver.snes.getIterationNumber()),
                "linear_iterations": int(solver.snes.getLinearSolveIterations()),
                "error": str(error),
            })
            atomic_json(output / "stages.json", stages)
            if beta == 0.0:
                raise
            break
        stages.append({
            "name": label, "beta": beta, "seconds": time.perf_counter() - started,
            "converged_reason": int(solver.snes.getConvergedReason()),
            "nonlinear_iterations": int(solver.snes.getIterationNumber()),
            "linear_iterations": int(solver.snes.getLinearSolveIterations()),
        })
        atomic_json(output / "stages.json", stages)
    final_beta = stages[-1]["beta"]
    velocity, pressure = state.subfunctions
    final_residual = navier_stokes_weak_form(u, p, v, q, Constant(0.01), Constant(final_beta), dx)
    assembled = assemble(final_residual, bcs=bcs)
    with assembled.dat.vec_ro as vector:
        weak_residual = float(vector.norm())
    normal = FacetNormal(mesh)
    q_in = float(assemble(dot(velocity, normal) * ds_v(INLET_MARKER)))
    q_out = float(assemble(dot(velocity, normal) * ds_v(OUTLET_MARKER)))
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    Q1 = FunctionSpace(mesh, "CG", 1)
    velocity_p1 = Function(V1).interpolate(velocity)
    pressure_p1 = Function(Q1).interpolate(pressure)
    coordinates = mesh.coordinates.dat.data_ro.copy()
    velocity_values = velocity_p1.dat.data_ro.copy()
    pressure_values = pressure_p1.dat.data_ro.copy()
    np.savez_compressed(
        output / "sliced_fields.npz", coordinates=coordinates,
        velocity=velocity_values, pressure=pressure_values,
    )
    with CheckpointFile(str(output / "solution.h5"), "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(velocity, name="velocity")
        checkpoint.save_function(pressure, name="pressure")
    delta_p = float(np.ptp(pressure_values))
    flow = abs(q_out)
    metrics = {
        **config, "status": "CONVERGED" if final_beta == 1.0 else "STOKES_OR_PARTIAL_NS",
        "final_beta": final_beta, "cells": int(mesh.cell_set.size),
        "velocity_dofs": V.dim(), "pressure_dofs": Q.dim(),
        "total_dofs": V.dim() + Q.dim(), "stages": stages,
        "stokes_time": stages[0]["seconds"],
        "navier_stokes_time": sum(stage["seconds"] for stage in stages[1:]),
        "total_time": sum(stage["seconds"] for stage in stages),
        "nonlinear_iterations": sum(stage["nonlinear_iterations"] for stage in stages),
        "linear_iterations": sum(stage["linear_iterations"] for stage in stages),
        "Q_in": q_in, "Q_out": q_out,
        "mass_imbalance": relative_mass_imbalance(q_in, q_out),
        "weak_residual": weak_residual,
        "divergence_L2": float(assemble(div(velocity) ** 2 * dx) ** 0.5),
        "delta_p": delta_p, "Q": flow,
        "resistance_3d": delta_p / (flow + 1e-30),
        "resistance_per_unit_depth": height * delta_p / (flow + 1e-30),
        "u_max": float(np.linalg.norm(velocity_values, axis=1).max()),
        "p_min": float(pressure_values.min()), "p_max": float(pressure_values.max()),
        "uz_energy_fraction": float(np.linalg.norm(velocity_values[:, 2]) /
                                    (np.linalg.norm(velocity_values) + 1e-30)),
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "mesh_metadata.json", {
        "cells": metrics["cells"], "velocity_dofs": V.dim(),
        "pressure_dofs": Q.dim(), "layers": layers,
        "nominal_h_xy": nominal_h, "layer_height": height / layers,
    })
    atomic_json(output / "DONE.json", {
        "status": "complete", "topology": topology, "eta": eta,
        "finite": bool(np.all(np.isfinite(velocity_values)) and np.all(np.isfinite(pressure_values))),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topologies", default="A,B,C,D,E,F")
    parser.add_argument("--etas", default="0.5,1,2,4")
    parser.add_argument("--nominal-h", type=float, default=0.003)
    parser.add_argument("--target-dz", type=float, default=0.002)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for topology in [value.strip() for value in args.topologies.split(",")]:
        for eta in [float(value) for value in args.etas.split(",")]:
            eta_name = str(eta).replace(".", "p")
            case = args.output / topology / f"eta_{eta_name}"
            if args.resume and (case / "DONE.json").is_file():
                continue
            run_case(topology, eta, case, args.nominal_h, args.target_dz)


if __name__ == "__main__":
    main()
