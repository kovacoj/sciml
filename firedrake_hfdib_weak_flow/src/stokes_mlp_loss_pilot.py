"""Matched-initialization MLP pilot for Stokes residual metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.sparse.linalg import splu

from .forms import FiredrakeContext
from .geometry import EmptyChannelGeometry
from .model import CoordinateMLP
from .stokes_loss_pilot import Loss, build_system


def run(nx: int, ny: int, seed: int, output: Path, maxiter: int = 300) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    operator, residual0, norm, metadata, n_velocity, n_pressure = build_system(nx, ny)
    exact = splu(operator.tocsc()).solve(-residual0)
    geometry = EmptyChannelGeometry()
    context = FiredrakeContext(geometry, nx, ny, formulation="h1_weak")
    coordinates_s = context._coordinates(context.S)
    coordinates_q = context._coordinates(context.Q)
    features_s = torch.from_numpy(np.column_stack((
        2.0 * coordinates_s[:, 0] / context.Lx - 1.0,
        2.0 * coordinates_s[:, 1] / context.Ly - 1.0,
        np.zeros(context.S.dim()), np.zeros(context.S.dim()),
    )))
    features_q = torch.from_numpy(np.column_stack((
        2.0 * coordinates_q[:, 0] / context.Lx - 1.0,
        2.0 * coordinates_q[:, 1] / context.Ly - 1.0,
        np.zeros(context.Q.dim()), np.zeros(context.Q.dim()),
    )))
    free = np.flatnonzero(context.velocity_mask)
    torch.manual_seed(seed)
    template = CoordinateMLP(input_dim=4, width=64, depth=4)
    initial_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
    results = []
    output.mkdir(parents=True, exist_ok=True)

    for loss_name in ("raw", "dual", "correction"):
        model = CoordinateMLP(input_dim=4, width=64, depth=4)
        model.load_state_dict(initial_state)
        parameters = tuple(model.parameters())
        shapes = tuple(parameter.shape for parameter in parameters)
        sizes = tuple(parameter.numel() for parameter in parameters)
        loss = Loss(loss_name, operator, residual0, norm)

        def fields():
            output_s = model(features_s)
            output_q = model(features_q)
            ux = torch.from_numpy(context.ux_lift) + 0.1 * torch.from_numpy(context.velocity_mask) * output_s[:, 0]
            uy = 0.1 * torch.from_numpy(context.velocity_mask) * output_s[:, 1]
            pressure = 0.2 * output_q[:, 2]
            return ux, uy, pressure

        def flatten_parameters():
            return np.concatenate([parameter.detach().numpy().ravel() for parameter in parameters])

        def assign_parameters(values):
            offset = 0
            with torch.no_grad():
                for parameter, shape, size in zip(parameters, shapes, sizes):
                    parameter.copy_(torch.from_numpy(values[offset:offset + size].reshape(shape)))
                    offset += size

        evaluations = 0
        history = []
        callback_count = 0

        def objective(values):
            nonlocal evaluations
            assign_parameters(values)
            model.zero_grad(set_to_none=True)
            ux, uy, pressure = fields()
            coefficients = torch.cat((ux[free], uy[free], pressure))
            value, coefficient_gradient = loss(coefficients.detach().numpy())
            coefficients.backward(torch.from_numpy(coefficient_gradient))
            gradient = np.concatenate([parameter.grad.numpy().ravel() for parameter in parameters])
            evaluations += 1
            return value, gradient

        def callback(values):
            nonlocal callback_count
            callback_count += 1
            if callback_count == 1 or callback_count % 10 == 0:
                history.append({"iteration": callback_count, "loss": float(objective(values)[0])})

        started = time.monotonic()
        optimized = minimize(
            objective, flatten_parameters(), method="L-BFGS-B", jac=True,
            callback=callback,
            options={"maxiter": maxiter, "maxfun": maxiter * 2,
                     "ftol": 1e-15, "gtol": 1e-10, "maxls": 50, "maxcor": 20},
        )
        assign_parameters(optimized.x)
        ux, uy, pressure = fields()
        coefficients = torch.cat((ux[free], uy[free], pressure)).detach().numpy()
        relative_error = float(np.linalg.norm(coefficients - exact) / np.linalg.norm(exact))
        velocity_slice = slice(0, 2 * n_velocity)
        pressure_slice = slice(2 * n_velocity, 2 * n_velocity + n_pressure)
        velocity_error = float(
            np.linalg.norm(coefficients[velocity_slice] - exact[velocity_slice])
            / np.linalg.norm(exact[velocity_slice])
        )
        computed_pressure = coefficients[pressure_slice] - coefficients[pressure_slice].mean()
        exact_pressure = exact[pressure_slice] - exact[pressure_slice].mean()
        pressure_error = float(
            np.linalg.norm(computed_pressure - exact_pressure)
            / (np.linalg.norm(exact_pressure) + 1e-30)
        )
        torch.save({
            "model_state": model.state_dict(), "seed": seed, "loss": loss_name,
            "nx": nx, "ny": ny,
        }, output / f"model_{loss_name}_{nx}x{ny}_seed{seed}.pt")
        results.append({
            "loss": loss_name, "seed": seed, "nx": nx, "ny": ny,
            "normalized_final": float(loss(coefficients)[0]),
            "relative_coefficient_error": relative_error,
            "relative_velocity_error": velocity_error,
            "relative_pressure_gauge_error": pressure_error,
            "iterations": int(optimized.nit), "evaluations": evaluations,
            "elapsed_seconds": time.monotonic() - started,
            "gradient_norm": float(np.linalg.norm(optimized.jac)),
            "success": bool(optimized.success), "message": str(optimized.message),
            "history_samples": history,
            "preconditioner_solve_calls": loss.solve_calls,
            "preconditioner_solve_seconds": loss.solve_seconds,
        })
    report = {"problem": metadata, "seed": seed, "maxiter": maxiter, "results": results}
    (output / f"mlp_{nx}x{ny}_seed{seed}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/stokes_mlp_loss_pilot"))
    arguments = parser.parse_args()
    run(arguments.nx, arguments.ny, arguments.seed, arguments.output, arguments.maxiter)


if __name__ == "__main__":
    main()
