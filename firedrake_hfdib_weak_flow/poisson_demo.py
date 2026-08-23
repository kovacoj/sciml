"""Minimal Firedrake test-function neural Poisson demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def run(output_dir: Path, resolution: int = 20) -> dict:
    from firedrake import (
        DirichletBC, Function, FunctionSpace, SpatialCoordinate, TestFunction,
        TrialFunction, UnitSquareMesh, assemble, dot, dx, grad,
    )

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    mesh = UnitSquareMesh(resolution, resolution)
    space = FunctionSpace(mesh, "CG", 1)
    trial, test = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(mesh)
    forcing = 2.0 * np.pi**2 * torch.tensor(1.0).item()
    exact_expression = __import__("ufl").sin(np.pi * x[0]) * __import__("ufl").sin(np.pi * x[1])
    rhs_expression = forcing * exact_expression
    bc = DirichletBC(space, 0.0, "on_boundary")
    matrix = assemble(dot(grad(trial), grad(test)) * dx, bcs=bc).petscmat
    rhs = assemble(rhs_expression * test * dx)
    with rhs.dat.vec_ro as vector:
        b = vector.array.copy()
    dense = matrix.convert("dense").getDenseArray().copy()
    coordinates = np.column_stack([
        Function(space).interpolate(x[index]).dat.data_ro.copy() for index in range(2)
    ])
    boundary = np.asarray(bc.nodes, dtype=np.int64)
    b[boundary] = 0.0
    mask = np.ones(space.dim(), dtype=np.float64)
    mask[boundary] = 0.0
    A = torch.from_numpy(dense)
    b_t = torch.from_numpy(b)
    mask_t = torch.from_numpy(mask)
    xy = torch.from_numpy(coordinates)
    gram = A.clone()

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-3)
    history = []

    def objective():
        values = mask_t * model(2.0 * xy - 1.0).squeeze(-1)
        residual = A @ values - b_t
        dual = torch.linalg.solve(gram, residual)
        return residual @ dual, values

    for step in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            history.append((step, float(loss.detach())))

    parameters = tuple(model.parameters())
    shapes = tuple(parameter.shape for parameter in parameters)
    sizes = tuple(parameter.numel() for parameter in parameters)

    def flatten():
        return np.concatenate([parameter.detach().numpy().ravel() for parameter in parameters])

    def assign(values):
        offset = 0
        with torch.no_grad():
            for parameter, shape, size in zip(parameters, shapes, sizes):
                parameter.copy_(torch.from_numpy(values[offset:offset + size].reshape(shape)))
                offset += size

    def scipy_objective(values):
        assign(values)
        model.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        gradient = np.concatenate([parameter.grad.numpy().ravel() for parameter in parameters])
        return float(loss.detach()), gradient

    from scipy.optimize import minimize
    result = minimize(
        scipy_objective, flatten(), method="L-BFGS-B", jac=True,
        options={"maxiter": 3000, "ftol": 1.0e-15, "gtol": 1.0e-10, "maxls": 50},
    )
    assign(result.x)
    final_loss, neural = objective()
    exact = Function(space).interpolate(exact_expression).dat.data_ro.copy()
    neural_np = neural.detach().numpy()
    relative_l2 = float(np.linalg.norm(neural_np - exact) / np.linalg.norm(exact))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "resolution": resolution,
        "weak_dual_loss": float(final_loss.detach()),
        "coefficient_relative_l2": relative_l2,
        "adam_steps": 500,
        "lbfgs_iterations": int(result.nit),
        "lbfgs_success": bool(result.success),
        "uses_solution_labels_for_training": False,
        "network": {"input_dim": 2, "width": 32, "depth": 2, "output_dim": 1},
    }
    (output_dir / "poisson_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        git_sha = subprocess.check_output(
            ["git", "-c", f"safe.directory={Path(__file__).resolve().parent.parent}",
             "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    torch.save({
        "model_state": model.state_dict(),
        "network": report["network"],
        "resolution": resolution,
        "git_sha": git_sha,
        "weak_dual_loss": report["weak_dual_loss"],
        "coefficient_relative_l2": report["coefficient_relative_l2"],
    }, output_dir / "poisson_model.pt")
    np.savetxt(output_dir / "poisson_training.csv", np.asarray(history), delimiter=",", header="step,loss", comments="")

    figure, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for axis, values, title in zip(
        axes, (exact, neural_np, np.abs(neural_np - exact)),
        ("Exact / direct FE", "Neural weak solution", "Absolute error"),
    ):
        image = axis.tricontourf(coordinates[:, 0], coordinates[:, 1], values, levels=30)
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output_dir / "poisson_result.png", dpi=190)
    plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/poisson_demo"))
    parser.add_argument("--resolution", type=int, default=20)
    arguments = parser.parse_args()
    run(arguments.output_dir, arguments.resolution)


if __name__ == "__main__":
    main()
