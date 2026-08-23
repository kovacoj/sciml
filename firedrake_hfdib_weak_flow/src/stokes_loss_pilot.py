"""Sparse coefficient-space pilot for Stokes residual preconditioning losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import LinearOperator, eigs

def _csr(matrix) -> csr_matrix:
    indptr, indices, values = matrix.petscmat.getValuesCSR()
    return csr_matrix((values, indices, indptr), shape=matrix.petscmat.getSize())


def build_system(nx: int, ny: int):
    from firedrake import TrialFunction, assemble, derivative
    from .forms import FiredrakeContext
    from .geometry import EmptyChannelGeometry

    geometry = EmptyChannelGeometry()
    context = FiredrakeContext(
        geometry, nx, ny, formulation="h1_weak", nu=0.01, uin=0.1, pout=0.0,
    )
    context.assign_boundary_lift()
    context.beta.assign(0.0)
    rx0, ry0, rc0 = context.residual_vectors()
    free = np.flatnonzero(context.velocity_mask)
    all_q = np.arange(context.Q.dim())
    coefficients = (context.ux, context.uy, context.p)
    spaces = (context.S, context.S, context.Q)
    forms = (context.rx_form, context.ry_form, context.rc_form)
    row_indices = (free, free, all_q)
    column_indices = (free, free, all_q)
    blocks = []
    for form, rows in zip(forms, row_indices):
        block_row = []
        for coefficient, space, columns in zip(coefficients, spaces, column_indices):
            matrix = _csr(assemble(derivative(form, coefficient, TrialFunction(space))))
            block_row.append(matrix[rows][:, columns])
        blocks.append(block_row)
    operator = bmat(blocks, format="csr")
    residual0 = np.concatenate((
        rx0.dat.data_ro[free], ry0.dat.data_ro[free], rc0.dat.data_ro,
    ))
    h1 = _csr(context.h1_matrix)[free][:, free]
    mass_q = _csr(context.q_mass_matrix)
    norm = bmat([
        [h1, None, None], [None, h1, None], [None, None, mass_q],
    ], format="csr")
    metadata = {
        "nx": nx, "ny": ny, "unknowns": int(operator.shape[0]),
        "velocity_free_dofs_per_component": int(len(free)),
        "pressure_dofs": int(context.Q.dim()),
    }
    return operator, residual0, norm, metadata, len(free), context.Q.dim()


class Loss:
    def __init__(
        self, name: str, operator, residual0, norm,
        n_velocity: int | None = None, exact: np.ndarray | None = None,
        richardson_steps: int | None = None, richardson_omega: float = 0.1,
    ):
        self.name = name
        self.A = operator
        self.r0 = residual0
        self.X = norm
        self.exact = exact
        self.n_velocity = n_velocity
        self.richardson_steps = richardson_steps
        self.richardson_omega = float(richardson_omega)
        self.A_factor_seconds = 0.0
        self.X_factor_seconds = 0.0
        self.block_factor_seconds = 0.0
        self.A_lu = self.X_lu = self.K_lu = self.Mp_lu = None
        if name in {"correction", "oracle"}:
            factor_started = time.monotonic()
            self.A_lu = splu(operator.tocsc())
            self.A_factor_seconds = time.monotonic() - factor_started
        if name == "dual":
            factor_started = time.monotonic()
            self.X_lu = splu(norm.tocsc())
            self.X_factor_seconds = time.monotonic() - factor_started
        if name == "block" or name.startswith("richardson_"):
            if n_velocity is None:
                raise ValueError("block loss requires n_velocity")
            factor_started = time.monotonic()
            split = 2 * n_velocity
            self.K_lu = splu(operator[:split, :split].tocsc())
            self.Mp_lu = splu(norm[split:, split:].tocsc())
            self.block_factor_seconds = time.monotonic() - factor_started
        if name.startswith("richardson_"):
            parsed_steps = int(name.split("_", 1)[1])
            if richardson_steps is not None and parsed_steps != richardson_steps:
                raise ValueError("Richardson name and step count disagree")
            self.richardson_steps = parsed_steps
        if name == "jacobi_ls":
            diagonal = np.asarray(operator.power(2).sum(axis=0)).ravel()
            self.jacobi_inverse = 1.0 / np.maximum(diagonal, 1.0e-30)
        self.solve_seconds = 0.0
        self.solve_calls = 0
        self.initial = self._unnormalized(np.zeros(operator.shape[1]))[0]

    def _block_apply(self, residual):
        split = 2 * self.n_velocity
        started = time.monotonic()
        value = np.concatenate((
            self.K_lu.solve(residual[:split]),
            self.Mp_lu.solve(residual[split:]),
        ))
        self.solve_seconds += time.monotonic() - started
        self.solve_calls += 2
        return value

    def _block_transpose_apply(self, value):
        split = 2 * self.n_velocity
        started = time.monotonic()
        result = np.concatenate((
            self.K_lu.solve(value[:split], trans="T"),
            self.Mp_lu.solve(value[split:], trans="T"),
        ))
        self.solve_seconds += time.monotonic() - started
        self.solve_calls += 2
        return result

    def _richardson_apply(self, residual):
        """Apply a fixed linear k-step Richardson approximation to A^-1."""
        correction = np.zeros_like(residual)
        for _ in range(self.richardson_steps):
            defect = residual - self.A @ correction
            correction += self.richardson_omega * self._block_apply(defect)
        return correction

    def _richardson_transpose_apply(self, value):
        """Apply the exact transpose of the fixed Richardson map."""
        correction = np.zeros_like(value)
        for _ in range(self.richardson_steps):
            defect = value - self.A.T @ correction
            correction += self.richardson_omega * self._block_transpose_apply(defect)
        return correction

    def _unnormalized(self, values):
        residual = self.A @ values + self.r0
        if self.name == "raw":
            value = float(residual @ residual)
            gradient = 2.0 * (self.A.T @ residual)
        elif self.name == "dual":
            started = time.monotonic()
            representer = self.X_lu.solve(residual)
            self.solve_seconds += time.monotonic() - started
            self.solve_calls += 1
            value = float(residual @ representer)
            gradient = 2.0 * (self.A.T @ representer)
        elif self.name == "correction":
            started = time.monotonic()
            correction = self.A_lu.solve(residual)
            self.solve_seconds += time.monotonic() - started
            self.solve_calls += 1
            value = float(correction @ (self.X @ correction))
            gradient = 2.0 * (self.X @ correction)
        elif self.name == "oracle":
            if self.exact is None:
                raise ValueError("oracle loss requires exact coefficients")
            error = values - self.exact
            value = float(error @ (self.X @ error))
            gradient = 2.0 * (self.X @ error)
        elif self.name == "jacobi_ls":
            correction = self.jacobi_inverse * (self.A.T @ residual)
            value = float(correction @ (self.X @ correction))
            adjoint = self.jacobi_inverse * (self.X @ correction)
            gradient = 2.0 * (self.A.T @ (self.A @ adjoint))
        elif self.name == "block":
            correction = self._block_apply(residual)
            value = float(correction @ (self.X @ correction))
            adjoint = self._block_transpose_apply(self.X @ correction)
            gradient = 2.0 * (self.A.T @ adjoint)
        elif self.name.startswith("richardson_"):
            correction = self._richardson_apply(residual)
            value = float(correction @ (self.X @ correction))
            adjoint = self._richardson_transpose_apply(self.X @ correction)
            gradient = 2.0 * (self.A.T @ adjoint)
        else:
            raise ValueError(self.name)
        return value, np.asarray(gradient)

    def __call__(self, values):
        value, gradient = self._unnormalized(values)
        return value / self.initial, gradient / self.initial


def directional_error(loss: Loss, values: np.ndarray, seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(values.shape)
    direction /= np.linalg.norm(direction)
    value, gradient = loss(values)
    del value
    exact = float(gradient @ direction)
    epsilon = 1.0e-6
    finite = (loss(values + epsilon * direction)[0] - loss(values - epsilon * direction)[0]) / (2 * epsilon)
    return abs(exact - finite) / max(abs(exact), abs(finite), 1.0e-14)


def richardson_stability(loss: Loss) -> dict:
    """Estimate whether I-omega P^-1 A is contractive for block Richardson."""
    size = loss.A.shape[0]
    operator = LinearOperator(
        (size, size),
        matvec=lambda value: value - loss.richardson_omega * loss._block_apply(
            loss.A @ value
        ),
        rmatvec=lambda value: value - loss.richardson_omega * loss.A.T @
        loss._block_transpose_apply(value),
        dtype=np.float64,
    )
    try:
        eigenvalues = eigs(
            operator, k=min(6, size - 2), which="LM", return_eigenvectors=False,
            maxiter=500,
        )
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        return {
            "estimated_spectral_radius": spectral_radius,
            "contractive": spectral_radius < 1.0,
            "eigenvalues": [[float(value.real), float(value.imag)] for value in eigenvalues],
        }
    except Exception as error:
        return {
            "estimated_spectral_radius": None,
            "contractive": False,
            "error": repr(error),
        }


def run(
    nx: int, ny: int, output: Path, maxiter: int = 300,
    methods: tuple[str, ...] = (
        "raw", "dual", "jacobi_ls", "block", "correction", "oracle",
    ),
    richardson_omega: float = 0.1,
) -> dict:
    operator, residual0, norm, metadata, n_velocity, n_pressure = build_system(nx, ny)
    exact = splu(operator.tocsc()).solve(-residual0)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in methods:
        loss = Loss(
            name, operator, residual0, norm,
            n_velocity=n_velocity, exact=exact,
            richardson_omega=richardson_omega,
        )
        x0 = np.zeros(operator.shape[1])
        gradient_error = directional_error(loss, x0)
        stability = richardson_stability(loss) if name.startswith("richardson_") else None
        if stability is not None and not stability["contractive"]:
            results.append({
                "loss": name,
                "status": "OPERATOR_UNSTABLE",
                "richardson_omega": richardson_omega,
                "directional_gradient_error": gradient_error,
                "stability": stability,
            })
            continue
        history = []
        callback_count = 0
        started = time.monotonic()

        def callback(intermediate):
            nonlocal callback_count
            callback_count += 1
            if callback_count == 1 or callback_count % 10 == 0:
                history.append({"iteration": callback_count, "loss": float(loss(intermediate)[0])})

        result = minimize(
            loss, x0, method="L-BFGS-B", jac=True, callback=callback,
            options={"maxiter": maxiter, "maxfun": maxiter * 2,
                     "ftol": 1.0e-15, "gtol": 1.0e-10, "maxls": 50},
        )
        relative_error = float(np.linalg.norm(result.x - exact) / np.linalg.norm(exact))
        velocity_slice = slice(0, 2 * n_velocity)
        pressure_slice = slice(2 * n_velocity, 2 * n_velocity + n_pressure)
        velocity_error = float(
            np.linalg.norm(result.x[velocity_slice] - exact[velocity_slice])
            / np.linalg.norm(exact[velocity_slice])
        )
        computed_pressure = result.x[pressure_slice] - result.x[pressure_slice].mean()
        exact_pressure = exact[pressure_slice] - exact[pressure_slice].mean()
        pressure_error = float(
            np.linalg.norm(computed_pressure - exact_pressure)
            / (np.linalg.norm(exact_pressure) + 1.0e-30)
        )
        final_loss = float(loss(result.x)[0])
        results.append({
            "loss": name, "normalized_initial": 1.0,
            "normalized_final": final_loss,
            "relative_coefficient_error": relative_error,
            "relative_velocity_error": velocity_error,
            "relative_pressure_gauge_error": pressure_error,
            "iterations": int(result.nit), "evaluations": int(result.nfev),
            "elapsed_seconds": time.monotonic() - started,
            "gradient_norm": float(np.linalg.norm(result.jac)),
            "directional_gradient_error": gradient_error,
            "success": bool(result.success), "message": str(result.message),
            "history_samples": history,
            "richardson_omega": (
                richardson_omega if name.startswith("richardson_") else None
            ),
            "stability": stability,
            "operator_factor_seconds": loss.A_factor_seconds,
            "norm_factor_seconds": loss.X_factor_seconds,
            "block_factor_seconds": loss.block_factor_seconds,
            "preconditioner_solve_calls": loss.solve_calls,
            "preconditioner_solve_seconds": loss.solve_seconds,
        })
    report = {"problem": metadata, "maxiter": maxiter, "results": results}
    (output / f"pilot_{nx}x{ny}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs/stokes_loss_pilot"))
    parser.add_argument(
        "--methods", default="raw,dual,jacobi_ls,block,correction,oracle",
    )
    parser.add_argument("--richardson-omega", type=float, default=0.1)
    arguments = parser.parse_args()
    run(
        arguments.nx, arguments.ny, arguments.output, arguments.maxiter,
        tuple(arguments.methods.split(",")),
        arguments.richardson_omega,
    )


if __name__ == "__main__":
    main()
