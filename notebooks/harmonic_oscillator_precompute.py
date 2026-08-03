#!/usr/bin/env python3
"""
Three-way neural comparison for the first ten eigenpairs of the 1D quantum
harmonic oscillator

    H psi = E psi,
    H = -1/2 d^2/dx^2 + 1/2 x^2,
    psi(-L) = psi(L) = 0.

The file contains three logically separate solvers and one comparison runner:

1. strong
   A conventional strong-form PINN. The loss contains the pointwise residual

       -1/2 psi'' + 1/2 x^2 psi - E psi.

2. weak
   An explicit finite-element weak residual. Firedrake constructs a CG1 trial
   and test space and assembles the Hamiltonian, mass, and test-space Gram
   matrices by quadrature. A PyTorch coordinate MLP produces FE coefficients.
   The residual component

       r_i = a(psi_h, phi_i) - E m(psi_h, phi_i)

   is the equation tested against the i-th Firedrake basis test function. The
   loss uses the discrete dual norm r^T G^{-1} r.

3. variational
   A neural Rayleigh-Ritz method. It minimizes

       a(psi, psi) / m(psi, psi)

   and does not explicitly construct an independent test-space residual.

All methods use the same ideas for mode discovery:

* hard homogeneous Dirichlet boundary conditions;
* hard orthogonal projection against all previously learned states;
* L2 normalization;
* identical parity information by default (even n -> even state, odd n -> odd
  state), which can be disabled with --no-parity;
* sequential computation of the first N eigenpairs.

Outputs
-------
<output-dir>/comparison.csv
<output-dir>/summary.json
<output-dir>/eigenvalue_errors.png
<output-dir>/training_times.png
<output-dir>/eigenfunctions_<method>.png
<output-dir>/gram_<method>.png

Recommended fair CPU run
------------------------
OMP_NUM_THREADS=1 MPLBACKEND=Agg python harmonic_oscillator_three_way.py \
    --method all --states 10 --device cpu --torch-threads 1

A full ten-state run is intentionally substantial. First validate the pipeline:

python harmonic_oscillator_three_way.py --method all --states 3 --quick

The Firedrake method is serial in this compact benchmark because the assembled
PETSc matrices are copied to dense PyTorch tensors. This is suitable for a 1D
comparison, not a scalable 3D implementation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# Generic configuration and data structures
# -----------------------------------------------------------------------------

EPS = 1.0e-14


@dataclass
class Config:
    method: str
    states: int
    domain_half_width: float
    grid_points: int
    fe_cells: int
    network_width: int
    network_depth: int
    pretrain_steps: int
    adam_steps: int
    lbfgs_steps: int
    learning_rate: float
    residual_energy_weight: float
    seed: int
    parity: bool
    device: str
    torch_threads: int
    output_dir: str
    log_every: int
    weak_trial_degree: int
    weak_test_degree: int
    weak_test_norm: str
    weak_residual_l2_weight: float
    weak_residual_weight: float
    weak_restarts: int
    weak_state_budget_slope: float
    weak_block: bool


@dataclass
class ModeResult:
    method: str
    state: int
    eigenvalue: float
    exact_eigenvalue: float
    eigenvalue_abs_error: float
    eigenvalue_rel_error: float
    l2_error: float
    h1_error: float
    residual_metric: float
    train_seconds: float


@dataclass
class MethodResult:
    name: str
    modes: list[ModeResult]
    setup_seconds: float
    total_train_seconds: float
    total_seconds: float
    gram_matrix: np.ndarray
    plot_x: np.ndarray
    predicted_functions: np.ndarray
    exact_functions: np.ndarray


# -----------------------------------------------------------------------------
# Reproducibility and numerical helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_trapezoid(values: Tensor, x: Tensor) -> Tensor:
    """Integrate values over a one-dimensional grid."""

    return torch.trapz(values.reshape(-1), x.reshape(-1))


def numpy_trapezoid(values: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, x))
    return float(np.trapz(values, x))


def derivative(values: Tensor, x: Tensor, create_graph: bool = True) -> Tensor:
    return torch.autograd.grad(
        outputs=values,
        inputs=x,
        grad_outputs=torch.ones_like(values),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def freeze_module(module: nn.Module) -> nn.Module:
    frozen = copy.deepcopy(module)
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


# -----------------------------------------------------------------------------
# Exact oscillator states
# -----------------------------------------------------------------------------


def hermite_physicists_torch(n: int, x: Tensor) -> Tensor:
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2.0 * x

    h_nm2 = torch.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(1, n):
        h_n = 2.0 * x * h_nm1 - 2.0 * k * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def exact_state_torch(n: int, x: Tensor) -> Tensor:
    coefficient = math.pi ** (-0.25) / math.sqrt(
        (2**n) * math.factorial(n)
    )
    return (
        coefficient
        * hermite_physicists_torch(n, x)
        * torch.exp(-0.5 * x**2)
    )


def exact_state_numpy(n: int, x: np.ndarray) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float64)
    return exact_state_torch(n, x_t).numpy()


def exact_state_derivative_numpy(n: int, x: np.ndarray) -> np.ndarray:
    """Derivative from the harmonic-oscillator ladder identity."""

    result = np.zeros_like(x, dtype=np.float64)
    if n > 0:
        result += math.sqrt(n / 2.0) * exact_state_numpy(n - 1, x)
    result -= math.sqrt((n + 1) / 2.0) * exact_state_numpy(n + 1, x)
    return result


# -----------------------------------------------------------------------------
# Shared neural coordinate representation
# -----------------------------------------------------------------------------


class CoordinateMLP(nn.Module):
    def __init__(
        self,
        half_width: float,
        width: int,
        depth: int,
        parity_sign: int | None,
    ) -> None:
        super().__init__()
        self.half_width = float(half_width)
        self.parity_sign = parity_sign

        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def unconstrained(self, x: Tensor) -> Tensor:
        return self.network(x / self.half_width)

    def field(self, x: Tensor) -> Tensor:
        raw = self.unconstrained(x)
        if self.parity_sign is not None:
            reflected = self.unconstrained(-x)
            raw = 0.5 * (raw + self.parity_sign * reflected)

        boundary_factor = 1.0 - (x / self.half_width) ** 2
        return boundary_factor * raw

    def forward(self, x: Tensor) -> Tensor:
        return self.field(x)


@dataclass
class ContinuousNeuralState:
    """A normalized linear combination of frozen raw coordinate networks."""

    models: tuple[CoordinateMLP, ...]
    combination: Tensor
    eigenvalue: float
    residual_metric: float
    train_seconds: float

    def evaluate(self, x: Tensor) -> Tensor:
        columns = [model.field(x).reshape(-1) for model in self.models]
        basis = torch.stack(columns, dim=1)
        return (basis @ self.combination).reshape(-1, 1)


def parity_for_state(state_index: int, enabled: bool) -> int | None:
    if not enabled:
        return None
    return 1 if state_index % 2 == 0 else -1


def continuous_previous_matrix(
    previous_states: Sequence[ContinuousNeuralState],
    x: Tensor,
) -> Tensor | None:
    if not previous_states:
        return None
    return torch.cat([state.evaluate(x) for state in previous_states], dim=1)


def project_and_normalize_continuous(
    model: CoordinateMLP,
    previous_states: Sequence[ContinuousNeuralState],
    x_quadrature: Tensor,
    evaluation_points: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project raw network output against previous states and L2-normalize.

    Returns
    -------
    u_q:
        Normalized current state on x_quadrature.
    u_eval:
        Normalized state on evaluation_points, or u_q when omitted.
    alpha:
        Projection coefficients against the previous states.
    norm:
        Norm of the projected raw function.
    """

    raw_q = model.field(x_quadrature)
    previous_q = continuous_previous_matrix(previous_states, x_quadrature)

    if previous_q is None:
        alpha = torch.empty(
            0, dtype=x_quadrature.dtype, device=x_quadrature.device
        )
        projected_q = raw_q
    else:
        # The previous states are close to orthonormal, but solve the full Gram
        # system to avoid accumulating numerical orthogonality errors.
        k = previous_q.shape[1]
        gram_entries: list[Tensor] = []
        rhs_entries: list[Tensor] = []
        for i in range(k):
            rhs_entries.append(
                torch_trapezoid(previous_q[:, i : i + 1] * raw_q, x_quadrature)
            )
            row: list[Tensor] = []
            for j in range(k):
                row.append(
                    torch_trapezoid(
                        previous_q[:, i : i + 1]
                        * previous_q[:, j : j + 1],
                        x_quadrature,
                    )
                )
            gram_entries.append(torch.stack(row))

        gram = torch.stack(gram_entries)
        rhs = torch.stack(rhs_entries)
        alpha = torch.linalg.solve(gram, rhs)
        projected_q = raw_q - previous_q @ alpha.reshape(-1, 1)

    norm_squared = torch_trapezoid(projected_q**2, x_quadrature)
    norm = torch.sqrt(norm_squared + EPS)
    u_q = projected_q / norm

    if evaluation_points is None:
        u_eval = u_q
    else:
        raw_eval = model.field(evaluation_points)
        previous_eval = continuous_previous_matrix(previous_states, evaluation_points)
        if previous_eval is None:
            projected_eval = raw_eval
        else:
            projected_eval = raw_eval - previous_eval @ alpha.reshape(-1, 1)
        u_eval = projected_eval / norm

    return u_q, u_eval, alpha, norm


def evaluate_projected_continuous(
    model: CoordinateMLP,
    previous_states: Sequence[ContinuousNeuralState],
    x: Tensor,
    alpha: Tensor,
    norm: Tensor,
) -> Tensor:
    raw = model.field(x)
    previous = continuous_previous_matrix(previous_states, x)
    if previous is not None:
        raw = raw - previous @ alpha.reshape(-1, 1)
    return raw / norm


def finalize_continuous_state(
    model: CoordinateMLP,
    previous_states: Sequence[ContinuousNeuralState],
    alpha: Tensor,
    norm: Tensor,
    eigenvalue: float,
    residual_metric: float,
    train_seconds: float,
) -> ContinuousNeuralState:
    frozen_current = freeze_module(model)

    target_device = norm.device
    if not previous_states:
        combination = torch.tensor(
            [1.0 / float(norm.detach())],
            dtype=torch.float64,
            device=target_device,
        )
        models = (frozen_current,)
    else:
        old_models = previous_states[-1].models
        number_old_models = len(old_models)

        # Column j contains state j in the shared raw-network basis.
        previous_combinations = []
        for state in previous_states:
            padded = torch.zeros(
                number_old_models, dtype=torch.float64, device=target_device
            )
            padded[: state.combination.numel()] = state.combination.detach().to(
                target_device
            )
            previous_combinations.append(padded)
        combination_matrix = torch.stack(previous_combinations, dim=1)

        old_part = -combination_matrix @ alpha.detach().to(target_device)
        unnormalized = torch.cat(
            [
                old_part,
                torch.ones(1, dtype=torch.float64, device=target_device),
            ]
        )
        combination = unnormalized / float(norm.detach())
        models = (*old_models, frozen_current)

    return ContinuousNeuralState(
        models=models,
        combination=combination,
        eigenvalue=eigenvalue,
        residual_metric=residual_metric,
        train_seconds=train_seconds,
    )


def continuous_energy(u: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
    du = derivative(u, x, create_graph=True)
    energy = torch_trapezoid(
        0.5 * du**2 + 0.5 * x**2 * u**2,
        x,
    )
    return energy, du


# -----------------------------------------------------------------------------
# Strong-form PINN
# -----------------------------------------------------------------------------


def train_strong_state(
    state_index: int,
    previous_states: list[ContinuousNeuralState],
    config: Config,
    device: torch.device,
) -> ContinuousNeuralState:
    set_seed(config.seed + state_index)
    model = CoordinateMLP(
        config.domain_half_width,
        config.network_width,
        config.network_depth,
        parity_for_state(state_index, config.parity),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    q_base = torch.linspace(
        -config.domain_half_width,
        config.domain_half_width,
        config.grid_points,
        dtype=torch.float64,
        device=device,
    ).reshape(-1, 1)

    print(f"\n[strong] state n={state_index}")
    start = time.perf_counter()

    # Stage 1: Rayleigh pretraining selects the lowest mode in the hard
    # orthogonal complement. This prevents convergence to an arbitrary
    # zero-residual eigenpair during the strong-residual stage.
    for step in range(config.pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        q = q_base.detach().clone().requires_grad_(True)
        u_q, _, _, _ = project_and_normalize_continuous(
            model, previous_states, q
        )
        energy, _ = continuous_energy(u_q, q)
        energy.backward()
        optimizer.step()

        if config.log_every > 0 and step % config.log_every == 0:
            print(
                f"  pretrain {step:5d} | E={energy.detach().item():.9f}"
            )

    def strong_objective(include_energy_selector: bool) -> dict[str, Tensor]:
        q = q_base.detach().clone().requires_grad_(True)
        u_q, _, alpha, norm = project_and_normalize_continuous(
            model, previous_states, q
        )
        energy, _ = continuous_energy(u_q, q)

        # Use a separate coordinate tensor for the pointwise strong residual,
        # reusing the projection and normalization computed on the quadrature grid.
        x_r = q_base[1:-1].detach().clone().requires_grad_(True)
        u_r = evaluate_projected_continuous(
            model, previous_states, x_r, alpha, norm
        )
        u_x = derivative(u_r, x_r, create_graph=True)
        u_xx = derivative(u_x, x_r, create_graph=True)
        residual = -0.5 * u_xx + 0.5 * x_r**2 * u_r - energy * u_r
        residual_loss = torch_trapezoid(residual**2, x_r)

        loss = residual_loss
        if include_energy_selector:
            loss = loss + config.residual_energy_weight * energy

        return {
            "loss": loss,
            "energy": energy,
            "residual": residual_loss,
            "alpha": alpha,
            "norm": norm,
        }

    for step in range(config.adam_steps):
        optimizer.zero_grad(set_to_none=True)
        values = strong_objective(include_energy_selector=True)
        values["loss"].backward()
        optimizer.step()

        if config.log_every > 0 and step % config.log_every == 0:
            print(
                f"  Adam    {step:5d} | E={values['energy'].detach().item():.9f} "
                f"| ||R||_L2^2={values['residual'].detach().item():.3e}"
            )

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            values = strong_objective(include_energy_selector=False)
            values["loss"].backward()
            return values["loss"]

        lbfgs.step(closure)

    final = strong_objective(include_energy_selector=False)
    elapsed = time.perf_counter() - start
    eigenvalue = final["energy"].detach().item()
    residual_metric = math.sqrt(max(final["residual"].detach().item(), 0.0))

    state = finalize_continuous_state(
        model,
        previous_states,
        final["alpha"],
        final["norm"],
        eigenvalue,
        residual_metric,
        elapsed,
    )

    print(
        f"  final | E={eigenvalue:.10f} exact={state_index + 0.5:.10f} "
        f"|dE|={abs(eigenvalue - state_index - 0.5):.3e} "
        f"||R||_L2={residual_metric:.3e} time={elapsed:.2f}s"
    )
    return state


def run_strong(config: Config) -> tuple[list[ContinuousNeuralState], float, float]:
    device = torch.device(config.device)
    setup_start = time.perf_counter()
    # Imports and static allocations are negligible, but report them explicitly.
    setup_seconds = time.perf_counter() - setup_start

    states: list[ContinuousNeuralState] = []
    train_start = time.perf_counter()
    for n in range(config.states):
        states.append(train_strong_state(n, states, config, device))
    train_seconds = time.perf_counter() - train_start
    return states, setup_seconds, train_seconds


# -----------------------------------------------------------------------------
# Variational neural Rayleigh-Ritz solver
# -----------------------------------------------------------------------------


def train_variational_state(
    state_index: int,
    previous_states: list[ContinuousNeuralState],
    config: Config,
    device: torch.device,
) -> ContinuousNeuralState:
    set_seed(config.seed + state_index)
    model = CoordinateMLP(
        config.domain_half_width,
        config.network_width,
        config.network_depth,
        parity_for_state(state_index, config.parity),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    q_base = torch.linspace(
        -config.domain_half_width,
        config.domain_half_width,
        config.grid_points,
        dtype=torch.float64,
        device=device,
    ).reshape(-1, 1)

    print(f"\n[variational] state n={state_index}")
    start = time.perf_counter()

    def variational_objective() -> dict[str, Tensor]:
        q = q_base.detach().clone().requires_grad_(True)
        u_q, _, alpha, norm = project_and_normalize_continuous(
            model, previous_states, q
        )
        energy, _ = continuous_energy(u_q, q)
        return {
            "loss": energy,
            "energy": energy,
            "alpha": alpha,
            "norm": norm,
        }

    total_adam_steps = config.pretrain_steps + config.adam_steps
    for step in range(total_adam_steps):
        optimizer.zero_grad(set_to_none=True)
        values = variational_objective()
        values["loss"].backward()
        optimizer.step()

        if config.log_every > 0 and step % config.log_every == 0:
            print(
                f"  Adam    {step:5d} | E={values['energy'].detach().item():.9f}"
            )

    if config.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=config.lbfgs_steps,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            values = variational_objective()
            values["loss"].backward()
            return values["loss"]

        lbfgs.step(closure)

    final = variational_objective()
    elapsed = time.perf_counter() - start
    eigenvalue = final["energy"].detach().item()

    # Report a strong residual only as a diagnostic; it is not the training loss.
    x_r = q_base[1:-1].detach().clone().requires_grad_(True)
    u_r = evaluate_projected_continuous(
        model, previous_states, x_r, final["alpha"], final["norm"]
    )
    u_x = derivative(u_r, x_r, create_graph=True)
    u_xx = derivative(u_x, x_r, create_graph=True)
    residual = -0.5 * u_xx + 0.5 * x_r**2 * u_r - final["energy"] * u_r
    residual_metric = math.sqrt(
        max(torch_trapezoid(residual**2, x_r).detach().item(), 0.0)
    )

    state = finalize_continuous_state(
        model,
        previous_states,
        final["alpha"],
        final["norm"],
        eigenvalue,
        residual_metric,
        elapsed,
    )

    print(
        f"  final | E={eigenvalue:.10f} exact={state_index + 0.5:.10f} "
        f"|dE|={abs(eigenvalue - state_index - 0.5):.3e} "
        f"diagnostic ||R||_L2={residual_metric:.3e} time={elapsed:.2f}s"
    )
    return state


def run_variational(
    config: Config,
) -> tuple[list[ContinuousNeuralState], float, float]:
    device = torch.device(config.device)
    setup_start = time.perf_counter()
    setup_seconds = time.perf_counter() - setup_start

    states: list[ContinuousNeuralState] = []
    train_start = time.perf_counter()
    for n in range(config.states):
        states.append(train_variational_state(n, states, config, device))
    train_seconds = time.perf_counter() - train_start
    return states, setup_seconds, train_seconds


# -----------------------------------------------------------------------------
# Firedrake explicit weak-test-space solver
# -----------------------------------------------------------------------------


@dataclass
class WeakFiredrakeContext:
    x_interior: Tensor
    x_all: np.ndarray
    interior_dofs: np.ndarray
    trial_mass: Tensor
    trial_hamiltonian: Tensor
    mixed_mass: Tensor
    mixed_hamiltonian: Tensor
    test_mass: Tensor
    test_stiffness: Tensor
    gram: Tensor
    gram_cholesky: Tensor
    mass_cholesky: Tensor
    trial_space: object
    setup_seconds: float


@dataclass
class WeakState:
    coefficients: Tensor
    eigenvalue: float
    residual_metric: float
    train_seconds: float
    objective: float = 0.0
    restart: int = 0


class WeakCoordinateMLP(nn.Module):
    """Coordinate MLP whose values are interior CG1 nodal coefficients."""

    def __init__(
        self,
        half_width: float,
        width: int,
        depth: int,
        parity_sign: int | None,
    ) -> None:
        super().__init__()
        self.half_width = float(half_width)
        self.parity_sign = parity_sign
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def unconstrained(self, x: Tensor) -> Tensor:
        return self.network(x / self.half_width)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.unconstrained(x)
        if self.parity_sign is not None:
            raw = 0.5 * (
                raw + self.parity_sign * self.unconstrained(-x)
            )
        return raw.reshape(-1)


def petsc_to_scipy_csr(matrix):
    import scipy.sparse as sp

    indptr, indices, values = matrix.getValuesCSR()
    return sp.csr_matrix(
        (
            np.asarray(values).copy(),
            np.asarray(indices).copy(),
            np.asarray(indptr).copy(),
        ),
        shape=matrix.getSize(),
    )


def build_weak_context(config: Config) -> WeakFiredrakeContext:
    setup_start = time.perf_counter()

    try:
        from firedrake import (
            DirichletBC,
            Function,
            FunctionSpace,
            IntervalMesh,
            SpatialCoordinate,
            TestFunction,
            TrialFunction,
            assemble,
            dot,
            dx,
            grad,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The weak method requires a Firedrake Python environment. "
            "Run this script with the Firedrake interpreter."
        ) from exc

    mesh = IntervalMesh(
        config.fe_cells,
        -config.domain_half_width,
        config.domain_half_width,
    )
    if mesh.comm.size != 1:
        raise RuntimeError(
            "This comparison script copies PETSc matrices into dense PyTorch "
            "tensors and therefore supports serial execution only."
        )

    U = FunctionSpace(mesh, "CG", config.weak_trial_degree)
    V = FunctionSpace(mesh, "CG", config.weak_test_degree)
    trial = TrialFunction(U)
    test = TestFunction(V)
    trial_square = TrialFunction(U)
    trial_square_test = TestFunction(U)
    test_square = TrialFunction(V)
    test_square_test = TestFunction(V)
    x = SpatialCoordinate(mesh)[0]
    quadrature_degree = max(
        8, 2 * (config.weak_trial_degree + config.weak_test_degree) + 2
    )
    dx_q = dx(metadata={"quadrature_degree": quadrature_degree})

    # Firedrake handles the quadrature and supplies the complete FE test basis.
    mass_form = trial * test * dx_q
    stiffness_form = dot(grad(trial), grad(test)) * dx_q
    potential_form = x**2 * trial * test * dx_q

    mass_fd = assemble(mass_form, mat_type="aij").petscmat
    stiffness_fd = assemble(stiffness_form, mat_type="aij").petscmat
    potential_fd = assemble(potential_form, mat_type="aij").petscmat

    mass_full = petsc_to_scipy_csr(mass_fd)
    stiffness_full = petsc_to_scipy_csr(stiffness_fd)
    potential_full = petsc_to_scipy_csr(potential_fd)

    trial_mass_full = petsc_to_scipy_csr(
        assemble(trial_square * trial_square_test * dx_q, mat_type="aij").petscmat
    )
    trial_stiffness_full = petsc_to_scipy_csr(
        assemble(
            dot(grad(trial_square), grad(trial_square_test)) * dx_q,
            mat_type="aij",
        ).petscmat
    )
    trial_potential_full = petsc_to_scipy_csr(
        assemble(
            x**2 * trial_square * trial_square_test * dx_q, mat_type="aij"
        ).petscmat
    )
    test_mass_full = petsc_to_scipy_csr(
        assemble(test_square * test_square_test * dx_q, mat_type="aij").petscmat
    )
    test_stiffness_full = petsc_to_scipy_csr(
        assemble(
            dot(grad(test_square), grad(test_square_test)) * dx_q,
            mat_type="aij",
        ).petscmat
    )

    trial_boundary = np.asarray(
        DirichletBC(U, 0.0, "on_boundary").nodes, dtype=np.int64
    )
    test_boundary = np.asarray(
        DirichletBC(V, 0.0, "on_boundary").nodes, dtype=np.int64
    )
    interior_dofs = np.setdiff1d(np.arange(U.dim()), trial_boundary)
    test_interior = np.setdiff1d(np.arange(V.dim()), test_boundary)

    def tensor(matrix) -> Tensor:
        return torch.tensor(matrix.toarray(), dtype=torch.float64, device="cpu")

    trial_mass = tensor(trial_mass_full[interior_dofs][:, interior_dofs])
    trial_stiffness = tensor(
        trial_stiffness_full[interior_dofs][:, interior_dofs]
    )
    trial_potential = tensor(
        trial_potential_full[interior_dofs][:, interior_dofs]
    )
    trial_hamiltonian = 0.5 * trial_stiffness + 0.5 * trial_potential
    mixed_mass = tensor(mass_full[test_interior][:, interior_dofs])
    mixed_stiffness = tensor(stiffness_full[test_interior][:, interior_dofs])
    mixed_potential = tensor(potential_full[test_interior][:, interior_dofs])
    mixed_hamiltonian = 0.5 * mixed_stiffness + 0.5 * mixed_potential
    test_mass = tensor(test_mass_full[test_interior][:, test_interior])
    test_stiffness = tensor(
        test_stiffness_full[test_interior][:, test_interior]
    )

    # H1 test-space Riesz map. The explicit residual loss is
    #     r^T G^{-1} r,
    # not the basis-dependent Euclidean norm r^T r.
    h = 2.0 * config.domain_half_width / config.fe_cells
    if config.weak_test_norm == "mass":
        gram = test_mass
    elif config.weak_test_norm == "scaled-h1":
        gram = test_mass + h**2 * test_stiffness
    else:
        gram = test_mass + test_stiffness
    gram_cholesky = torch.linalg.cholesky(gram)
    mass_cholesky = torch.linalg.cholesky(test_mass)

    x_function = Function(U).interpolate(x)
    x_all = np.asarray(x_function.dat.data_ro).copy()
    x_interior = torch.tensor(
        x_all[interior_dofs], dtype=torch.float64, device="cpu"
    ).reshape(-1, 1)

    return WeakFiredrakeContext(
        x_interior=x_interior,
        x_all=x_all,
        interior_dofs=interior_dofs,
        trial_mass=trial_mass,
        trial_hamiltonian=trial_hamiltonian,
        mixed_mass=mixed_mass,
        mixed_hamiltonian=mixed_hamiltonian,
        test_mass=test_mass,
        test_stiffness=test_stiffness,
        gram=gram,
        gram_cholesky=gram_cholesky,
        mass_cholesky=mass_cholesky,
        trial_space=U,
        setup_seconds=time.perf_counter() - setup_start,
    )


def weak_mass_inner(context: WeakFiredrakeContext, a: Tensor, b: Tensor) -> Tensor:
    return torch.dot(a, context.trial_mass @ b)


def weak_project_normalize(
    raw: Tensor,
    previous_states: Sequence[WeakState],
    context: WeakFiredrakeContext,
) -> Tensor:
    if previous_states:
        psi = torch.stack([state.coefficients for state in previous_states], dim=1)
        gram_previous = psi.T @ context.trial_mass @ psi
        rhs = psi.T @ (context.trial_mass @ raw)
        alpha = torch.linalg.solve(gram_previous, rhs)
        raw = raw - psi @ alpha

    norm_squared = weak_mass_inner(context, raw, raw)
    return raw / torch.sqrt(norm_squared + EPS)


def weak_energy_only(
    model: WeakCoordinateMLP,
    previous_states: Sequence[WeakState],
    context: WeakFiredrakeContext,
) -> Tensor:
    raw = model(context.x_interior)
    coefficients = weak_project_normalize(raw, previous_states, context)
    return torch.dot(coefficients, context.trial_hamiltonian @ coefficients)


def weak_quantities(
    model: WeakCoordinateMLP,
    previous_states: Sequence[WeakState],
    context: WeakFiredrakeContext,
    residual_weight: float,
    residual_l2_weight: float,
) -> dict[str, Tensor]:
    raw = model(context.x_interior)
    coefficients = weak_project_normalize(raw, previous_states, context)

    hc = context.trial_hamiltonian @ coefficients
    energy = torch.dot(coefficients, hc)

    # Row i is the weak equation tested with the i-th CG1 basis function.
    residual = (
        context.mixed_hamiltonian @ coefficients
        - energy * (context.mixed_mass @ coefficients)
    )
    riesz = torch.cholesky_solve(
        residual.reshape(-1, 1), context.gram_cholesky
    ).reshape(-1)
    residual_dual_squared = torch.dot(residual, riesz)
    riesz_l2 = torch.cholesky_solve(
        residual.reshape(-1, 1), context.mass_cholesky
    ).reshape(-1)
    residual_l2_squared = torch.dot(residual, riesz_l2)
    combined_residual = (
        residual_dual_squared + residual_l2_weight * residual_l2_squared
    )
    scaled_residual = combined_residual / (energy.square() + 1.0)
    loss = energy + residual_weight * scaled_residual

    return {
        "loss": loss,
        "coefficients": coefficients,
        "energy": energy,
        "residual": combined_residual,
        "residual_hminus1": residual_dual_squared,
        "residual_l2": residual_l2_squared,
    }


def train_weak_state(
    state_index: int,
    previous_states: list[WeakState],
    config: Config,
    context: WeakFiredrakeContext,
    restart: int = 0,
) -> WeakState:
    set_seed(config.seed + state_index + 1000 * restart)
    model = WeakCoordinateMLP(
        config.domain_half_width,
        config.network_width,
        config.network_depth,
        parity_for_state(state_index, config.parity),
    ).to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"\n[weak/Firedrake] state n={state_index} restart={restart}")
    start = time.perf_counter()
    budget_scale = 1.0 + config.weak_state_budget_slope * state_index
    pretrain_steps = round(config.pretrain_steps * budget_scale)
    adam_steps = round(config.adam_steps * budget_scale)
    lbfgs_steps = round(config.lbfgs_steps * budget_scale)

    # Stage 1: variational selection in the constrained FE coefficient space.
    for step in range(pretrain_steps):
        optimizer.zero_grad(set_to_none=True)
        energy = weak_energy_only(model, previous_states, context)
        energy.backward()
        optimizer.step()

        if config.log_every > 0 and step % config.log_every == 0:
            print(
                f"  pretrain {step:5d} | E={energy.detach().item():.9f}"
            )

    for step in range(adam_steps):
        optimizer.zero_grad(set_to_none=True)
        values = weak_quantities(
            model,
            previous_states,
            context,
            residual_weight=config.weak_residual_weight,
            residual_l2_weight=config.weak_residual_l2_weight,
        )
        values["loss"].backward()
        optimizer.step()

        if config.log_every > 0 and step % config.log_every == 0:
            print(
                f"  Adam    {step:5d} | E={values['energy'].detach().item():.9f} "
                f"| ||R||_*^2={values['residual'].detach().item():.3e}"
            )

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.8,
            max_iter=lbfgs_steps,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad(set_to_none=True)
            values = weak_quantities(
                model,
                previous_states,
                context,
                residual_weight=config.weak_residual_weight,
                residual_l2_weight=config.weak_residual_l2_weight,
            )
            values["loss"].backward()
            return values["loss"]

        lbfgs.step(closure)

    final = weak_quantities(
        model,
        previous_states,
        context,
        residual_weight=config.weak_residual_weight,
        residual_l2_weight=config.weak_residual_l2_weight,
    )
    elapsed = time.perf_counter() - start
    eigenvalue = final["energy"].detach().item()
    residual_metric = math.sqrt(max(final["residual"].detach().item(), 0.0))

    state = WeakState(
        coefficients=final["coefficients"].detach(),
        eigenvalue=eigenvalue,
        residual_metric=residual_metric,
        train_seconds=elapsed,
        objective=final["loss"].detach().item(),
        restart=restart,
    )

    print(
        f"  final | E={eigenvalue:.10f} exact={state_index + 0.5:.10f} "
        f"|dE|={abs(eigenvalue - state_index - 0.5):.3e} "
        f"||R||_*={residual_metric:.3e} time={elapsed:.2f}s"
    )
    return state


def run_weak(
    config: Config,
) -> tuple[list[WeakState], WeakFiredrakeContext, float]:
    context = build_weak_context(config)
    states: list[WeakState] = []
    train_start = time.perf_counter()
    for n in range(config.states):
        candidates = [
            train_weak_state(n, states, config, context, restart)
            for restart in range(config.weak_restarts)
        ]
        best = min(candidates, key=lambda state: state.objective)
        print(
            f"  selected restart={best.restart} objective={best.objective:.6e}"
        )
        states.append(best)
    train_seconds = time.perf_counter() - train_start
    return states, context, train_seconds


class WeakBlockMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, config.network_width), nn.Tanh()]
        for _ in range(config.network_depth - 1):
            layers.extend([nn.Linear(config.network_width, config.network_width), nn.Tanh()])
        layers.append(nn.Linear(config.network_width, config.states))
        self.network = nn.Sequential(*layers)
        self.half_width = config.domain_half_width
        self.parity = config.parity

    def forward(self, x: Tensor) -> Tensor:
        raw = self.network(x / self.half_width)
        if self.parity:
            reflected = self.network(-x / self.half_width)
            signs = torch.tensor(
                [1.0 if n % 2 == 0 else -1.0 for n in range(raw.shape[1])],
                dtype=raw.dtype,
                device=raw.device,
            )
            raw = 0.5 * (raw + reflected * signs)
        return raw


def weak_block_quantities(
    model: WeakBlockMLP,
    context: WeakFiredrakeContext,
    config: Config,
) -> dict[str, Tensor]:
    raw = model(context.x_interior)
    overlap = raw.T @ context.trial_mass @ raw
    overlap = overlap + EPS * torch.eye(
        overlap.shape[0], dtype=overlap.dtype, device=overlap.device
    )
    chol = torch.linalg.cholesky(overlap)
    coefficients = torch.linalg.solve_triangular(chol, raw.T, upper=False).T

    rayleigh = coefficients.T @ context.trial_hamiltonian @ coefficients
    eigenvalues, rotation = torch.linalg.eigh(rayleigh)
    coefficients = coefficients @ rotation

    residual = (
        context.mixed_hamiltonian @ coefficients
        - (context.mixed_mass @ coefficients) * eigenvalues.reshape(1, -1)
    )
    riesz_h1 = torch.cholesky_solve(residual, context.gram_cholesky)
    riesz_l2 = torch.cholesky_solve(residual, context.mass_cholesky)
    residual_h1 = torch.sum(residual * riesz_h1, dim=0)
    residual_l2 = torch.sum(residual * riesz_l2, dim=0)
    combined = residual_h1 + config.weak_residual_l2_weight * residual_l2
    scaled = combined / (eigenvalues.square() + 1.0)
    loss = eigenvalues.sum() + config.weak_residual_weight * scaled.sum()
    return {
        "loss": loss,
        "coefficients": coefficients,
        "eigenvalues": eigenvalues,
        "residuals": combined,
    }


def run_weak_block(
    config: Config,
) -> tuple[list[WeakState], WeakFiredrakeContext, float]:
    context = build_weak_context(config)
    best_values: dict[str, Tensor] | None = None
    best_loss = math.inf
    best_restart = 0
    total_start = time.perf_counter()

    for restart in range(config.weak_restarts):
        set_seed(config.seed + 1000 * restart)
        model = WeakBlockMLP(config).to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        steps = config.pretrain_steps + config.adam_steps
        print(f"\n[weak-block/Firedrake] restart={restart}")
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            values = weak_block_quantities(model, context, config)
            values["loss"].backward()
            optimizer.step()
            if config.log_every > 0 and step % config.log_every == 0:
                energies = ", ".join(
                    f"{value:.5f}" for value in values["eigenvalues"].detach().tolist()
                )
                print(f"  Adam {step:5d} | loss={values['loss'].item():.4e} | E=[{energies}]")

        if config.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=0.8,
                max_iter=config.lbfgs_steps,
                tolerance_grad=1.0e-11,
                tolerance_change=1.0e-13,
                line_search_fn="strong_wolfe",
            )

            def closure() -> Tensor:
                lbfgs.zero_grad(set_to_none=True)
                current = weak_block_quantities(model, context, config)
                current["loss"].backward()
                return current["loss"]

            lbfgs.step(closure)

        final = weak_block_quantities(model, context, config)
        final_loss = final["loss"].detach().item()
        if final_loss < best_loss:
            best_loss = final_loss
            best_restart = restart
            best_values = {key: value.detach() for key, value in final.items()}

    assert best_values is not None
    elapsed = time.perf_counter() - total_start
    print(f"  selected block restart={best_restart} objective={best_loss:.6e}")
    states = []
    for n in range(config.states):
        states.append(
            WeakState(
                coefficients=best_values["coefficients"][:, n],
                eigenvalue=best_values["eigenvalues"][n].item(),
                residual_metric=math.sqrt(
                    max(best_values["residuals"][n].item(), 0.0)
                ),
                train_seconds=elapsed / config.states,
                objective=best_loss,
                restart=best_restart,
            )
        )
    return states, context, elapsed


def run_direct_fe(
    config: Config,
) -> tuple[list[WeakState], WeakFiredrakeContext, float]:
    import scipy.linalg

    context = build_weak_context(config)
    start = time.perf_counter()
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        context.trial_hamiltonian.numpy(),
        context.trial_mass.numpy(),
        subset_by_index=(0, config.states - 1),
    )
    elapsed = time.perf_counter() - start
    states = []
    for n in range(config.states):
        coefficients = torch.tensor(eigenvectors[:, n], dtype=torch.float64)
        residual = (
            context.mixed_hamiltonian @ coefficients
            - eigenvalues[n] * (context.mixed_mass @ coefficients)
        )
        riesz = torch.cholesky_solve(
            residual.reshape(-1, 1), context.gram_cholesky
        ).reshape(-1)
        states.append(
            WeakState(
                coefficients=coefficients,
                eigenvalue=float(eigenvalues[n]),
                residual_metric=math.sqrt(max(torch.dot(residual, riesz).item(), 0.0)),
                train_seconds=elapsed / config.states,
            )
        )
    return states, context, elapsed


# -----------------------------------------------------------------------------
# Common accuracy assessment
# -----------------------------------------------------------------------------


def assess_continuous_method(
    name: str,
    states: Sequence[ContinuousNeuralState],
    setup_seconds: float,
    train_seconds: float,
    config: Config,
) -> MethodResult:
    device = torch.device(config.device)
    x_np = np.linspace(
        -config.domain_half_width,
        config.domain_half_width,
        max(4001, 8 * config.grid_points + 1),
        dtype=np.float64,
    )
    x = torch.tensor(x_np, dtype=torch.float64, device=device).reshape(-1, 1)

    predicted_rows: list[np.ndarray] = []
    exact_rows: list[np.ndarray] = []
    mode_results: list[ModeResult] = []

    values_for_gram: list[np.ndarray] = []
    total_start = time.perf_counter()

    for n, state in enumerate(states):
        x_grad = x.detach().clone().requires_grad_(True)
        predicted = state.evaluate(x_grad)
        pred_norm = torch.sqrt(torch_trapezoid(predicted**2, x_grad) + EPS)
        predicted = predicted / pred_norm

        exact = exact_state_torch(n, x_grad)
        exact_norm = torch.sqrt(torch_trapezoid(exact**2, x_grad) + EPS)
        exact = exact / exact_norm

        overlap = torch_trapezoid(predicted * exact, x_grad)
        sign = 1.0 if overlap.detach().item() >= 0.0 else -1.0
        predicted = sign * predicted

        dp = derivative(predicted, x_grad, create_graph=True)
        de = derivative(exact, x_grad, create_graph=False)
        l2_error = math.sqrt(
            max(torch_trapezoid((predicted - exact) ** 2, x_grad).detach().item(), 0.0)
        )
        h1_error = math.sqrt(
            max(
                torch_trapezoid(
                    (predicted - exact) ** 2 + (dp - de) ** 2,
                    x_grad,
                ).detach().item(),
                0.0,
            )
        )

        predicted_np = predicted.detach().cpu().numpy().reshape(-1)
        exact_np = exact.detach().cpu().numpy().reshape(-1)
        predicted_rows.append(predicted_np)
        exact_rows.append(exact_np)
        values_for_gram.append(predicted_np)

        exact_e = n + 0.5
        abs_error = abs(state.eigenvalue - exact_e)
        mode_results.append(
            ModeResult(
                method=name,
                state=n,
                eigenvalue=state.eigenvalue,
                exact_eigenvalue=exact_e,
                eigenvalue_abs_error=abs_error,
                eigenvalue_rel_error=abs_error / exact_e,
                l2_error=l2_error,
                h1_error=h1_error,
                residual_metric=state.residual_metric,
                train_seconds=state.train_seconds,
            )
        )

    values = np.asarray(values_for_gram)
    gram = np.empty((len(states), len(states)), dtype=np.float64)
    for i in range(len(states)):
        for j in range(len(states)):
            gram[i, j] = numpy_trapezoid(values[i] * values[j], x_np)

    assessment_seconds = time.perf_counter() - total_start
    return MethodResult(
        name=name,
        modes=mode_results,
        setup_seconds=setup_seconds,
        total_train_seconds=train_seconds,
        total_seconds=setup_seconds + train_seconds + assessment_seconds,
        gram_matrix=gram,
        plot_x=x_np,
        predicted_functions=np.asarray(predicted_rows),
        exact_functions=np.asarray(exact_rows),
    )


def evaluate_piecewise_linear(
    x_nodes: np.ndarray,
    values_nodes: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x_nodes)
    x_sorted = x_nodes[order]
    values_sorted = values_nodes[order]

    values = np.interp(x_eval, x_sorted, values_sorted)
    intervals = np.searchsorted(x_sorted, x_eval, side="right") - 1
    intervals = np.clip(intervals, 0, len(x_sorted) - 2)
    slopes = np.diff(values_sorted) / np.diff(x_sorted)
    derivatives = slopes[intervals]
    return values, derivatives


def assess_weak_method(
    states: Sequence[WeakState],
    context: WeakFiredrakeContext,
    train_seconds: float,
    config: Config,
    name: str = "weak",
) -> MethodResult:
    x_eval = np.linspace(
        -config.domain_half_width,
        config.domain_half_width,
        max(4001, 8 * config.fe_cells + 1),
        dtype=np.float64,
    )

    predicted_rows: list[np.ndarray] = []
    exact_rows: list[np.ndarray] = []
    mode_results: list[ModeResult] = []
    values_for_gram: list[np.ndarray] = []

    assessment_start = time.perf_counter()
    for n, state in enumerate(states):
        full = np.zeros_like(context.x_all, dtype=np.float64)
        full[context.interior_dofs] = state.coefficients.cpu().numpy()
        from firedrake import Function

        fe_function = Function(context.trial_space)
        fe_function.dat.data[:] = full
        predicted = np.asarray(
            fe_function.at(x_eval.reshape(-1, 1), tolerance=1.0e-10),
            dtype=np.float64,
        ).reshape(-1)
        predicted_derivative = np.gradient(predicted, x_eval, edge_order=2)

        exact = exact_state_numpy(n, x_eval)
        exact_derivative = exact_state_derivative_numpy(n, x_eval)

        pred_norm = math.sqrt(max(numpy_trapezoid(predicted**2, x_eval), EPS))
        exact_norm = math.sqrt(max(numpy_trapezoid(exact**2, x_eval), EPS))
        predicted /= pred_norm
        predicted_derivative /= pred_norm
        exact /= exact_norm
        exact_derivative /= exact_norm

        overlap = numpy_trapezoid(predicted * exact, x_eval)
        if overlap < 0.0:
            predicted *= -1.0
            predicted_derivative *= -1.0

        l2_error = math.sqrt(
            max(numpy_trapezoid((predicted - exact) ** 2, x_eval), 0.0)
        )
        h1_error = math.sqrt(
            max(
                numpy_trapezoid(
                    (predicted - exact) ** 2
                    + (predicted_derivative - exact_derivative) ** 2,
                    x_eval,
                ),
                0.0,
            )
        )

        predicted_rows.append(predicted.copy())
        exact_rows.append(exact.copy())
        values_for_gram.append(predicted.copy())

        exact_e = n + 0.5
        abs_error = abs(state.eigenvalue - exact_e)
        mode_results.append(
            ModeResult(
                method=name,
                state=n,
                eigenvalue=state.eigenvalue,
                exact_eigenvalue=exact_e,
                eigenvalue_abs_error=abs_error,
                eigenvalue_rel_error=abs_error / exact_e,
                l2_error=l2_error,
                h1_error=h1_error,
                residual_metric=state.residual_metric,
                train_seconds=state.train_seconds,
            )
        )

    values = np.asarray(values_for_gram)
    gram = np.empty((len(states), len(states)), dtype=np.float64)
    for i in range(len(states)):
        for j in range(len(states)):
            gram[i, j] = numpy_trapezoid(values[i] * values[j], x_eval)

    assessment_seconds = time.perf_counter() - assessment_start
    return MethodResult(
        name=name,
        modes=mode_results,
        setup_seconds=context.setup_seconds,
        total_train_seconds=train_seconds,
        total_seconds=context.setup_seconds + train_seconds + assessment_seconds,
        gram_matrix=gram,
        plot_x=x_eval,
        predicted_functions=np.asarray(predicted_rows),
        exact_functions=np.asarray(exact_rows),
    )


# -----------------------------------------------------------------------------
# Reports and plots
# -----------------------------------------------------------------------------


def write_csv(results: Sequence[MethodResult], output_dir: Path) -> None:
    path = output_dir / "comparison.csv"
    fieldnames = list(asdict(results[0].modes[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for method in results:
            for mode in method.modes:
                writer.writerow(asdict(mode))


def method_summary(result: MethodResult) -> dict[str, float | str]:
    eigen_errors = np.array([mode.eigenvalue_abs_error for mode in result.modes])
    l2_errors = np.array([mode.l2_error for mode in result.modes])
    h1_errors = np.array([mode.h1_error for mode in result.modes])
    off_diagonal = result.gram_matrix - np.eye(result.gram_matrix.shape[0])
    return {
        "method": result.name,
        "setup_seconds": result.setup_seconds,
        "training_seconds": result.total_train_seconds,
        "total_seconds_including_assessment": result.total_seconds,
        "mean_seconds_per_state": result.total_train_seconds / len(result.modes),
        "mean_abs_eigenvalue_error": float(np.mean(eigen_errors)),
        "max_abs_eigenvalue_error": float(np.max(eigen_errors)),
        "mean_l2_error": float(np.mean(l2_errors)),
        "max_l2_error": float(np.max(l2_errors)),
        "mean_h1_error": float(np.mean(h1_errors)),
        "max_h1_error": float(np.max(h1_errors)),
        "max_gram_error": float(np.max(np.abs(off_diagonal))),
    }


def write_summary(
    results: Sequence[MethodResult], config: Config, output_dir: Path
) -> None:
    data = {
        "config": asdict(config),
        "methods": [method_summary(result) for result in results],
        "metric_notes": {
            "strong_residual": "L2 norm of pointwise strong residual",
            "variational_residual": "diagnostic L2 norm of pointwise strong residual",
            "weak_residual": "Firedrake FE residual dual norm sqrt(r^T G^{-1} r)",
            "timing": "training excludes common plotting/report generation; weak setup includes Firedrake assembly",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)


def write_web_data(
    results: Sequence[MethodResult], config: Config, output_dir: Path
) -> None:
    labels = {
        "strong": ("Strong PINN", "#c2412d"),
        "weak": ("Weak Firedrake + PyTorch", "#167d9a"),
        "variational": ("Variational Rayleigh-Ritz", "#7c5ca6"),
        "direct-fe": ("Direct finite element", "#3f7d4c"),
    }
    methods = {}
    for result in results:
        if result.name not in labels:
            continue
        label, color = labels[result.name]
        methods[result.name] = {
            "label": label,
            "color": color,
            "states": [
                {
                    "n": mode.state,
                    "eigenvalue": mode.eigenvalue,
                    "eigenvalueError": mode.eigenvalue_abs_error,
                    "l2Error": mode.l2_error,
                    "trainSeconds": mode.train_seconds,
                    "values": np.round(result.predicted_functions[index], 9).tolist(),
                }
                for index, mode in enumerate(result.modes)
            ],
        }
    first = next(result for result in results if result.name in labels)
    payload = {
        "schemaVersion": 1,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "domain": [-config.domain_half_width, config.domain_half_width],
        "coordinates": np.round(first.plot_x, 9).tolist(),
        "exactEigenvalues": [n + 0.5 for n in range(config.states)],
        "methods": methods,
        "provenance": {
            "command": " ".join(sys.argv),
            "source": "notebooks/harmonic_oscillator_precompute.py",
            "notes": [
                "Curves are mass-normalized and sign-aligned to the analytic state.",
                "The browser performs no solver optimization.",
            ],
        },
    }
    with (output_dir / "harmonic-oscillator-comparison.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, separators=(",", ":"))


def plot_eigenvalue_errors(results: Sequence[MethodResult], output_dir: Path) -> None:
    plt.figure(figsize=(8.5, 5.5))
    for result in results:
        x = [mode.state for mode in result.modes]
        y = [max(mode.eigenvalue_abs_error, 1.0e-16) for mode in result.modes]
        plt.semilogy(x, y, marker="o", label=result.name)
    plt.xlabel("state index n")
    plt.ylabel(r"$|E_n^{\mathrm{pred}}-E_n|$")
    plt.title("Harmonic-oscillator eigenvalue errors")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "eigenvalue_errors.png", dpi=200)
    plt.close()


def plot_training_times(results: Sequence[MethodResult], output_dir: Path) -> None:
    plt.figure(figsize=(8.5, 5.5))
    for result in results:
        x = [mode.state for mode in result.modes]
        y = [mode.train_seconds for mode in result.modes]
        plt.plot(x, y, marker="o", label=result.name)
    plt.xlabel("state index n")
    plt.ylabel("training time [s]")
    plt.title("Per-state training time")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_times.png", dpi=200)
    plt.close()


def plot_functions(result: MethodResult, output_dir: Path) -> None:
    plt.figure(figsize=(10, 12))
    for n, (predicted, exact) in enumerate(
        zip(result.predicted_functions, result.exact_functions)
    ):
        shift = 1.45 * n
        plt.plot(
            result.plot_x,
            exact + shift,
            linewidth=1.8,
            label=f"exact n={n}",
        )
        plt.plot(
            result.plot_x,
            predicted + shift,
            "--",
            linewidth=1.3,
            label=f"{result.name} n={n}",
        )
    plt.xlabel("x")
    plt.ylabel("eigenfunctions, vertically shifted")
    plt.title(f"First {len(result.modes)} states: {result.name}")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"eigenfunctions_{result.name}.png", dpi=200)
    plt.close()


def plot_gram(result: MethodResult, output_dir: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    image = plt.imshow(result.gram_matrix, vmin=-1.0, vmax=1.0)
    plt.colorbar(image, label=r"$\langle \psi_i,\psi_j\rangle$")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.title(f"Orthonormality Gram matrix: {result.name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"gram_{result.name}.png", dpi=200)
    plt.close()


def print_summary_table(results: Sequence[MethodResult]) -> None:
    print("\n" + "=" * 110)
    print("COMPARISON SUMMARY")
    print("=" * 110)
    header = (
        f"{'method':<14} {'train[s]':>12} {'s/state':>12} "
        f"{'mean |dE|':>14} {'max |dE|':>14} "
        f"{'mean L2':>14} {'mean H1':>14} {'Gram err':>14}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        summary = method_summary(result)
        print(
            f"{result.name:<14} "
            f"{summary['training_seconds']:12.3f} "
            f"{summary['mean_seconds_per_state']:12.3f} "
            f"{summary['mean_abs_eigenvalue_error']:14.3e} "
            f"{summary['max_abs_eigenvalue_error']:14.3e} "
            f"{summary['mean_l2_error']:14.3e} "
            f"{summary['mean_h1_error']:14.3e} "
            f"{summary['max_gram_error']:14.3e}"
        )

    print("\nPer-state results")
    print("-" * 110)
    for result in results:
        print(f"\n[{result.name}]")
        for mode in result.modes:
            print(
                f"n={mode.state:2d} E={mode.eigenvalue:12.8f} "
                f"|dE|={mode.eigenvalue_abs_error:9.2e} "
                f"L2={mode.l2_error:9.2e} H1={mode.h1_error:9.2e} "
                f"res={mode.residual_metric:9.2e} "
                f"time={mode.train_seconds:8.2f}s"
            )


# -----------------------------------------------------------------------------
# CLI and main runner
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare strong, explicit weak, and variational neural eigensolvers."
    )
    parser.add_argument(
        "--method",
        choices=("all", "website", "strong", "weak", "weak-block", "direct-fe", "variational"),
        default="all",
    )
    parser.add_argument("--states", type=int, default=10)
    parser.add_argument("--L", dest="domain_half_width", type=float, default=8.0)
    parser.add_argument("--grid-points", type=int, default=513)
    parser.add_argument("--fe-cells", type=int, default=512)
    parser.add_argument("--width", dest="network_width", type=int, default=64)
    parser.add_argument("--depth", dest="network_depth", type=int, default=4)
    parser.add_argument("--pretrain-steps", type=int, default=600)
    parser.add_argument("--adam-steps", type=int, default=1800)
    parser.add_argument("--lbfgs-steps", type=int, default=120)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--residual-energy-weight",
        type=float,
        default=5.0e-2,
        help="Small Rayleigh selector during strong/weak Adam refinement.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Strong/variational device. Weak/Firedrake remains CPU-only.",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-dir", default="harmonic_oscillator_comparison")
    parser.add_argument("--log-every", type=int, default=300)
    parser.add_argument("--weak-trial-degree", type=int, default=2)
    parser.add_argument("--weak-test-degree", type=int, default=3)
    parser.add_argument(
        "--weak-test-norm",
        choices=("mass", "scaled-h1", "h1"),
        default="scaled-h1",
    )
    parser.add_argument("--weak-residual-l2-weight", type=float, default=0.05)
    parser.add_argument("--weak-residual-weight", type=float, default=10.0)
    parser.add_argument("--weak-restarts", type=int, default=1)
    parser.add_argument("--weak-state-budget-slope", type=float, default=0.5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test settings; not suitable for accuracy comparison.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    if args.states < 1:
        raise ValueError("--states must be at least 1.")
    if args.grid_points < 5:
        raise ValueError("--grid-points must be at least 5.")
    if args.fe_cells < 4:
        raise ValueError("--fe-cells must be at least 4.")
    if args.weak_trial_degree < 1 or args.weak_test_degree < 1:
        raise ValueError("Weak trial and test degrees must be positive.")
    if args.weak_restarts < 1:
        raise ValueError("--weak-restarts must be at least 1.")

    if args.quick:
        args.pretrain_steps = min(args.pretrain_steps, 100)
        args.adam_steps = min(args.adam_steps, 250)
        args.lbfgs_steps = min(args.lbfgs_steps, 20)
        args.grid_points = min(args.grid_points, 257)
        args.fe_cells = min(args.fe_cells, 256)
        args.network_width = min(args.network_width, 40)
        args.network_depth = min(args.network_depth, 3)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    return Config(
        method=args.method,
        states=args.states,
        domain_half_width=args.domain_half_width,
        grid_points=args.grid_points,
        fe_cells=args.fe_cells,
        network_width=args.network_width,
        network_depth=args.network_depth,
        pretrain_steps=args.pretrain_steps,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        learning_rate=args.learning_rate,
        residual_energy_weight=args.residual_energy_weight,
        seed=args.seed,
        parity=not args.no_parity,
        device=args.device,
        torch_threads=args.torch_threads,
        output_dir=args.output_dir,
        log_every=args.log_every,
        weak_trial_degree=args.weak_trial_degree,
        weak_test_degree=args.weak_test_degree,
        weak_test_norm=args.weak_test_norm,
        weak_residual_l2_weight=args.weak_residual_l2_weight,
        weak_residual_weight=args.weak_residual_weight,
        weak_restarts=args.weak_restarts,
        weak_state_budget_slope=args.weak_state_budget_slope,
        weak_block=args.method == "weak-block",
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration")
    print(json.dumps(asdict(config), indent=2))
    print(
        "\nFor an apples-to-apples speed comparison, use --device cpu and "
        "OMP_NUM_THREADS=1. The weak method is CPU-only in this script."
    )

    results: list[MethodResult] = []

    if config.method in ("all", "website", "strong"):
        states, setup_seconds, train_seconds = run_strong(config)
        results.append(
            assess_continuous_method(
                "strong", states, setup_seconds, train_seconds, config
            )
        )

    if config.method in ("all", "website", "weak"):
        weak_states, weak_context, train_seconds = run_weak(config)
        results.append(
            assess_weak_method(weak_states, weak_context, train_seconds, config)
        )

    if config.method in ("all", "weak-block"):
        weak_states, weak_context, train_seconds = run_weak_block(config)
        results.append(
            assess_weak_method(
                weak_states,
                weak_context,
                train_seconds,
                config,
                name="weak-block",
            )
        )

    if config.method in ("all", "website", "direct-fe"):
        weak_states, weak_context, solve_seconds = run_direct_fe(config)
        results.append(
            assess_weak_method(
                weak_states,
                weak_context,
                solve_seconds,
                config,
                name="direct-fe",
            )
        )

    if config.method in ("all", "website", "variational"):
        states, setup_seconds, train_seconds = run_variational(config)
        results.append(
            assess_continuous_method(
                "variational", states, setup_seconds, train_seconds, config
            )
        )

    write_csv(results, output_dir)
    write_summary(results, config, output_dir)
    write_web_data(results, config, output_dir)
    plot_eigenvalue_errors(results, output_dir)
    plot_training_times(results, output_dir)
    for result in results:
        plot_functions(result, output_dir)
        plot_gram(result, output_dir)

    print_summary_table(results)
    print(f"\nResults written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
