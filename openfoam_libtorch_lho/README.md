# OpenFOAM + LibTorch Neural Finite-Volume Harmonic Oscillator

## Project Overview

This project implements a neural finite-volume eigensolver that couples OpenFOAM's mesh and finite-volume discretization with LibTorch's automatic differentiation and neural networks. The goal is to solve the quantum harmonic oscillator eigenvalue problem using a Rayleigh-Ritz optimization approach without reference data.

## Mathematical Problem

The dimensionless 1D harmonic oscillator:

```
-1/2 d²ψ/dx² + 1/2 x² ψ = E ψ    on (-L, L),   L = 8
ψ(-L) = ψ(L) = 0
```

Analytical eigenvalues: `E_n = n + 1/2` for n = 0, 1, 2, ...

## Discretization

### Finite-Volume Hamiltonian

The domain is discretized using OpenFOAM's finite-volume method. For each cell c:
- Cell volume: V_c
- Cell center coordinate: x_c
- Potential: V(x_c) = 1/2 x_c²

For internal faces f connecting owner o and neighbour n:
- Face area: A_f = |S_f|
- Normal: n_f = S_f / |S_f|
- Distance: d_f = |(C_n - C_o) · n_f|
- Conductance: g_f = kineticScale × A_f / d_f, where kineticScale = 1/2

The discrete Hamiltonian K is assembled as:
- K[o,o] += g_f, K[n,n] += g_f
- K[o,n] -= g_f, K[n,o] -= g_f

Dirichlet boundary faces contribute only to the diagonal of the adjacent cell.

The mass matrix M is diagonal with entries M[c,c] = V_c.

### Generalized Eigenproblem

The discrete problem is: **K ψ = E M ψ**

This is solved directly using torch.linalg.eigh after transforming to standard form via M^(-1/2).

## Neural Network Approach

### Coordinate MLP

A neural network ψ_θ(ξ) takes normalized coordinates ξ = x/L ∈ [-1, 1] and outputs eigenfunction values. Architecture:

```
Linear(1, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, 1)
```

### Parity Symmetrization (Optional)

For state n:
- Even n: ψ_raw(x) = [f(x) + f(-x)] / 2
- Odd n: ψ_raw(x) = [f(x) - f(-x)] / 2

### Training Objective

Minimize the finite-volume Rayleigh quotient:

```
E[ψ_θ] = (ψ_θᵀ K ψ_θ) / (ψ_θᵀ M ψ_θ)
```

Higher states are obtained by M-orthogonal projection against previously computed states.

### Optimization

1. **Adam**: 5000 steps at lr=1e-3
2. **LBFGS**: 300 iterations, history size 100

Three deterministic restarts are performed; the best is selected.

## Division of Responsibility

| Component | Responsibility |
|-----------|----------------|
| **OpenFOAM** | Mesh generation, geometry queries, finite-volume operator assembly |
| **LibTorch** | Neural networks, automatic differentiation, optimizers (Adam, LBFGS), tensor operations |

**Important**: We do NOT differentiate through OpenFOAM. The Rayleigh quotient is evaluated as a tensor expression; autograd computes ∇_θ E automatically.

## Why No Data?

This is a label-free training approach. The neural network learns by minimizing the physical functional (Rayleigh quotient), not by fitting reference eigenfunctions. Direct FV solutions and analytical eigenfunctions are used ONLY for post-training validation.

## Installation

### Prerequisites

- Ubuntu 22.04, 24.04, or 26.04
- x86_64 architecture
- sudo access

### Setup Commands

```bash
cd openfoam_libtorch_lho

# Install system packages and OpenFOAM v14
./scripts/bootstrap_ubuntu.sh

# Install LibTorch 2.0.1+cpu
./scripts/install_libtorch.sh

# Source environment
source env.sh

# Build applications
./Allwmake

# Create Python virtual environment (for tests/plots)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas matplotlib pytest

# Generate case
python scripts/generate_case.py --cells 256 --half-width 8 --output cases/lho_256

# Create mesh
blockMesh -case cases/lho_256
checkMesh -case cases/lho_256

# Gate 0: Smoke test
torchFoamSmoke -case cases/lho_256

# Direct FV eigensolver
fvNeuralLHO -case cases/lho_256 -mode direct

# Coefficient optimization (Gate 1)
fvNeuralLHO -case cases/lho_256 -mode coefficients

# Neural training (Gates 2-3)
fvNeuralLHO -case cases/lho_256 -mode neural

# Run tests
pytest -q tests

# Generate plots and summary
python scripts/summarize_results.py --case cases/lho_256 --output outputs/summary.csv
python scripts/plot_results.py --case cases/lho_256 --output-dir outputs
```

## Repository Structure

```
openfoam_libtorch_lho/
├── README.md
├── env.sh                 # Environment setup script
├── Allwmake               # Build script
├── Allclean               # Clean script
├── .deps/                 # LibTorch installation
├── applications/
│   ├── torchFoamSmoke/    # Gate 0 smoke test
│   └── fvNeuralLHO/        # Main eigensolver
├── cases/
│   └── lho_template/      # Case template
├── scripts/
│   ├── bootstrap_ubuntu.sh
│   ├── install_libtorch.sh
│   ├── capture_environment.sh
│   ├── generate_case.py
│   └── ...
├── tests/
│   ├── run_tests.sh
│   ├── test_matrix.py
│   ├── test_spectrum.py
│   └── test_outputs.py
└── outputs/
```

## Acceptance Gates

### Gate 0 — Integration
- [ ] OpenFOAM + LibTorch smoke test passes
- [ ] Autograd produces correct gradients [2, 4, 6]

### Gate 1 — Finite-Volume Operator
- [ ] K symmetric to < 1e-12
- [ ] Direct C++ spectrum matches SciPy spectrum
- [ ] Boundary treatment verified

### Gate 2 — Trainable Coefficients
- [ ] Ground-state energy agrees with direct FV to < 1e-8

### Gate 3 — Neural Ground State
- [ ] Neural energy agrees with direct FV to < 1e-4 (target < 1e-6)
- [ ] Eigenfunction overlap > 0.999

### Gate 4 — First Six States
- [ ] All states finite and normalized
- [ ] Orthogonality near identity
- [ ] Correct parity
- [ ] No state collapse

### Gate 5 — Convergence
- [ ] Direct FV discretization error decreases under refinement
- [ ] Neural and discretization errors reported separately

### Gate 6 — Output
- [ ] Fields open in ParaView
- [ ] CSV and plots reproducible

## Current Limitations

1. Dense matrix storage (acceptable up to N ~ 1024)
2. Serial CPU execution only
3. Uniform orthogonal 1-D mesh
4. Stationary linear operator
5. No differentiation through OpenFOAM

## Planned Progression

1. Sparse tensor operator
2. Extracted fvScalarMatrix backend
3. Extension to Poisson equation
4. Advection-diffusion
5. Stokes/Navier-Stokes
6. DAFoam adjoint coupling

## Error Decomposition

Three distinct error sources are tracked separately:

1. **Discretization error**: |E_directFV - E_exact|
2. **Neural optimization error**: |E_NN - E_directFV|
3. **Total error**: |E_NN - E_exact|

## License

This is a research prototype for educational purposes.
