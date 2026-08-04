# DAFoam API notes (verified against the pinned v5.0.0 image)

Anything marked *(pending)* will be confirmed by `tests/test_environment.py`
and `python/probe_dafoam.py` inside the container; do not rely on unverified
names.

## State / residual plumbing (solver level)

`PYDAFOAM` exposes:

```python
states = DASolver.getStates()
DASolver.setStates(states)
residuals = DASolver.getResiduals()
```

`setStates` updates the OpenFOAM fields from the state vector; `getResiduals`
allocates the local adjoint-state-sized residual vector and fills it from the
solver. After `setStates`, intermediate variables must be refreshed with:

```python
DASolver.solverAD.calcPrimalResidualStatistics("calc")
```

## Reverse-mode Jacobian-transpose-vector product

DAFoam's own MPhys wrapper computes (∂R/∂W)ᵀ v via:

```python
DASolver.solverAD.calcJacTVecProduct(
    state_name,      # e.g. f"{discipline}_states"
    "stateVar",
    state_array,
    residual_name,   # e.g. f"{discipline}_residuals"
    "residual",
    seed,
    product,
)
```

Naming convention used here (from the MPhys layer):

```python
self.discipline = self.solver.getOption("discipline")
self.state_name = f"{self.discipline}_states"
self.residual_name = f"{self.discipline}_residuals"
```

## State vector layout (IMPORTANT)

For DASimpleFoam the registered state vector is NOT just (U, p). It contains:

* cell pressure (p);
* cell velocity (U);
* face flux (phi);
* any model states resolved at runtime.

The bridge treats the state as an opaque vector W ∈ R^N_state and documents
the actual layout in `state_layout_report.json` (Gate B) before any
coordinate-based parameterization is attempted.

## Materialization of the loss

L(W) = ½ R(W)ᵀ R(W),  ∇L = (∂R/∂W)ᵀ R  (JTV with seed = R, optionally
D²-weighted later).  This is a residual JTV, NOT the design adjoint solve.
