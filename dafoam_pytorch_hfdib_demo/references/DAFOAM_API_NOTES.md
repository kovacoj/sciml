# DAFoam API notes

Status: **verified against the pinned `dafoam/opt-packages:v5.0.0` image**
(2026-08-04), inside the container.

## Verified environment facts

* Container python is Miniconda at
  `$DAFOAM_ROOT_PATH/packages/miniconda3/bin/python` (3.10.8) — there is no
  bare `python3` until `source $HOME/dafoam/loadDAFoam.sh` runs.
* OpenFOAM is **OpenFOAM-v2506** (openfoam.com) plus a separate
  `OpenFOAM-AD` tree built in ADF and ADR variants.
* Solver executables are named `simpleFoam` / `simpleFoamADR` etc.
  **There are no `DASimpleFoam`/`DASimpleFoamReverseAD` CLI binaries in
  v5.0.0** — PYDAFOAM drives the compiled in-process solver libraries
  (`dafoam/libs/pyDASolvers*.so`, normal/ADF/ADR variants). The Gate-A
  "binary exists" check from older docs is N/A; its equivalent is
  `import dafoam` + initializing `PYDAFOAM(options={'solverName': ...})`.
* PYDAFOAM resolves the OpenFOAM case from the **current working directory**
  (`os.chdir(case_dir)`); there is no `runCaseDir` option key. Valid option
  keys were copied from the shipped regression scripts.
* Running as the bind-mounted host uid: `$HOME` in the image is mode 700
  owned by uid 1002; the Dockerfile grants read/traverse to all. Matplotlib
  also wants a writable config dir: export `MPLCONFIGDIR=/tmp/mplcfg`
  (handled by `scripts/run_in_container.sh`-driven calls).

## Case provenance and the one SEGV lesson

`cases/channel_baseline` is the supported **ConvergentChannel** case copied
from `DAFoam/reg_test_files` (main branch tarball, fetched 2026-08-04),
variant: `0.incompressible`, `system.incompressible`,
`constant/turbulenceProperties.dummy` (DAFoam's laminar path: zero model
states), plus an np=1 `decomposeParDict`.

A hand-rolled 2-D `blockMesh` channel (front/back `empty` patches) made
`solver.getStates()` SEGV deterministically at the first native read. The
canonical ConvergentChannel (343 cells, 3-D, all-walls box) works.
Working hypothesis for the record: DAFoam v5 state registration is unhappy
with `empty` front/back patch semantics; treat `empty` patches as untested
for the state/residual bridge until vetted separately. (This matters later:
the article case is pseudo-2D — see remaining blockers.)


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
