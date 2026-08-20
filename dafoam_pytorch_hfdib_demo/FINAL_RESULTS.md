# Final Results

## Scope

This study evaluates a solver-informed neural flow surrogate trained from
truncated SIMPLE trajectories. It is not a classical PINN: the training targets
are states obtained after a fixed number of SIMPLE iterations from a common
initial state, and converged CFD solutions are reserved for evaluation.

The primary benchmark is the local 64 x 64 four-port HFDIB problem. An audit of
the published TPFM data was also completed, but exact cross-benchmark comparison
was rejected because the published arrays cover an interior area of interest
rather than the full computational domain.

## Model And State

- Model: `SimpleFlowNet`, 3,153,264 trainable parameters.
- Mesh: 4,096 cells, 8,064 internal faces, and 16,512 total faces.
- State size: 32,896, ordered as cell-major velocity, pressure, then face flux.
- Predicted fluxes: 8,064 internal faces and 16 outlet faces.
- Scales: velocity `0.1`, pressure `0.01`, and flux `4e-7`.
- Held-out local cases: `topology_128` through `topology_143`.

## Validated Local Results

### Fixed Solver Budget

The target-generation budget was held near `N x K = 2560`, where `N` is the
number of training geometries and `K` is the number of SIMPLE iterations used
to construct each target.

| N | K | Relative velocity error | Relative pressure error |
|---:|---:|---:|---:|
| about 496 | 5 | 0.448 | 7.439 |
| 256 | 10 | 0.296 | 0.943 |
| 128 | 20 | 0.286 | 0.398 |
| 64 | 40 | 0.366 | 0.368 |
| 32 | 80 | 0.458 | 0.373 |

Within this budget, intermediate target depth gave the best velocity accuracy.
Very shallow targets were particularly poor for pressure, while deeper targets
reduced geometric diversity and degraded velocity generalization.

### Training-Set Scaling

Single-run evaluations gave:

| N | Relative velocity error | Relative pressure error |
|---:|---:|---:|
| 128 | 0.304 | 0.388 |
| 256 | 0.296 | 0.392 |
| 384, best run | about 0.105 | about 0.341 |
| 512, best run | about 0.126 | about 0.339 |

The best-run figures must not be interpreted as seed-robust trends. The
three-seed experiment below is the stronger comparison.

### Three-Seed Large-N Comparison

All runs used 10,000 optimization steps.

| N | Seed | Relative velocity error | Relative pressure error |
|---:|---:|---:|---:|
| 384 | 0 | 0.1131 | 0.3415 |
| 384 | 1 | 0.2586 | 0.3531 |
| 384 | 2 | 0.1129 | 0.3441 |
| 512 | 0 | 0.1549 | 0.3608 |
| 512 | 1 | 0.3086 | 0.3589 |
| 512 | 2 | 0.1082 | 0.3457 |

Aggregate results:

| N | Relative velocity error, mean +/- SD | Relative pressure error, mean +/- SD |
|---:|---:|---:|
| 384 | 0.1615 +/- 0.0686 | 0.3462 +/- 0.0050 |
| 512 | 0.1906 +/- 0.0856 | 0.3551 +/- 0.0067 |

Increasing the training set from 384 to 512 cases produced no measurable
improvement within seed variability. Velocity training showed substantial seed
sensitivity, while pressure performance was comparatively stable.

### Negative Results

- Direct raw-residual or Jacobian-transpose-vector descent did not correlate
  reliably with field accuracy.
- Moving detached SIMPLE targets were unstable.
- These observations motivated fixed truncated-SIMPLE targets.

## Provenance-Limited Results

An earlier 16-case warm-start experiment reported:

- Median iterations to `rel_U < 0.20`: cold start 20, neural start 5, a 4x
  iteration ratio.
- Median iterations to `rel_U < 0.15`: cold start 20, neural start 10, a 2x
  iteration ratio.
- Mean pressure-drop ratio: approximately 0.857, with 11 of 16 cases within
  +/-20%.

These are preliminary observations, not final acceleration claims. The six
available N=384 and N=512 checkpoints were classified as
`INCOMPLETE_PROVENANCE`: they omit the random seed, target depth, exact training
IDs, state and layout hashes, and normalization metadata. An attempted rerun
also produced ambiguous initial neural errors consistent with an older model.
Consequently, these checkpoints must not be used for definitive warm-start
claims.

New checkpoints now serialize seed, target depth, training IDs and split hash,
base-state hash, mesh hashes, normalization hash, scales, and state layout. A
new provenance-complete training and warm-start run is required before reporting
solver acceleration as a validated result.

## TPFM Compatibility Audit

The official `mixer_64.npz` archive contains 550 inputs of shape `(1, 64, 64)`
and outputs of shape `(4, 64, 64)`. The output channels were empirically
identified as `Ux`, `Uy`, `Uz`, and pressure, with `Uz = 0`. The topology
convention is lambda zero for fluid and lambda one for solid.

The TPFM area-of-interest coordinates match the local mesh to approximately
`1.39e-17`; both use `h = 0.002 m` over a `0.128 m x 0.128 m` region. Nominal
parameters also match: inlet velocity `0.1`, kinematic viscosity `0.01`, and
outlet pressure `0`.

Exact CFD equivalence nevertheless failed on three cases:

| Case | Relative velocity error | Gauge-centered pressure error | Local/published pressure-range ratio |
|---:|---:|---:|---:|
| 1 | 0.1804 | about 0.377-0.464 | 1.98 |
| 2 | 0.1956 | about 0.377-0.464 | 2.38 |
| 3 | 0.2035 | about 0.377-0.464 | 2.10 |

The failure is attributable to a computational-domain mismatch:

- The published 64 x 64 arrays are an interior area of interest, not the full
  CFD domain.
- Published left-edge velocity is spatially varying and therefore is not the
  imposed inlet boundary condition.
- Published right-edge pressure ranges from about 1.28 to 3.45 and therefore is
  not the zero-pressure outlet boundary.
- The paper shows upstream and downstream port extensions outside the reported
  region, but the public materials do not specify their lengths or full mesh.
- The HFDIB interface treatment also differs: the paper uses
  `0.5 * (1 - tanh(sigma / h))`, while the local signed-distance construction
  uses a `1.5h` transition and snaps negative-distance cells to lambda one.

Reconstructing unspecified extension lengths from figure pixels would not be
academically defensible. No TPFM truncated-SIMPLE targets were therefore
generated, and no apples-to-apples TPFM model claim is made. The geometries may
only be reused under the explicit label "TPFM topology geometries under our
DAFoam/HFDIB flow formulation."

## Conclusions

The local results support truncated SIMPLE trajectories as useful supervision
for a solver-informed neural flow surrogate without converged CFD training
labels. Under a fixed target-generation budget, target depth and geometric
diversity must be balanced; more geometries alone did not improve the large-N
three-seed result. Definitive CFD acceleration remains unproven until a fresh,
provenance-complete checkpoint is evaluated in a controlled cold-start versus
warm-start convergence study.

The TPFM audit is a hard scope boundary rather than a failed tuning exercise:
matching the reported grid and nominal physical parameters is insufficient when
the published arrays omit the upstream and downstream portions of the CFD
domain.

## Evidence Artifacts

- `outputs/final_campaign/checkpoint_audit.json`
- `outputs/final_campaign/test_set.json`
- `outputs/final_campaign/tpfm_dataset_manifest.json`
- `outputs/final_campaign/tpfm_mesh_compatibility.json`
- `outputs/final_campaign/tpfm_cfd_cross_validation.json`
- `outputs/final_campaign/tpfm_cfd_cross_validation_1_5h.json`
- `outputs/final_campaign/tpfm_roi_boundary_probe.json`
- `outputs/final_campaign/tpfm_roi_boundary_probe.png`
- `outputs/final_campaign/configuration_audit.json`
- `outputs/final_campaign/tpfm_geometry_comparison.json`
- `outputs/fixed_budget_study/`
- `outputs/n384_seeds/`
- `outputs/n512_seeds/`
