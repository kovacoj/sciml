# Firedrake HFDIB weak-flow scaffold

This directory contains the lightweight geometry/network core and serial
Firedrake training infrastructure for a weak-flow HFDIB experiment. It provides:

- reconstruction of signed distance and normals from `mixer_64.npz`;
- interpolation of reconstructed geometry fields at arbitrary coordinates;
- first- and second-order HFDIB interpolation in float64 PyTorch, with samples
  at normal coordinates `0`, `d1`, and `d1 + d2`;
- outward normal search for fluid interpolation points; and
- a coordinate MLP for `(u_x, u_y, p)`;
- literal strong HFDIB residuals plus a shared Case0 first-derivative weak form
  with fixed discrete Riesz maps; and
- resumable training, diagnostics, field snapshots, and atomic run status.

## Data

The JSON configurations point to the sibling reference checkout by default:

```text
../tpfm_unet_reference/data/mixer_64.npz
```

`TPFMGeometry` reads only the archive's `inputs` array. Override the path when
constructing it if the dataset is elsewhere:

```python
from firedrake_hfdib_weak_flow.src import TPFMGeometry

geometry = TPFMGeometry("/path/to/mixer_64.npz", sample_index=0)
sigma = geometry.interpolate([[0.04, 0.06]], "signed_distance")
```

The input convention is `lambda > 0.5` on the solid side. At diffuse-interface
cells, signed distance is recovered from
`sigma = h * atanh(1 - 2*lambda)`; elsewhere it comes from Euclidean distance
transforms. Normals point from solid to fluid.

### Dataset port audit

Audit invariant boundary openings and fluid connectivity across every sample
without importing Firedrake or running training:

```bash
python3 src/audit_tpfm_ports.py \
  --dataset ../tpfm_unet_reference/data/mixer_64.npz \
  --output-dir outputs/tpfm_port_audit
```

The audit writes `invariant_port_structure.json` and
`invariant_port_structure.png`. It uses the exact binary predicate
`fluid = lambda < 0.5`: intermediate diffuse-interface values below 0.5 count
as fluid, while `lambda == 0.5` and larger values do not. It reports edge-fluid
frequencies and invariant row intervals, exact left/right edge-mask patterns,
partial or closed expected DAFoam bands `[8,16)` and `[48,56)`, four-neighbor
fluid components, whole-edge left-to-right paths, and connectivity from each
expected left port to either expected right port. These are structural dataset
observations, not evidence that the cropped ROI boundaries are physical CFD
inlets/outlets and not grounds to revise the controlled domain specification.

## Lightweight tests

From this directory, with NumPy, SciPy, PyTorch, and pytest installed:

```bash
python -m pytest -q
```

The controlled geometry smoke uses a 20 by 10 mesh. Controlled topology
configurations use a 40 by 20 mesh. Their physical values come only from the
controlled domain spec.

Manufactured geometry smokes also use 20 by 10 meshes; manufactured neural and
direct-reference production configurations use 40 by 20 meshes.

## Geometry handoff

Commit `e78cf3f` is the endpoint of the cropped-ROI experiment and the starting
point for the geometry handoff. DAFoam is authoritative for all full-domain
dimensions, bounds, extension counts, patch intervals, ROI cell mapping, and
physical values. This project does not independently interpret or infer any of
those values.

`geometry/domain_spec.template.json` is a deliberately unfilled, valid-JSON
contract template. DAFoam must replace every `null`, including the complete
`roi_cell_indices` array with shape `(roi.ny, roi.nx)`. The strict loader rejects
missing and unknown fields, placeholder values, inconsistent grids, invalid
mappings, and overlapping or out-of-range patch intervals:

```python
from src import FullDomainGeometry, TPFMGeometry, load_domain_spec

spec = load_domain_spec("geometry/domain_spec.json")
roi = TPFMGeometry("/path/to/mixer_64.npz", spacing=spec.dx)
geometry = FullDomainGeometry(roi, spec)
```

Set `"domain_spec": "geometry/domain_spec.json"` in a training configuration;
the existing CLI remains `python3 -m src.train --config CONFIG --output-dir
OUTPUT`. A spec also requires an explicit `benchmark_mode`: `article` accepts
only `RECONSTRUCTED_TPFM_DOMAIN`, and `controlled` accepts only
`CONTROLLED_TPFM_DERIVED_DOMAIN`. The spec supplies `uin`, `pout`, and `nu`, so
those keys must be absent from benchmark configs.

Segmented inlet/outlet channels are reconstructed and constrained from physical
tangential intervals. The spec must partition every external side completely
with ordered inlet, outlet, or wall intervals; gaps and overlaps are rejected.
Firedrake selects the exact velocity and pressure DOFs and treats the explicitly
declared remainder as no-slip wall. Each
inlet or outlet patch collection must currently stay on one external side, but
may contain multiple nonoverlapping intervals. Arbitrary ROI mappings are still
validated by the loader but deferred by the rectangular geometry builder.

Without `domain_spec`, training retains the legacy cropped geometry with a
runtime warning. Cropped results are diagnostic only: they exercise the HFDIB
operator and gradient machinery but are not full-domain CFD reconstructions.

### Claim levels

1. **Operator verification.** Synthetic, manufactured, and controlled tests verify the discrete
   HFDIB operator, hard boundary conditions, dual norms, and external gradients.
   This is a numerical implementation claim only.
2. **Manufactured benchmarks.** `MANUFACTURED_EMPTY_CHANNEL` is an obstacle-free
   rectangular channel and `MANUFACTURED_CIRCULAR_HFDIB` is the same channel
   with an analytic diffuse circular obstacle. Both use `uin=0.1`, `pout=0`, and
   `nu=0.01`; neither uses TPFM data, has an ROI, or supports an article claim.
3. **Controlled benchmark.** `geometry/controlled_tpfm_32cell.json` records the
   exact 32-cell port-extension generator geometry and is classified
   `CONTROLLED_TPFM_DERIVED_DOMAIN`. Its provenance retains the failed source
   gate and `article_reproduction: false`. Results must be labelled “controlled
   TPFM-derived port-extension domain” and carry no article comparison claim.
4. **Article reproduction.** This level requires a separately supplied domain
   contract classified `RECONSTRUCTED_TPFM_DOMAIN` and `benchmark_mode: article`.
   Controlled results cannot be relabelled into this level.

The completed DAFoam campaign tested 8-, 16-, and 32-cell extensions and
classified its source as `TPFM_TOPOLOGIES_ONLY`; both source gates failed.
`geometry/dafoam_handoff_status.json` remains the authoritative record of that
outcome. The controlled contract does not alter it.

Run the geometry-only hard gate without constructing a network or optimizer:

```bash
python3 -m src.geometry_smoke \
  --config configs/controlled_geometry_smoke.json \
  --output-dir outputs/controlled_geometry_smoke
```

This writes `geometry_compatibility.json`, `geometry_controlled.png`, and an
accepted copy named `controlled_domain_spec.json`. Validation requires compatible
inlet/outlet nodes, finite geometry fields, center-based four-neighbor fluid
connectivity, and FE cell aspect error at most 0.1. Set
`allow_anisotropic_mesh: true` only as an explicit diagnostic override.

Run the manufactured geometry-only gates without loading a dataset or creating
a network/optimizer:

```bash
python3 -m src.geometry_smoke --config configs/manufactured_empty_geometry.json --output-dir outputs/manufactured_empty_geometry
python3 -m src.geometry_smoke --config configs/manufactured_circle_geometry.json --output-dir outputs/manufactured_circle_geometry
```

Each writes `geometry_compatibility.json`, `geometry_labels.json`, and a labelled
PNG. The circular signed distance, normals, and diffuse lambda are evaluated
analytically at arbitrary query coordinates rather than reconstructed by a
distance transform.

## Direct reference

The independent empty-channel reference is a conventional mixed P2/P1 steady
Navier-Stokes solve with advective convection, symmetric viscous stress,
full-side inlet velocity, top/bottom no slip, and a natural zero-traction outlet
that fixes the pressure level without removing outlet continuity test functions.
A tested full-outlet mixed pressure Dirichlet condition was rejected because it
materially degraded coarse-grid mass conservation. The solve runs serial
Newton/SNES with direct MUMPS LU:

```bash
python3 -m src.direct_reference \
  --config configs/manufactured_empty_neural.json \
  --output-dir outputs/manufactured_empty_direct
```

It writes FE coefficients to `direct_reference.npz` and convergence, divergence,
flux, mass, speed, pressure, and config metadata to
`direct_reference_metrics.json`. These files are evaluation artifacts and are
never read by neural training. A circular direct invocation intentionally writes
only deferred metrics: reproducing the literal strong HFDIB term requires a
nonlocal `u_ib` fixed-point mapper, while a body-fitted obstacle solve would be a
different operator and is not substituted.

After a neural run, compare matching-grid coefficient snapshots with the direct
reference using simple, explicitly unweighted coefficient l2 errors and a
mean-centered pressure gauge:

```bash
python3 -m src.compare_reference \
  --neural outputs/manufactured_empty_neural/fields_0400.npz \
  --reference outputs/manufactured_empty_direct/direct_reference.npz \
  --output outputs/manufactured_empty_neural/reference_comparison.json
```

`--neural` may instead name a `.pt` checkpoint when the matching neural config
is supplied with `--config configs/manufactured_empty_neural.json`; the helper
evaluates its fields but does not train or alter the checkpoint.

Manufactured neural configs are fresh-only across geometry identities: resume or
initialization checkpoints must carry the same `geometry_kind` and
`benchmark_case`. Existing legacy exact-resume behavior remains unchanged.
Their pressure output scale is `0.2`, set from the channel viscous estimate
`12 * nu * uin * Lx / Ly^2 = 0.1875`, rather than inherited from the TPFM
diagnostic configuration.

### Manufactured validation outcome

The final empty-channel neural run completed 400 steps and reduced normalized
total loss to `0.05828`, but it did not reproduce the conventional mixed-FE
reference. Final mass imbalance was `0.6930` (direct: `1.16e-14`), velocity-x
coefficient error was `0.6980`, and gauge-centered pressure error was `0.9589`.
The circle and controlled-TPFM neural runs are therefore intentionally not
launched.

This is a method-validation stop, not a geometry failure: the manufactured
channel has compatible boundary conditions, no immersed solid, a connected
fluid domain, a passing external-gradient check, and a converged direct
reference. The present weighted strong-residual objective can decrease while
retaining a near-stagnant, globally imbalanced field. Future work must repair
that objective or its admissible test treatment before adding HFDIB complexity.

The Case 0 replacement uses one shared first-derivative Taylor-Hood weak form
for direct solving and dual-residual assembly. Injecting the direct Stokes field
gives dual loss `1.07e-30`. A nondimensionalized 40 by 20 coefficient-space
L-BFGS solve reaches loss `5.40e-10`, `ux` error `3.86e-5`, gauge-centered
pressure error `3.08e-4`, and mass imbalance `2.39e-5`. This validates the weak
objective independently of the coordinate MLP; HFDIB remains on the unchanged
literal-strong path pending successful neural Case 0 validation.

The coordinate MLP was then trained without reference labels for 1000 Adam
steps and refined for three fixed blocks of 1000 exact-gradient L-BFGS
iterations. Its final Stokes result has loss `3.33e-4`, `ux` error `0.0136`,
gauge-centered pressure error `0.0626`, divergence L2 `0.03150` (direct:
`0.03195`), and mass imbalance `0.0120`. This is close in field shape but misses
the predeclared `ux < 0.01` and mass-imbalance `< 1e-4` validation gates.
Optimization is stopped at that fixed budget; circular HFDIB and controlled
TPFM neural runs remain blocked.

An additional analytic A/B/C channel-network family tests a separate
Navier-Stokes-Brinkman application (not article HFDIB). Direct segmented-port
solves are mass-balanced, but the 32 by 32 penalty gate is not acceptable:
solid leakage remains `0.235` even at `alpha=1e6`, where divergence L2 degrades
to `0.677`. No Brinkman neural or recycling runs are launched from this failed
physical/discretization gate.

The bounded 64 by 64 diagnosis tests only alpha 625/1250/2500. Leakage remains
`0.694/0.621/0.538`; a single sharp-indicator run at alpha 2500 gives `0.403`.
Thus one refinement and removal of diffuse lambda improve but do not resolve
the gate. The appendix reports the Brinkman length `sqrt(nu/alpha)` relative to
mesh spacing and freezes the experiment without neural fitting.

A bounded fitted-domain follow-up meshes analytic Topology A channels
conformingly instead of penalizing a fixed background domain. Its direct
Taylor-Hood Stokes solve has assembled weak residual norm `4.88e-16` and mass
imbalance `2.93e-15`, demonstrating that the level-set-derived BVP is clean.
Neural A/B recycling is deferred because it requires a dedicated variable-mesh
residual mapper; this was not improvised after the implementation time gate.

That result is retained as the strong-residual diagnostic. Case0 now selects
`"residual_formulation": "h1_weak"`: direct and neural/coefficient paths call
`src.weak_forms.navier_stokes_weak_form`, use first derivatives only, leave all
pressure coefficients free, and obtain `p_out=0` from natural zero traction.
`h1_weak` is rejected for circular, controlled, reconstructed, and legacy
geometries; those paths retain `literal_strong_hfdib` unchanged.

The validation hierarchy for the shared form is:

1. Inject the matching-grid direct P2/P1 solution and require each assembled
   weak residual coefficient norm below `1e-8`.
2. Check the full external gradient at `beta=0` and `beta=0.25` by centered
   finite differences.
3. Optimize free FE coefficients from the boundary lift without reading the
   direct solution, then optionally compare only after optimization:

```bash
python3 -m src.optimize_fe_coefficients \
  --config configs/manufactured_empty_fe_coefficients.json \
  --output-dir outputs/manufactured_empty_fe_coefficients \
  --reference outputs/manufactured_empty_direct/direct_reference.npz
```

The optimizer writes `normalization.json`, `final_fields.npz`, and `metrics.json`.
Both training paths report normalized momentum/continuity components, signed
constant-test continuity, absolute mass defect, and relative mass imbalance.

## Training

Launch and inspect the smoke run without running it in the foreground:

```bash
scripts/launch_nohup.sh smoke configs/smoke.json
scripts/status.sh smoke
scripts/stop.sh smoke
```

The launcher automatically resumes `checkpoint_latest.pt`. Direct CLI use in
the Firedrake image is also supported with `python3 -m src.train --config ...
--output-dir ...`. Generate available figures and file-backed presentation
metrics with `python3 plot_results.py`.
