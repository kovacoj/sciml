# Stokes Preconditioning Research Claims

## Supported

- Exact PDE-correction training closely tracks an unavailable FE-error oracle
  in the validated coefficient experiment.
- Raw and dual residual optimization become substantially less effective under
  mesh refinement in the tested fixed 300-iteration budget.
- Exact correction is consistently best in the existing five-seed coordinate
  MLP pilot at 16x8 and 32x16.
- Directional gradient checks pass for every validated loss.
- The exact correction and oracle objectives agree in value and coefficient
  gradient to numerical tolerance under the stated discrete assumptions.

## Not Yet Established

- Cheap approximate corrections retain exact-correction performance.
- Neural convergence is mesh independent.
- Exact correction offers a practical wall-time advantage over conventional FE
  solution or over all approximate residual metrics.
- The discrete linear equivalence extends as a general theorem to nonlinear
  Navier-Stokes or infinite-dimensional spaces.
- Any Richardson/preconditioner result from the currently conflicting
  uncommitted worktree experiment.

## Scope

The theorem and experiments concern a finite-dimensional, gauge-fixed,
invertible Taylor-Hood Stokes operator with fixed boundary conditions. Direct FE
coefficients are used only for post-optimization error measurement and the
explicit oracle control, never by raw, dual, block, or correction training.

## Primary Validated Numbers

At 64x32 with 300 coefficient L-BFGS iterations:

| Method | Velocity error | Pressure error | Wall time |
|---|---:|---:|---:|
| Raw | 0.978 | 1.015 | 2.37 s |
| Dual | 0.224 | 0.711 | 7.49 s |
| Jacobi-LS | 0.998 | 1.001 | 4.63 s |
| Block | 0.189 | 0.606 | 12.32 s |
| Exact correction | 0.00607 | 0.429 | 10.16 s |
| Oracle | 0.00523 | 0.427 | 1.84 s |

Five-seed median coordinate-MLP velocity errors:

| Mesh | Raw | Dual | Correction |
|---|---:|---:|---:|
| 16x8 | 0.327 | 0.0847 | 0.0513 |
| 32x16 | 0.428 | 0.0933 | 0.0625 |

All tables and figures are generated from the validated archived CSV/JSON files.
