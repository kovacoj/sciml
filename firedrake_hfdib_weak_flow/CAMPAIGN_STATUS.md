# Six-Hour Stokes Preconditioning Campaign

## Status

`PARTIAL_ENVIRONMENT_BLOCKED`

Docker/containerd repeatedly failed actual container starts while `docker info`
briefly remained responsive. Failures included connection refusal and connection
reset on a one-line Alpine probe. No partial numerical runs are treated as data.

## Completed Evidence

### Coefficient-space mesh study

Velocity error after 300 matched L-BFGS iterations:

| Mesh | Raw | Dual | Exact correction |
|---|---:|---:|---:|
| 16x8 | 0.520 | 0.00698 | 0.000143 |
| 32x16 | 0.908 | 0.0520 | 0.00313 |
| 64x32 | 0.978 | 0.224 | 0.00607 |

### Five-seed coordinate-MLP study

Median velocity error after 300 matched L-BFGS iterations:

| Mesh | Raw | Dual | Exact correction |
|---|---:|---:|---:|
| 16x8 | 0.327 | 0.0847 | 0.0513 |
| 32x16 | 0.428 | 0.0933 | 0.0625 |

### Exact and approximate coefficient losses at 64x32

| Method | Velocity error | Pressure error | Wall time |
|---|---:|---:|---:|
| Raw | 0.978 | 1.015 | 2.37 s |
| Dual | 0.224 | 0.711 | 7.49 s |
| Jacobi-LS | 0.998 | 1.001 | 4.63 s |
| Block | 0.189 | 0.606 | 12.32 s |
| Exact correction | 0.00607 | 0.429 | 10.16 s |
| Oracle | 0.00523 | 0.427 | 1.84 s |

Exact correction tracks the unavailable oracle closely. Jacobi-LS is
ineffective, and the simple velocity/pressure block loses robustness under mesh
refinement.

## Not Run

- 64x32 five-seed MLP campaign;
- 128x64 coefficient campaign;
- Richardson Pareto campaign;
- balanced block-norm campaign.

These are `NOT_RUN_DOCKER_UNSTABLE`, not negative numerical results.

## Reproducibility

Existing artifacts and 30 network checkpoints are archived under:

`outputs/stokes_preconditioning_research/`

with SHA-256 hashes in `manifest.json`.

## Resume Order

1. Restore stable Docker/containerd operation and verify three consecutive
   container starts.
2. Resolve the currently conflicting uncommitted Richardson worktree state.
3. Run 64x32 five-seed MLP raw/dual/correction serially.
4. Attempt 128x64 coefficient raw/dual/correction/oracle under a memory cap.
5. Validate fixed-linear approximate inverses with transpose and directional
   gradient checks before optimization.
