# Stokes Preconditioning Research Results

## Completed

- Three-mesh deterministic coefficient comparison.
- Five-seed matched-initialization MLP comparison at 16x8 and 32x16.
- Exact correction/oracle equivalence and gradient tests.
- Expanded coefficient comparison with raw, dual, Jacobi-LS, block, exact,
  and oracle metrics.
- Thirty serialized MLP checkpoints with SHA-256 manifest.

## Main result

Five-seed median MLP velocity errors after 300 identical L-BFGS iterations:

| mesh | raw | dual | correction |
|---|---:|---:|---:|
| 16x8 | 0.327 | 0.0847 | 0.0513 |
| 32x16 | 0.428 | 0.0933 | 0.0625 |

Correction is best and substantially less mesh-sensitive. In coefficient space,
exact correction closely follows the unavailable oracle objective.

## Approximate inverses

Diagonal normal-equation Jacobi is ineffective. A simple exact velocity-block
plus pressure-mass block improves over dual at coarse meshes but degrades under
refinement. No practical approximate inverse yet bridges the dual-to-exact gap.

## Environment blocker

The planned 64x32 five-seed MLP and 128x64 coefficient campaigns could not
start because Docker/containerd repeatedly returned connection-refused errors,
including with one worker and a 4 GB cap. No partial campaign artifacts were
created. These entries are recorded as `NOT_RUN_DOCKER_UNSTABLE`, not failed
numerical experiments.

## Next experiment

After environment recovery:

1. finish 64x32 five-seed MLP raw/dual/correction;
2. finish 128x64 coefficient raw/dual/correction/oracle;
3. implement fixed linear Richardson operators with exact transpose actions;
4. compare error versus total wall time and time to fixed velocity thresholds;
5. investigate velocity-H1/pressure-L2 block scaling.
