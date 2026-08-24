# DAFoam U-Net Parity Campaign

## Status

`PHASE_A_COMPLETE_SUPERVISED_BLOCKED`

- Existing solver-distilled model benchmarked on eight frozen held-out cases.
- Paper-style metric normalization implemented and tested.
- TV is reconstructed forward-difference TV, not claimed bit-identical.
- Solver-distilled velocity parity gate failed: median `0.04053 > 0.025`.
- Converged labels available: train `0/256`, validation `0/32`, test `8/16`.
- Docker/containerd is unavailable, so valid train/validation labels cannot be generated.
- Supervised and physics-augmented models were not trained.

Training on the eight available test references is prohibited because it would
violate the frozen split and invalidate unseen-topology claims.
