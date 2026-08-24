# Paper Parity Report

Metric normalization: `paper_figure6`. TV definition: reconstructed mean absolute forward differences; not bit-identical to the unavailable reference implementation.

| Method | velocity MSE-TV | e_u | pressure MSE-TV | e_p | pressure-drop error |
|---|---:|---:|---:|---:|---:|
| Paper reported range | 0.015-0.040 | N/A | 0.053-0.140 | N/A | 13-15% |
| Solver-distilled k=0 | 0.0405 | 0.646 | 3843.758 | 0.995 | 99.5% |
| W20 teacher | 0.0130 | 0.406 | 0.0396 | 0.379 | 23.9% |
| Supervised DAFoam | NOT TRAINED | | | | |
| Physics augmented | NOT TRAINED | | | | |

Solver-distilled paper velocity parity: **FAIL** (`0.0405 > 0.025`). Supervised training is blocked because converged labels exist for 0/256 training and 0/32 validation cases, and Docker is unavailable.
