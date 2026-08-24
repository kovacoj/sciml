# DAFoam Parity Table

| Method | Labels | velocity MSE-TV | e_u | pressure MSE-TV | e_p | pressure-drop error |
|---|---|---:|---:|---:|---:|---:|
| Paper U-Net reported | converged CFD | 0.015-0.040 | N/A | 0.053-0.140 | N/A | 13-15% |
| Solver-distilled | W20 | 0.04053 | 0.646 | 3843.76 | 0.995 | 99.5% |
| W20 teacher | truncated SIMPLE | 0.01298 | 0.406 | 0.0396 | 0.379 | 23.9% |
| Supervised DAFoam | converged CFD | NOT TRAINED | | | | |
| Physics augmented | converged CFD plus physics | NOT TRAINED | | | | |

Metrics use our reconstructed paper-style TV under the local DAFoam/HFDIB formulation.
