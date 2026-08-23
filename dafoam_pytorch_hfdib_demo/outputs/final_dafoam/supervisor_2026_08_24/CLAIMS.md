# Supported

- Training used K=20 truncated SIMPLE states, not converged CFD labels.
- Evaluation uses held-out TPFM topologies under our DAFoam/HFDIB formulation.
- Equal-budget neural starts improve velocity error on the evaluated cases.

# Not Supported

- Exact reproduction of the published CFD benchmark.
- One-shot CFD replacement or accurate pressure prediction.
- Reduced convergence iterations; the measured eight-case result is negative.
