# DAFoam Final Campaign

## Compatibility
Exact TPFM CFD reproduction did not satisfy the gate. Results use the TPFM topology distribution under our DAFoam/HFDIB formulation.

## Training
Seed 22 was selected on validation performance. Training targets are states after 20 SIMPLE iterations; converged CFD states were not training labels.

## Result
Neural initialization improves equal-budget flow approximation, but does not reduce total convergence iterations.

- Test cases: 8/16 available campaign cases (frozen first eight)
- Median iteration saving: -2.41%
- Mean iteration saving: -2.18%
- Iteration wins: 0/8
- Median wall speedup: 1.064x
- Equal-budget k=5 mean velocity error: cold 0.7272, neural 0.5674
- Equal-budget k=5 wins: 8/8

## Limitation
The one-shot network is not a converged CFD replacement, and pressure accuracy remains weak.
