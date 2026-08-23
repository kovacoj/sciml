# Stokes Residual Preconditioning Study

This branch isolates a post-presentation research question. The PDE, P2/P1
spaces, boundary conditions, initial coefficients, and optimizer are fixed.
Only the residual metric changes:

1. raw coefficient residual: `r.T @ r`;
2. test-space dual norm: `r.T @ solve(G, r)`;
3. exact PDE correction: `delta.T @ X @ delta`, `A delta = r`.

For affine Stokes residual `r(z)=Az-b`, exact correction gives
`delta=z-z_star`; therefore its loss is exactly the discrete error in norm `X`
and has Hessian `2X`, while raw least squares has Hessian `2 A.T A`.

The pilot compares optimizer iterations, residual evaluations, wall time, and
coefficient error over mesh refinement. Exact correction-solve cost is counted;
direct FE solutions are used only to measure post-optimization error.

## Deterministic coefficient pilot

All methods start from the same boundary lift and receive 300 L-BFGS
iterations. Velocity relative errors are:

| mesh | raw | dual | exact correction |
|---|---:|---:|---:|
| 16x8 | 5.20e-1 | 6.98e-3 | 1.43e-4 |
| 32x16 | 9.08e-1 | 5.20e-2 | 3.13e-3 |
| 64x32 | 9.78e-1 | 2.24e-1 | 6.07e-3 |

Raw normalized loss decreases under refinement even while its actual field
error approaches one, so the raw loss is not mesh-comparable. The correction
metric has the largest per-step cost but is markedly less mesh-sensitive in
field error. Pressure converges more slowly than velocity at the fixed budget,
which motivates block-norm/preconditioner variants.

Multiple seeds are intentionally deferred for this convex coefficient pilot:
with a fixed initial vector they add no statistical information. The next stage
will compare identical coordinate-MLP initializations across losses and use
multiple seeds there, where nonconvex optimization makes them meaningful.
