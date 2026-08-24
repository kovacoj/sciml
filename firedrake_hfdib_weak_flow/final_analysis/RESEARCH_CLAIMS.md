# Supported

- Exact PDE-correction training closely tracks an unavailable FE-error oracle in the coefficient experiment.
- Raw and dual residual optimization become substantially less effective under mesh refinement at the tested fixed optimization budget.
- Correction loss is consistently best in the existing five-seed neural pilot.
- Gradient and exact correction/oracle equivalence tests pass.

# Not Yet Established

- Cheap approximate corrections retain exact-correction performance.
- Mesh-independent neural convergence.
- A practical wall-time advantage over conventional FEM.
- A general theorem for nonlinear Navier-Stokes.
- Richardson/preconditioner results from the conflicting uncommitted experiment.

The uncommitted Richardson reversal is preserved and excluded from this package.
