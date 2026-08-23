# Firedrake CFD Gallery

## 2D family
Six deterministic connected article-inspired fitted geometries were solved with Taylor-Hood FEM. Stokes initializes steady Navier-Stokes at `uin=0.1`, `nu=0.01`, and natural zero outlet traction.

- Cases: 6/6 converged
- Maximum mass imbalance: 1.789e-14
- Maximum assembled weak residual: 1.817e-16
- Pressure range: 12.235 to 23.768

## 3D extension
Topology E was extruded through `H=0.016 m` with no-slip top and bottom walls. This is a genuine `u(x,y,z)` solution, not a rendered 2D extrusion.

- Velocity DOFs: 40527
- Pressure DOFs: 2115
- Solve time: 55.0 s
- Mass imbalance: 3.150e-16
- Weak residual: 8.676e-12
- Pressure range: 19.873

## Claim boundary
Procedurally generated article-inspired fitted geometries solved with Firedrake Taylor-Hood FEM. This is not reproduction of the paper and does not use HFDIB or a trained 3D surrogate.
