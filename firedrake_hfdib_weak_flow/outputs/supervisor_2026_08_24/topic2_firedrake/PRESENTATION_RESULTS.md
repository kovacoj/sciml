# Firedrake CFD Gallery

## Main Result

Six procedurally generated connected article-inspired fitted geometries were solved with Taylor-Hood FEM. The fluid boundary is represented by the conforming mesh; no Brinkman or HFDIB approximation is used in these gallery solves.

- Cases converged: 6/6
- Maximum mass imbalance: 1.789e-14
- Maximum assembled weak residual: 1.817e-16
- Pressure-drop range: 12.235 to 23.768

| Topology | Pressure drop | Maximum speed | Mass imbalance | Weak residual |
|---|---:|---:|---:|---:|
| A | 23.768 | 0.141 | 6.505e-15 | 1.817e-16 |
| B | 15.487 | 0.157 | 9.758e-15 | 1.528e-16 |
| C | 16.138 | 0.301 | 1.422e-14 | 1.788e-16 |
| D | 15.670 | 0.160 | 1.789e-14 | 1.392e-16 |
| E | 12.235 | 0.188 | 1.097e-14 | 1.299e-16 |
| F | 12.612 | 0.181 | 6.984e-15 | 1.005e-16 |

## Genuine 3D Topology E

Topology E was extruded through a 0.016 m depth with no-slip top and bottom walls. This is a solved three-dimensional velocity field, not a graphical extrusion.

- Velocity DOFs: 40,527
- Pressure DOFs: 2,115
- Solve time: 55.0 s
- Mass imbalance: 3.150e-16
- Weak residual: 8.676e-12
- Pressure drop/range proxy: 19.873

## Recommended Figures

### Main slides

1. `figures/firedrake_cfd_gallery_relative.png`: topology diversity and internal flow structure.
2. `figures/firedrake_streamline_gallery.png`: visually clear jets, splits, merges, and bottlenecks.
3. `figures/firedrake_3d_flow_topology_E.png`: strongest demonstration of genuine 3D CFD.
4. `figures/firedrake_detailed_topology.png`: fitted mesh plus detailed fields.

### Scientific/appendix

- `figures/firedrake_cfd_gallery_common_scale.png`: comparable absolute field scales.
- `figures/firedrake_pressure_drop_topologies.png`: resistance comparison.
- `figures/firedrake_2d_vs_3d.png`: completed 2D/3D comparison.
- `figures/firedrake_fitted_mesh_examples.png`: conforming-mesh explanation.

## Claim Boundary

These are procedurally generated article-inspired fitted geometries solved with Firedrake Taylor-Hood FEM. They are not reproductions of the article CFD domain and do not use HFDIB. The paper uses fixed-grid HFDIB; this gallery uses geometry-dependent fitted meshes and strong no-slip wall conditions.

The final figures use grayscale geometry, viridis velocity, and coolwarm pressure, matching the DAFoam presentation package.
