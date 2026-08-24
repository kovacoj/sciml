# Paper Comparison Notes

## Reference paper

- Geometry: 8x8 bitmap -> 64x64 lambda field -> fixed-grid HFDIB.
- Lambda convention: 0 fluid, 1 solid.
- Physics: uin=0.1, nu=1e-2, pout=0, steady low-Re Navier-Stokes.
- Figure 6: three unseen 64x64 topologies.
- Columns: lambda, IBM relative velocity, U-Net relative velocity, IBM relative pressure, U-Net relative pressure.

| Case | MSE-TV u | MSE-TV p | global continuity | pressure-drop ratio |
|---|---:|---:|---:|---:|
| 1 | 0.040 | 0.140 | -3.9e-6 | 1.14 |
| 2 | 0.025 | 0.093 | 3.8e-6 | 1.13 |
| 3 | 0.015 | 0.053 | -6.6e-7 | 1.15 |

The paper reports velocity fields as visually difficult to distinguish, occasional ghost currents, more apparent pressure differences, and about 15% excessive pressure drop in these unseen cases.

## Firedrake morphology mapping

- R1 complex/tortuous network -> Topology D: asymmetric offset multi-bend paths.
- R2 merge/split/chamber -> Topology E: broad chamber connected to four branches.
- R3 parallel channels -> Topology A: two long horizontal paths with one bridge.

Reference uses fixed-grid HFDIB topology fields. Firedrake uses geometry-dependent fitted meshes and direct Taylor-Hood Navier-Stokes solves. These are comparable in physical intent, not identical discretizations or reproduced geometries.

Paper global continuity and Firedrake mass imbalance/assembled weak residual are not definition-identical. Do not claim an orders-of-magnitude improvement from juxtaposing them.
