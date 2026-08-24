# Paper-like 64x64 fitted CFD

## Topology generation

- Source: official TPFM `mixer_64.npz` (550 samples)
- Selection: geometry-only descriptors and robust-z class scoring
- Coarse interior representation: 8x8 bitmap extracted from 64x64 field
- Common display/evaluation raster: 64x64
- Two fixed left ports (rows 10:14, 50:54)
- Two fixed right ports (same rows)
- Eight-cell external stubs on both sides
- All four ports connected in one fluid component

## Selected cases

| Case | Source ID | Class | Tortuosity | Fluid fraction |
|---|---:|---|---:|---:|
| P64-A | 34 | tortuous_network | 1.794 | 0.361 |
| P64-B | 431 | central_merge_split | 1.159 | 0.409 |
| P64-C | 508 | parallel_channels | 1.000 | 0.382 |

## Physics

- uin = 0.1
- nu = 0.01
- Taylor-Hood P2/P1
- Steady Navier-Stokes
- Continuation: beta = 0, 0.25, 0.5, 1.0

## CFD status

Docker/containerd was unavailable during this sprint. All three cases have
complete geometry artifacts (lambda, binary mask, fitted mesh, connectivity)
but no CFD solution. The solver code (`src/solve_rugged_topologies.py`) is
resumable and will solve these exact cases when Docker recovers.

## Claim boundary

These are official TPFM topology fields used under our fitted-domain
Taylor-Hood FEM formulation. They are not exact reproductions of the article
CFD domain and do not use HFDIB. The fitted mesh preserves the blocky bitmap
character of the topology exactly.
