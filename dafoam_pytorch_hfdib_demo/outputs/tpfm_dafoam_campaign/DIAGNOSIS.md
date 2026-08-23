# TPFM/DAFoam baseline diagnosis

## Baseline

| Sample | Relative velocity L2 | Gauge-centered pressure L2 | Local/reference pressure range |
|---:|---:|---:|---:|
| 0 | 0.180377 | 0.376740 | 1.980712 |
| 274 | 0.195568 | 0.464235 | 2.382900 |
| 549 | 0.203469 | 0.456230 | 2.103710 |

Identity is the best tested orientation for all three samples.

## Published ROI boundaries

Published left-edge fluid-opening `Ux` is not the prescribed `0.1`: means range from approximately `0.027` to `0.184`, with substantial spatial variation. Published left-edge `Uy` is also nonzero, reaching about `0.119` in magnitude. Published right-edge pressure is not the prescribed zero: opening means range from approximately `1.28` to `3.45`.

The archive semantics are `lambda=0` fluid and `lambda=1` solid. Mean published speed is `0.108904` in lambda-near-zero cells and `4.52e-7` in lambda-near-one cells.

## Working hypothesis

**A. 64x64 is an interior ROI of a larger physical CFD domain.**

The physical values in the local case match the article, but the local case applies them directly on the exported ROI edges. The observed published edge fields are inconsistent with those edges being the physical inlet and outlet. The article schematic shows fixed external port extensions, but their numerical lengths and the original full OpenFOAM case are not available in the public repository.
