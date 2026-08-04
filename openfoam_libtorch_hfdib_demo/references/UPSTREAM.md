# Upstream References

Inspection date: 2026-08-04. Both repositories were cloned to a scratch area
outside this git repository (nothing vendored in).

## 1. tpfm_unet (supervised U-Net reference)

* URL: https://github.com/techMathGroup/tpfm_unet
* Commit inspected: `c0566779b54877ca46053a8ff44096c3351f4805`
  (2026-07-03, "Add dataset DOI badge to README")
* Dataset pointer in README: DOI 10.5281/zenodo.21159304 (full CFD dataset,
  not downloaded).

### Files consulted

`README.md`, `src/model.py`, `src/datamodule.py`,
`src/dataset_preparation.py`, `src/utils.py`, `configs/config.yaml`,
`configs/model/unet.yaml`, `configs/dataset/mixer_64.yaml`, `data/`.

### Key technical facts

* Supervised PyTorch-Lightning project. Default model (`configs/model/unet.yaml`):
  `in_channels: 1` (topology only), `out_channels: 4` (Ux, Uy, Uz, p̃),
  base_filters 16, depth 3, BatchNorm+ReLU blocks, lr 3e-4.
* Loss: MSE + `tv_weight=0.1` total variation on the prediction + an optional
  fixed Gaussian smoothing convolution (kernel 5, sigma 1.0, depthwise,
  `use_smoothing: True`).
* Data normalization (`datamodule.Normalizer`): dataset-wide per-channel
  mean/std fitted on the **targets only**.
* `dataset_preparation.py`: raw `.dat` files, first 4 channels are outputs
  (Ux, Uy, Uz, p̃), channels 4+ are inputs; `coordinate_64.csv`-style
  coordinate file maps cellI -> (x,y,z,V).
* `data/coordinates_64.csv` (in repo): 4096 cells; cell centers x,y in
  [0.001, 0.127]; **dx = dy = 0.002 m**; z = 0.001; **V = 8e-9 = 0.002³**
  (cubic cells). => The 64x64 area of interest is 0.128 m x 0.128 m and the
  extrusion thickness is 0.002 m.

### Lambda convention conflict (IMPORTANT)

* README prose: "binary mask where 0 = wall, 1 = fluid".
* Code: `src/utils.py:10 mask_lambda` computes `mask = 1.0 - y[:, lambda_index]`
  and multiplies the velocity channels by it -> velocity is zeroed where
  lambda == 1. `src/model.py:120 mask_lambda` zeroes predictions where
  `y[:, lambda_index] == 1`.
* `data/mixer_64.npz` (550-sample demo in repo): inputs[:, 0] takes exactly
  six values {0, 0.27016124, 0.33120811, 0.66879189, 0.72983876, 1} — bulk 0/1
  plus two symmetric intermediate pairs (sum to 1), i.e. a 5-level body field
  with interface cells at intermediate spacing. Mean solid fraction 0.656.
* => The data and masking utility implement the paper's convention
  **lambda = 0 fluid, lambda = 1 solid, 0 < lambda < 1 interface**.
  This project uses that convention internally (never the README wording).

### Dataset-derived geometry facts

* Uz output channel is exactly 0 in all samples (pseudo-2D).
* Block-majority 8x8 bitmaps (each bitmap cell = 8x8 field cells of 0.016 m):
  112/550 samples are exactly block-uniform; bitmap statistics over all 550
  samples show the left/right edges are always open at bitmap rows 1 and 6
  (row 0 = top) and open elsewhere only ~20-50% of the time
  => **ports are located at bitmap rows 1 and 6**
  (y in [0.096, 0.112] and [0.016, 0.032], symmetric about y = 0.064).
* Bitmaps are NOT restricted to inlet-to-outlet channel networks; porosity
  ranges 0.375-0.594.
* Selected first topology: `mixer_64.npz`, sample index 1 (block-majority
  bitmap, single connected fluid component incl. all four ports, porosity
  0.531, narrow passages) -> `cases/four_port_64/constant/topology8x8.csv`.

## 2. openHFDIB-DEM (HFDIB physics reference)

* URL: https://github.com/techMathGroup/openHFDIB-DEM
* Commit inspected: `863f4c268147a98589e2567ac7147c969897fc06`
  (2026-07-24, "fix: drag switch for lambda to correct rotation")
* Targets OpenFOAM v8 (not the v14 used here) -> reimplementation, not port.

### Files consulted

`README.md`, `LICENSE`, `src/HFDIBDEM/openHFDIBDEM.{H,C}`,
`src/HFDIBDEM/ibInterpolation/{interpolationInfo.H, lineInt/lineInt.{H,C},
lineInt/lineIntInfo.{H,C}, lineInt/intPoint.H}`,
`src/HFDIBDEM/geomModels/stlBased/nonConvexBody.C`,
`src/HFDIBDEM/geomModels/geomModel.C`.

### License ambiguity

README states GPLv3+; the bundled LICENSE file and source headers state
LGPLv3. This is treated as an unresolved ambiguity. **No source code was
copied**; the static-body subset of the algorithm was reimplemented from the
mathematical description below.

### Reimplemented algorithm facts (static-body subset)

* Body field (`createImmersedBody`-path in `nonConvexBody.C`): per cell, a
  vertex-fraction/center-inside test classifies the cell; interior cells
  become solid, outside fluid. Surface cells get a smeared value
  (their "body", equivalent to our lambda):
  `cBody = 0.5*(±tanh(intSpan_*signedDist / V_cell^(1/3)) + 1)`,
  sign negative when the cell center is outside the solid
  => with sigma>0 in fluid: **lambda = 0.5*(1 - tanh(intSpan*sigma/V^(1/3)))**,
  and lambda -> 0 (fluid) / 1 (solid) far from the surface; `body` is
  clipped to [0,1]. This matches the paper formula used in this project
  (intSpan = 1 here).
* `openHFDIBDEM::interpolateIB` builds an imposed field `Vs` and calls
  `ibInterp_->ibInterpolate(...)` per body.
* `lineIntInfo`: for each surface cell, stores the immersed-boundary point
  and normal, then marches fluid-side along the normal cell-to-cell
  (`getFaceInDir` walks across faces in the direction of the outward normal)
  to collect up to `ORDER` interpolation points (intPoint = position + cell
  index + processor).
* `lineInt::correctVelocity` (line 65-134) — order hierarchy for a surface
  cell at distance `ds = |x_cell - x_ib|` from the surface, wall value
  `u_ibPoint` (zero for stationary bodies):
  * order 0: `u = u_ibPoint`
  * order 1: `u = linCoeff*ds + u_ibPoint`,
    `linCoeff = (u1 - u_ibPoint)/deltaR1`
  * order 2: `u = quadCoeff*ds^2 + linCoeff*ds + u_ibPoint` with
    `quadCoeff = ((VP2 - VP1)*deltaR1 - VP1*deltaR2) / (deltaR1*deltaR2*(deltaR1+deltaR2))`,
    `linCoeff = ((VP1 - VP2)*deltaR1^2 + 2*VP1*deltaR1*deltaR2 + VP1*deltaR2^2) / (deltaR1*deltaR2*(deltaR1+deltaR2))`,
    `VPi = u_i - u_ibPoint`, `deltaR1 = |x1 - x_ib|`, `deltaR2 = |x2 - x1|`.
  i.e. a quadratic through `(0, u_wall), (deltaR1, u1), (deltaR1+deltaR2, u2)`
  evaluated at `ds` — the same formula stated in the implementation brief.

### What was NOT consulted/copied

DEM, contact mechanics, virtual meshes, moving bodies, adaptive refinement,
leastSquaresInt (mentioned only), tutorials for moving particles — all out of
scope per the brief.

## Ideas reimplemented in this project

* 5-level HFDIB body field lambda = 0.5*(1 - tanh(sigma/V^(1/3))) with
  sigma > 0 in fluid (paper/openHFDIB convention).
* Static line-interpolation: surface point + fluid normal, two fluid-side
  interpolation points at ~1 and ~2 cell sizes, zero/linear/quadratic
  polynomial evaluated at the cell center's distance ds, wall value zero.
* The four-port metamaterial case geometry, recovered from
  `tpfm_unet/data/` (cell size 0.002 m, AOI 0.128 m, ports at bitmap rows 1
  and 6) — port extension channels outside the AOI are **not** in the
  dataset and are reconstructed (see case docs).
