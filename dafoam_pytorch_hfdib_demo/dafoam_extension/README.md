# DAFoam extension: static HFDIB (fvSource)

## What this is

`DAFvSourceHFDIBStaticRect` — a runtime-selectable `DAFvSource` subclass
that adds a first-order static immersed-boundary momentum source for a
single stationary, axis-aligned rectangular solid to the taped DAFoam
residual (both normal and reverse-AD builds; the same C++ source compiles
against both OpenFOAM-AD builds).

Forcing per paper formulation (stationary wall, `u_Gamma = 0`):

```
f_ib = chi * ( M_h(U_ib) + grad(p) )
chi  = ceil(lambda)
U_ib = B_lambda U   (0 in pure solid; linear profile at interface cells)
```

Geometry is precomputed once and is entirely passive (lambda, chi, signed
distance, surface points/normals, fluid-side IDW stencils); nothing
prediction-dependent leaves the AD tape. Brinkman (`alphaPorosity`)
remains unused.

Option schema (inside `daOptions` / PYDAFOAM):

```python
"fvSource": {
    "obstacle": {                                  # any name; first sub-dict
        "type": "hfdibStaticRect",
        "bounds": [x0, y0, z0, x1, y1, z1],
        "d1Factor": 1.5,                            # optional
    },
}
```

## Provenance pins

| Item | Value |
|---|---|
| Upstream image | `dafoam/opt-packages:v5.0.0` |
| Upstream source | DAFoam commit `31433b41d1d59638b459d24f02c7f89a756848c2` (`v4.0.4-2-g31433b4`) |
| fork/branch | fork at `/home/cady/chapel/dafoam_hfdib_fork`, ext commit `23cca51` |
| Patch | `patches/0001-hfdibstatic-rect-fvsource.patch` |

## Apply + build (inside `sciml-dafoam-torch:v5.0.0`)

```bash
git -C $DAFOAM_SRC apply dafoam_extension/patches/0001-hfdibstatic-rect-fvsource.patch
# then rebuild normal + ADR variants of the adjoint library:
source $HOME/dafoam/loadDAFoam.sh
( cd src/adjoint && wmake -j 6 )                       # normal
sed -i 's/export WM_AD_MODE=.*/export WM_AD_MODE=ADR/' \
    $DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-AD/etc/bashrc
. $DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-AD/etc/bashrc
( cd src/adjoint && wmake -j 6 )                       # ADR
```

The registration macro relies on each DAFvSource header including
`addToRunTimeSelectionTable.H`; the patch carries that include, matching
the upstream subclass pattern (root-caused via `#define` preprocessing:
without it the registration macro silently no-ops into a function call).

## Verification plan (user-side)

1. Ordinary HFDIB solve on `cases/single_obstacle/` — velocity ~0 at the
   rectangle surface, flow deflecting; residuals converge.
2. HFDIB JTV dot-product check on a partially converged state
   (`python/probe_hfdib_combined.py`) — decisive AD-tape inclusion test.
3. One warm-started residual-optimization run from the bridge.

All geometry parameters live inside the DAFoam source fvSource; no OpenFOAM
flow labels are used for training.
