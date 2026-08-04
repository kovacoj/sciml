# Upstream commits and pins

## DAFoam

* Image: `dafoam/opt-packages:v5.0.0` (Docker Hub, official precompiled).
* Source reference for the container-contents: recorded on GitHub at
  mdolab/dafoam. The exact tag/commit the image was built from is recorded
  below after container introspection (Gate A).

| Component   | Pin | Source |
|-------------|-----|--------|
| DAFoam      | v5.0.0 image | recorded after Gate A introspection |
| OpenFOAM    | image-bundled | `foamVersion` inside container |
| OpenFOAM-AD | image-bundled | recorded after introspection |
| CoDiPack    | image-bundled | recorded after introspection |
| PyTorch     | latest torch wheel installed into venv at image build | pip_freeze.txt |

## Article-project references (carried from the previous attempt)

* techMathGroup/tpfm_unet @ `c0566779b54877ca46053a8ff44096c3351f4805`
  (2026-07-03) — supervised U-Net reference implementation; dataset schema
  and geometry provenance.
* techMathGroup/openHFDIB-DEM @ `863f4c268147a98589e2567ac7147c969897fc06`
  (2026-07-24) — HFDIB static-body interpolation reference algorithm.
* Article: Ledl, Kubíčková, Isoz, "Using U-Net to Estimate Fluid Flow in
  Mechanical Metamaterials", TPFM 2026, DOI:10.14311/TPFM.2026.019.
