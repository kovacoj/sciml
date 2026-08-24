# HFDIB Active Path Audit

## Result

```
HFDIB_IMPLEMENTATION = GENERIC_FIELD (hfdibSignedDistance)
BRINKMAN_USED = NO
ARBITRARY_TOPOLOGY_3D_READY = YES (signed-distance field reader, dimension-agnostic)
AD_TAPED = YES (useAD: reverse)
```

## Evidence

### Production topology pipeline

The production four-port 64×64 topology pipeline (used by all 256+16 training/test cases) uses:

```python
# python/common.py line 110-117
"fvSource": {
    "obstacle": {
        "type": "hfdibSignedDistance",
        "geometryFile": "hfdibGeometry/signedDistance",
        "solidSign": -1,
        "d1Factor": 1.5,
    }
}
```

This is NOT `hfdibStaticRect`. It reads an arbitrary signed-distance field from
a file (`constant/hfdibGeometry/signedDistance`), which is written per-topology
by `python/unet/generate_case.py:write_signed_distance_file()`.

### Static rect is legacy only

`hfdibStaticRect` (single axis-aligned rectangle with `bounds: [x0,y0,z0,x1,y1,z1]`)
is defined in `common.py:HFDIB_OBSTACLE` but is used only by:

- `single_obstacle` case
- Gate G experiments

It is NOT used by any production topology case.

### Brinkman

No `alphaPorosity` or Brinkman source is configured anywhere in the production
pipeline. The HFDIB forcing is:

    f_ib = chi * [M_h(U_ib) + grad(p)]

with `chi = ceil(lambda)`.

### 3D readiness

The `hfdibSignedDistance` source reads a scalar field from a file. The field
format is a standard OpenFOAM scalar list with cell-count entries. The C++
implementation processes all cells in the mesh, including 3D cells.

The current 64×64×1 mesh is quasi-2D because:
- Only 1 cell in z
- frontAndBack patches are symmetry

But the HFDIB source itself is dimension-agnostic: it operates on the mesh's
cell list, computes lambda/sigma/normals from the signed-distance field, and
applies forcing per-cell.

### Geometry manifest

`hfdib/geometry_manifest.py` reads per-cell geometry (cx, cy, cz, lambda, chi,
sigma, normals) written by the C++ source. It already has 3D fields
(surface_z, normal_z, cz).

### Conclusion

The existing `hfdibSignedDistance` fvSource already supports arbitrary signed-
distance fields on any mesh dimensionality. To run a genuine 3D case:

1. Create a 3D mesh (e.g. 64×64×8)
2. Write a 3D signed-distance field
3. Change front/back boundaries from symmetry to wall
4. Run with the same `hfdibSignedDistance` configuration

No C++ modification is needed for the extruded topology experiment.
