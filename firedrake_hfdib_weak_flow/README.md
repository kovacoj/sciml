# Firedrake HFDIB weak-flow scaffold

This directory contains the lightweight, Firedrake-independent core for a
weak-flow HFDIB experiment on the TPFM four-port geometries. It currently
provides:

- reconstruction of signed distance and normals from `mixer_64.npz`;
- interpolation of reconstructed geometry fields at arbitrary coordinates;
- first- and second-order HFDIB interpolation in float64 PyTorch, with samples
  at normal coordinates `0`, `d1`, and `d1 + d2`;
- outward normal search for fluid interpolation points; and
- a coordinate MLP for `(u_x, u_y, p)`.

There is deliberately no mock or partial Firedrake bridge. A variational
operator should be added only in an environment with Firedrake available.

## Data

The JSON configurations point to the sibling reference checkout by default:

```text
../tpfm_unet_reference/data/mixer_64.npz
```

`TPFMGeometry` reads only the archive's `inputs` array. Override the path when
constructing it if the dataset is elsewhere:

```python
from firedrake_hfdib_weak_flow.src import TPFMGeometry

geometry = TPFMGeometry("/path/to/mixer_64.npz", sample_index=0)
sigma = geometry.interpolate([[0.04, 0.06]], "signed_distance")
```

The input convention is `lambda > 0.5` on the solid side. At diffuse-interface
cells, signed distance is recovered from
`sigma = h * atanh(1 - 2*lambda)`; elsewhere it comes from Euclidean distance
transforms. Normals point from solid to fluid.

## Lightweight tests

From this directory, with NumPy, SciPy, PyTorch, and pytest installed:

```bash
python -m pytest -q
```

The smoke configuration uses a 12 by 12 mesh. Topology configurations use a
20 by 20 mesh and the default physical values `uin=0.1`, `pout=0`, and
`nu=0.01`.
