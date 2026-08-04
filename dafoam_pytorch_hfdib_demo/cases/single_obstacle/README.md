# single_obstacle case

Thin 3D isothermal channel (`channel_isothermal` geometry: 40x16x1 duct,
1 cell thick, physical walls incl. front/back, no empty patches) plus ONE
stationary rectangular solid immersed via the static HFDIB fvSource
(`dafoam_extension/`).

## Operator wiring

The obstacle is NOT in the OpenFOAM dictionaries. It lives in daOptions
(`python/common.py::hfdib_options`), which carries:

```python
"fvSource": {
    "obstacle": {
        "type": "hfdibStaticRect",
        "bounds": [0.45, 0.03, 0.0, 0.55, 0.07, 0.005],
        "d1Factor": 1.5,
        "sourceField": "U",                       # informational
    },
}
```

The rectangle spans y in [0.03, 0.07] (i.e. fills much of the 0.1 m duct
height), x in [0.45, 0.55], all of the 0.005 m slab depth — a first-order
stationary no-slip obstacle inside the laminar duct.

## Flow settings

* nu = 1.0e-3 m^2/s, U_inlet = 0.2 m/s, p_outlet = 0, dummy-laminar path
* states [U, p, phi] = 5176 (same as channel_isothermal)

All files copied from `channel_isothermal`; the isothermal probe report for
the geometry is `cases/channel_isothermal/isothermal_probe.json`.
