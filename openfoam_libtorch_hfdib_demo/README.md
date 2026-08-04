# OpenFOAM + LibTorch Shallow HFDIB Physics Demo

A **label-free shallow neural finite-volume flow solver trained from
differentiable Stokes/Navier–Stokes and HFDIB constraints**, loosely
following the mechanical-metamaterial flow problem of Ledl, Kubíčková,
and Isoz (TPFM 2026, DOI:10.14311/TPFM.2026.019).

**Division of responsibility:** OpenFOAM supplies the mesh, geometry,
boundary metadata, finite-volume discretization specification, and
immersed-boundary geometry. LibTorch evaluates the prediction-dependent
discrete residual and differentiates it with respect to the network
parameters. We do **not** differentiate through OpenFOAM, and no velocity
or pressure labels are used during training.

This project is a minimal demonstration, not an exact reproduction of the
reference article (the authors' complete OpenFOAM/HFDIB case files are not
public). See `references/UPSTREAM.md` for exactly what was consulted and
what was reimplemented.

Layout mirrors the working reference eigensolver project
`../openfoam_libtorch_lho` (TorchCompat, build conventions, LibTorch path
handling, OpenFOAM v14 environment).
