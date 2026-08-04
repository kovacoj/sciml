# Architecture

    OpenFOAM mesh/case  ──┐
                          ├─>  DAFoam residual R(W)          (C++ reverse-AD)
    W(θ)  (PyTorch)  ────┘         │
                                   │  (∂R/∂W)ᵀ v  via
                                   │  DASolver.solverAD.calcJacTVecProduct
                                   ▼
                       PyTorch backward (custom autograd.Function)
                                   │
                            optimizer.step(θ)

* DAFoam owns discretization, boundary conditions, physics models and the
  reverse-AD tape. PyTorch owns parameterization and optimization.
* One `PYDAFOAM` instance is kept alive for the whole training run; states
  are exchanged as NumPy arrays via `getStates`/`setStates`; there is no
  file conversion, no OpenFOAM subprocess per step, no primal solve in the
  loop.
* Loss = ½‖R(W)‖² (optionally D²-diagonally weighted later once the state
  layout is decoded). Gradient = (∂R/∂W)ᵀ(seed), seed = R (or D²R).
* HFDIB (Gate G+) becomes a DAFoam `DAFvSource` extension
  (`DAFvSourceHFDIBStatic`, see `dafoam_extension/`) so the immersed forcing
  lives inside the AD tape. Brinkman/porosity (`alphaPorosity`) is NOT used.

Division of responsibility sentence to reuse:

> DAFoam/OpenFOAM supplies the discretization and reverse-mode residual
> derivative; PyTorch evaluates the prediction-dependent boundary of the
> problem (state parameterization) and differentiates through (∂R/∂W)ᵀ.
