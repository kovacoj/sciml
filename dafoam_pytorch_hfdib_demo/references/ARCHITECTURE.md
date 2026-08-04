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

    W0 ──(k SIMPLE iterations, DAFoam primal)──▶ Wk ──(PyTorch/JTV)──▶ Wθ

* DAFoam owns discretization, boundary conditions, physics models and the
  reverse-AD tape. PyTorch owns parameterization and optimization.
* One `PYDAFOAM` instance is kept alive for the whole training run; states
  are exchanged as NumPy arrays via `getStates`/`setStates`; there is no
  file conversion, no OpenFOAM subprocess per step, no primal solve inside
  the loop (the primal runs only as the documented k-iteration warm start).
* Loss = ½‖R(W)‖². Gradient = (∂R/∂W)ᵀ·R via reverse JTV.
* The JTV is validated in a trust region around actual partial-primal
  iterates (warm-start sweep); training uses a forward-loss step-rejection
  policy (`python/train_free_state.py`) and may not rely on the JTV in the
  cold-start far field (measured not reliable there).
* HFDIB (Gate G+) becomes a DAFoam `DAFvSource` extension
  (`DAFvSourceHFDIBStatic`, see `dafoam_extension/`) so the immersed forcing
  lives inside the AD tape. Its warm start must run the SAME HFDIB residual
  as is differentiated (not plain DASimpleFoam).
* Brinkman/porosity (`alphaPorosity`) is NOT HFDIB and is not used.

Division of responsibility sentence to reuse:

> DAFoam/OpenFOAM supplies the discretization and reverse-mode residual
> derivative; PyTorch evaluates the prediction-dependent boundary of the
> problem (state parameterization) and differentiates through (∂R/∂W)ᵀ.
