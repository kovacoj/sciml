# DAFoam + PyTorch HFDIB Demo (migration)

A **label-free shallow neural flow solver trained from the differentiable
DAFoam residual**, migrating the previous hand-rolled OpenFOAM+LibTorch FVM
experiments to:

    Python PyTorch  ->  DAFoam residual  ->  (∂R/∂W)ᵀ v  ->  PyTorch backward

DAFoam (v5.0.0, precompiled image) supplies the CFD discretization and the
reverse-mode AD Jacobian-transpose-vector product; PyTorch owns optimization.
No CFD velocity/pressure labels are used for training.

## Status

* Previous approach (hand-rolled differentiable FVM + LibTorch) preserved
  under `../openfoam_libtorch_hfdib_demo` on `feat/shallow-hfdib-physics-demo`.
* Working reference project `../openfoam_libtorch_lho` is untouched.

## Environment

Docker image `dafoam/opt-packages:v5.0.0` (pinned). See `environment/`.

```bash
./environment/run_container.sh
# inside the container:
source /home/dafoamuser/dafoam/loadDAFoam.sh
```

## Gates

A environment — B state/residual round-trip — C DAFoam JTV dot-product —
D PyTorch bridge gradient — E free-state training — F shallow decoder —
G static HFDIB extension — H four-port metamaterial article case.

See `references/ARCHITECTURE.md` and `references/DAFOAM_API_NOTES.md`.
