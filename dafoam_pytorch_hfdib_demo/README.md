# DAFoam + PyTorch HFDIB Demo (migration)

A **hybrid DAFoam–neural solver/accelerator**: a few-step DAFoam
continuation followed by label-free PyTorch optimization that drives the
DAFoam residual to zero through the reverse JTV:

    Python PyTorch  ->  DAFoam residual  ->  (∂R/∂W)ᵀ v  ->  PyTorch backward

DAFoam (v5.0.0, precompiled image) supplies the CFD discretization and the
reverse-mode AD Jacobian-transpose-vector product; PyTorch owns
optimization. No CFD velocity/pressure labels are used for training;
DAFoam performs only a small, documented SIMPLE-iteration warm start.

**Cold-start DAFoam JTV is not globally reliable for this case.** A
documented partial-primal continuation is used to enter an empirically
verified JTV trust region before PyTorch optimization. Full honest status,
including cold-start failure, lives in `STATUS.md`.

## Status

* See `STATUS.md` for the gate matrix (A–C/D0 pass; D1/E0 cold-start fail
  recorded; E1 in progress).
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
