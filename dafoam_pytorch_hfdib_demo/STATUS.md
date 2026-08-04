# STATUS

Updated 2026-08-04 · branch `feat/dafoam-pytorch-hfdib-demo`

## Gate matrix

| Gate | Description | Status |
|------|-------------|--------|
| A | environment | **Pass** — pinned `dafoam/opt-packages:v5.0.0`, torch-in-venv, imports verified |
| B | state/residual plumbing | **Pass** — getStates/setStates/getResiduals, residual round-trip bit-identical |
| C | global JTV smoke test | **Pass** (accepted band 1e-4; systematic 5e-5 floor, not truncation) |
| D0 | PyTorch custom-backward plumbing | **Pass** — autograd.Function forward/backward wired; np path == torch path to 2e-16 |
| D1 | cold-start gradient accuracy | **Fail** — direction-dependent 2.4e-5..2.5e-3 at the raw initial state |
| D2 | warm-start gradient accuracy | pending (test harness ready: `tests/test_torch_autograd.py` at chosen W_k) |
| E0 | cold-start free-state optimization | **Fail** — Adam oscillates, ~1.5× in 500–1500 steps |
| E1 | warm-start free-state optimization | in progress — warm descent healthy (see below), 100× bar not yet crossed |
| F | shallow decoder | blocked by E1 |
| G | static HFDIB | blocked |
| H | four-port article case | blocked |

## Certified state layout (DASimpleFoam + ConvergentChannel)

`adjStateOrdering = "state"`, `getNLocalAdjointStates() = 2891`:

```
[ U : 1029 ]  cell-major (Ux,Uy,Uz) x 343 cells
[ p : 343 ]
[ T : 343 ]
[ phi : 1176 ]  (all faces incl. boundary)
```

No `nuTilda` model state (dummy turbulence). Provenance + verification:
`python/state_layout.py`, `tests/test_state_layout.py` (PASS),
`outputs/state_layout/state_layout_summary.json`.

## Continuation method (chosen approach)

    W0 --(k SIMPLE iterations)--> Wk --(PyTorch/JTV optimization)--> W_final

* Warm start `k=8` (of ~112 to convergence ≈ 7% of iterations). No labels,
  no converged fields in the loss; DAFoam performs only the small
  initialization.
* This is a **hybrid DAFoam–neural solver/accelerator**, not yet a pure
  one-shot neural solver. Cold-start far-field JTV inaccuracy is a recorded
  result (see `references/FAR_FIELD_JTV_MISMATCH.md`), not hidden.

## Warm-start experiment results (channel, IPv4~0.5s per primal iter)

* Warm-started full-state Adam (lr 1e-3→1.3e-4 via rejection policy,
  no clip, 1500 steps): L_warm 1.4487e5 → 2.077e3 = **68.7×**, monotone,
  3 initial rejections + trust ceiling held (policy works).
* + LBFGS 100 its: → 2.094e3 (**69.2×**), grind ~0.2%/20-iter.
* + Gauss-Newton (CG on normal equations, 300 iters, Jacobi
  preconditioned): implemented; resolves conditioning but line-search
  alpha stays ~0.03 — plateau R≈71 with |g|≈1e5.
* Best warm total from cold start: 203× (L(W0)/L(final)).

**Current verdict: gate E1 (≥100× from warm) not yet met; first-order and
truncated-GN methods both slow down near R≈65 (primal-k≈19 equivalence).
Next lever candidates: stronger preconditioning, longer Adam+LBFGS budget,
or the primal-injection curriculum (hybrid continuation phases from the
revised plan section 11).**

## Reproduction commands

```bash
# environment (host)
cd dafoam_pytorch_hfdib_demo
docker build -t sciml-dafoam-torch:v5.0.0 -f environment/Dockerfile .

./scripts/prepare_case.sh                              # sanity: mesh present
./scripts/run_in_container.sh \
  "export MPLCONFIGDIR=/tmp/mplcfg && mpirun -np 1 python python/probe_dafoam.py"   # Gate B
./scripts/run_gradient_check.sh                        # Gate C
./scripts/run_in_container.sh \
  "export MPLCONFIGDIR=/tmp/mplcfg && mpirun -np 1 python tests/test_state_layout.py"
./scripts/run_partial_primal.sh                        # k sweep states
./scripts/run_in_container.sh \
  "export MPLCONFIGDIR=/tmp/mplcfg && mpirun -np 1 python python/warm_start_jtv_sweep.py"

# Gate E1 (recommended config: full state, warm k=8, Adam then GN)
./scripts/run_free_state.sh --k 8 --steps 800 --lr 1e-3 \
  --no-mask-phi --gn-iters 12 --cg-iters 300 --pc-probes 8
```
