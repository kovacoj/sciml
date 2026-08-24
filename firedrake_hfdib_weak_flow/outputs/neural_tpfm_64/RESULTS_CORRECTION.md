# Neural TPFM 64x64 Results — DIAGNOSTIC ONLY

## Status: OPTIMIZER_COMPLETED_PHYSICAL_GATE_FAILED

The three N64 runs (A/B/C) completed 400 optimizer steps each but **failed the
physical mass-balance gate**. Mass imbalance is approximately 1.0 (100% flux
defect) with outlet flux near zero.

## Formulation issue

The N64 configs do not specify `residual_formulation`, so `benchmark_setup.py`
defaults to `literal_strong_hfdib`. These runs therefore used the **literal
strong HFDIB residual**, not the first-derivative weak Navier-Stokes
formulation described in the presentation.

The `h1_weak` formulation is currently restricted to manufactured empty
channel geometry and cannot be used directly on TPFM topology fields.

## Results (diagnostic)

| Case | Topology | Initial loss | Final loss | Mass imbalance | Outlet flux |
|---|---:|---:|---:|---:|---:|
| N64-A | 508 | 1.943 | 0.270 | 1.000 | ~0 |
| N64-B | 431 | 1.223 | 0.253 | 0.999 | ~0 |
| N64-C | 34 | 4.939 | 0.234 | 1.000 | ~0 |

## Safe statement

Three topology-conditioned optimization runs reduced their residual objectives,
but failed the global mass-balance gate and are retained as diagnostics only.
The remaining defensible neural Firedrake result is the validated empty-channel
weak Stokes MLP: velocity error 1.36%, mass imbalance 1.20%.
