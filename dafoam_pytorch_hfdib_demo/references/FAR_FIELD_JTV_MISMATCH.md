# Far-field JTV mismatch (bounded investigation)

Status: measured, root cause open. Time-box: 1 day, *after* the warm-start
path (decision A). If no k ≤ 20 actual SIMPLE iterate enters the validated
JTV region, this investigation starts immediately.

## Measured facts (2026-08-04, ConvergentChannel, reverse AD)

1. At the **converged primal state** (‖R‖₂ = 2.5e-7): JTV-vs-FD relative
   error is ~4e-9 on the state-direction dot test (p/U blocks also 1e-9
   class). The reverse tape is essentially exact **there**.
2. At the **OpenFOAM initial state** (‖R‖₂ = 923): JTV-vs-FD mismatch up to
   ~3e-1 relative on some seeds (Gate D parameter directions 2.4e-5..2.5e-3).
3. Along W(t) = W_conv + t·(W_init − W_conv): the error collapses to ~2e-6
   for t ≤ 0.3 (‖R‖ < ~380); the bad zone is close to the raw initial state,
   not a linear function of residual norm.
4. Per-entry full FD gradient at warm iterates W_8/W_20: U/p/T block entries
   clean (≤ 6e-5 at k=8, ≤ 2e-6 at k=20); a **state-dependent subset of
   `phi` (face-flux) state entries** is wrong (33 entries ≤ 42% rel at k=8;
   62 entries ≤ 0.17 rel at k=20). Median everywhere ≤ 1e-7.
5. Adam free-state training from the cold start oscillates and plateaus
   (~1.5× over 1500 steps), consistent with the wrong gradient subspace.
6. Masking phi rows + freezing phi states from a warm start *seemed* clean
   per-entry, but training then descends only ~3.9× (masked loss floor at
   R≈271): freezing phi breaks the U/p↔phi physical manifold, reintroducing
   inconsistency. Full-state (unfrozen) warm training descends cleanly.

## Hypotheses (ranked)

H1. **Intermediate-variable staleness on the normal (non-AD) solver copy.**
    `setStates` updates both `solver` and `solverAD` field values, but
    derived quantities (wall functions, HbyA pieces, per-state correction
    terms) may only refresh on the AD copy under
    `calcPrimalResidualStatistics` while the FD forward path
    (`solver.getResiduals()`) uses different refresh semantics. Main
    suspect: `DAField`/`DAInput`/`updateStateBoundaryConditions`.

H2. **Hand-coded partial derivatives for `phi` coupling.**
    DAFoam computes dRdW with a mix of taped AD and hand partials
    (`DAPartDeriv`). A subset of face entries suggests off-diagonal
    convection/`fvm::div(phi_, U_)` couplings are approximated/neglected at
    faces where local flux imbalance is large (consistent with the
    state-dependent subset in fact 4).

H3. **Equation relaxation coupling.**
    `UEqn.relax()` / `TEqn.relax()` are derivative-invariant *at* the current
    state only in exact arithmetic for some assembled pieces; near-solution
    this vanishes (matches fact 1 vs 2).

## Candidate files to read first

* `src/adjoint/DAResidual/DAResidualSimpleFoam.C` (`phiRes_ = phiHbyA - pEqn.flux() - phi_`)
* `src/adjoint/DAPartDeriv/` (phi/dphi coupling terms)
* `src/adjoint/DAField/`, `src/adjoint/DAInput/`
* `solver.updateStateBoundaryConditions` usage sites

## Non-goals

Not fixing cold-start Adam behavior by tuning; not designing loss masks
around the defect; not touching the converged-state JTV (it is exact).
