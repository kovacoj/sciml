# Firedrake Supervisor Notes

## Claim hierarchy

### Article reproduction

**Blocked.** The published 64 by 64 arrays are an internal observation window,
not the complete CFD domain. The controlled DAFoam reconstruction failed its
predeclared reproduction gates and remains classified `TPFM_TOPOLOGIES_ONLY`.

### Cropped Firedrake diagnostic

**Not a valid article BVP.** The artificial full-side inlet overlaps 16 of 41
solid/interface velocity DOFs. Increasing continuity weight improves mass only
by sacrificing momentum, exposing incompatible boundary conditions.

### Manufactured weak-residual validation

**Validated in coefficient space.** Direct and residual-minimization paths share
one first-derivative Taylor-Hood weak form:

\[
2\nu(\varepsilon(u),\varepsilon(v))
+\beta((u\cdot\nabla)u,v)
-(p,\nabla\cdot v)
+(q,\nabla\cdot u).
\]

For test-basis residual vector `r` and test Gram matrix `G`, training minimizes

\[
\mathcal L=r^T G^{-1}r.
\]

The direct FE field evaluates to loss `1.07e-30`. Independent optimization of
the FE coefficients reaches loss `5.40e-10`, velocity-x error `3.86e-5`, and
mass imbalance `2.39e-5`. This establishes that the weak dual-residual minimum
recovers the Taylor-Hood root.

### Neural restriction

Replacing arbitrary FE coefficients by a 12,995-parameter coordinate MLP gives
loss `3.33e-4`, velocity-x error `1.36%`, gauge-centered pressure error `6.26%`,
and mass imbalance `1.20%` after the frozen Adam/L-BFGS budget. The MLP closely
approximates the reference but misses the predeclared 1% velocity and `1e-4`
mass gates, so circular HFDIB and controlled TPFM neural runs were not launched.

The neural and direct divergence L2 values (`0.03150` and `0.03208`) should be
described as the same characteristic weak-incompressibility level of the
Taylor-Hood discretization. Global mass defect is the remaining discrepancy.

## Poisson demonstration

The compact `poisson_demo.py` assembles `A` and `b` with Firedrake, evaluates a
coordinate MLP at FE nodes, and minimizes

```python
residual = A @ U_theta - b
loss = residual @ solve(G, residual)
```

No direct solution enters training. At the frozen optimizer cap it reaches weak
loss `1.08e-5` and coefficient relative error `2.10e-3`.

## TPFM topology audit

Across all 550 official lambda fields:

- every sample has one connected fluid component and a left-to-right path;
- expected port bands are never fully closed;
- rows `[10,14)` and `[50,54)` are invariant fluid on both ROI edges;
- these correspond to `[0.020,0.028)` m and `[0.100,0.108)` m.

This identifies defensible attachment locations for a future controlled
TPFM-derived benchmark, but does not recover the article CFD domain.

## Slide sequence

1. Pointwise strong residual versus FE test-function residual.
2. Derive `r_i = F(u_theta; v_i)` and `r^T G^-1 r`.
3. Show the three-panel Poisson demonstration.
4. Explain why cellwise second derivatives of CG2 omitted facet jumps.
5. Show direct FE injection at `1e-30` and passing external gradients.
6. Show the Case 0 comparison table and field figure.
7. Show coefficient versus MLP optimization curves.
8. Close with the progression: Poisson, Stokes, circular HFDIB, controlled TPFM.

## Safe result statement

> Firedrake exposes finite-element test-function residuals as differentiable
> neural objectives. A shared Taylor-Hood weak dual residual has the direct FE
> Stokes solution as its numerical root, and independent coefficient-space
> minimization recovers that root to high accuracy. Restricting the coefficients
> to a coordinate MLP gives an approximate Stokes solution; its missed mass gate
> motivates further representation and optimization work before HFDIB escalation.

## Brinkman application gate

A separate analytic A/B/C channel-network family was created for a fully
variational Brinkman application. This is explicitly not the article's HFDIB
discretization. Standard mixed Taylor-Hood direct solves on segmented physical
ports are globally mass-balanced, but the 32 by 32 alpha gate failed: solid
leakage decreases only from `0.808` at alpha 100 to `0.235` at alpha `1e6`,
while divergence L2 degrades to `0.677`. A bounded 64 by 64 check gives leakage
`0.694`, `0.621`, and `0.538` at alpha 625/1250/2500; replacing diffuse lambda
by sharp chi at alpha 2500 improves this only to `0.403`. The issue therefore
combines penalty-layer resolution, diffuse-interface width, and the global
leakage metric rather than being rescued by one refinement. Neural and recycling
runs were not launched. Use these figures only as appendix/next-work material.

The bounded fitted-domain follow-up succeeds for Topology A: an exact union of
axis-aligned level-set channels is meshed conformingly, giving a direct Stokes
solve with weak residual norm `4.88e-16` and mass imbalance `2.93e-15`. This
removes Brinkman leakage entirely. A new variable-mesh neural bridge would be
required for A/B recycling; it was deferred at the implementation time gate
rather than improvised. The direct fitted field is suitable as a final
"next application" appendix figure.
