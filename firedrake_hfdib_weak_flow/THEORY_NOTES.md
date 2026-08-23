# Operator-Preconditioned Weak Neural Stokes

## Discrete setting

Let `X_h = V_h x Q_h` be a stable, finite-dimensional Taylor-Hood space after
essential boundary conditions and pressure-gauge handling. Let

\[
A_h : X_h \to X_h^*, \qquad A_h U_h=b_h
\]

denote the invertible discrete linear Stokes operator.

## Proposition: exact correction equals discrete error

For arbitrary `U in X_h`, define the correction `delta(U)` by

\[
A_h\delta(U)=b_h-A_hU.
\]

Then

\[
A_h\delta(U)=A_h(U_h-U),
\]

and invertibility gives

\[
\boxed{\delta(U)=U_h-U.}
\]

For any SPD discrete norm operator `M_X`,

\[
J_{corr}(U)=\delta(U)^T M_X\delta(U)
=(U-U_h)^T M_X(U-U_h).
\]

Consequently, the unique minimizer is `U_h` and

\[
\nabla_U J_{corr}=2M_X(U-U_h).
\]

This is a finite-dimensional statement. It assumes a stable mixed pair,
well-posed boundary conditions, and an invertible gauge-fixed operator. It is
not asserted as an unrestricted infinite-dimensional theorem.

## Contrast with raw least squares

For residual `r(U)=A_hU-b_h`,

\[
J_{raw}=r^Tr,
\qquad
\nabla J_{raw}=2A_h^Tr,
\qquad
\nabla^2J_{raw}=2A_h^TA_h.
\]

Raw residual optimization therefore inherits squared singular-value
conditioning. Exact correction has Hessian `2M_X`, removing PDE conditioning
in coefficient space.

## Fixed linear approximate inverse

For fixed linear `B_k : X_h^* -> X_h`, let

\[
\delta_k=B_kr, \qquad J_k=\delta_k^TM_X\delta_k.
\]

Then

\[
\nabla_UJ_k=2A_h^TB_k^TM_XB_kr.
\]

Every approximate method must implement both `B_k` and its transpose action,
then pass a directional finite-difference gradient check. A truncated GMRES map
is generally residual-dependent through its Krylov basis and is not treated as
a fixed matrix without differentiating that algorithm.

## Pilot evidence

At 64x32 and 300 coefficient L-BFGS iterations, velocity errors are:

| metric | velocity error | wall time |
|---|---:|---:|
| raw | 0.978 | 2.37 s |
| dual | 0.224 | 7.49 s |
| Jacobi normal-equation correction | 0.998 | 4.63 s |
| velocity/pressure block correction | 0.189 | 12.32 s |
| exact correction | 0.00607 | 10.16 s |
| oracle error | 0.00523 | 1.84 s |

Exact correction tracks oracle closely, while simple approximate inverses do
not retain enough benefit under refinement. The gap between oracle cost and
exact-correction cost motivates fixed Richardson, multigrid, and better Schur
approximations.
For the fixed block-Richardson map used in the next campaign,

\[
\delta_{j+1}=\delta_j+\omega P^{-1}(r-A\delta_j),\qquad \delta_0=0.
\]

With fixed `k`, `omega`, and linear `P^-1`, this defines a fixed linear map
`B_k`. Its transpose is applied by the analogous recurrence with `A^T` and
`P^-T`; this is used in the loss gradient rather than treating an inexact solve
as an exact inverse.
