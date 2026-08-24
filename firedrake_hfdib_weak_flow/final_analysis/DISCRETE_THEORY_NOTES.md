# Discrete Correction-Loss Equivalence

Let the gauge-fixed finite-dimensional Stokes system be

\[
A_h U_h=b_h,
\]

with invertible constrained matrix `A_h`. For arbitrary coefficients `U`, define
the PDE correction by

\[
A_h\delta(U)=b_h-A_hU.
\]

Subtracting the discrete solution equation gives

\[
A_h[\delta(U)-(U_h-U)]=0.
\]

Invertibility therefore implies

\[
\boxed{\delta(U)=U_h-U}.
\]

For an SPD norm matrix `M_X`,

\[
J_{corr}(U)=\delta(U)^TM_X\delta(U)
           =(U-U_h)^TM_X(U-U_h).
\]

Thus exact correction training equals discrete FE-error-oracle training in the
chosen norm, without using FE solution labels in the objective. Its Hessian is
`2 M_X`. Raw residual least squares,

\[
J_{raw}(U)=(A_hU-b_h)^T(A_hU-b_h),
\]

instead has Hessian `2 A_h^T A_h`, which squares the operator conditioning.

This proposition is finite-dimensional and discrete. It is not asserted as a
general infinite-dimensional or nonlinear Navier-Stokes theorem.
