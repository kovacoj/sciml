"""Shared first-derivative Navier-Stokes weak forms."""

from firedrake import div, dot, grad, inner, sym


def momentum_weak_action(u, p, v, nu, beta, measure):
    """Momentum residual acting on a vector test function."""
    return (
        2.0 * nu * inner(sym(grad(u)), sym(grad(v)))
        + beta * dot(dot(grad(u), u), v)
        - p * div(v)
    ) * measure


def continuity_weak_action(u, q, measure):
    """Continuity residual acting on a scalar test function."""
    return q * div(u) * measure


def navier_stokes_weak_form(u, p, v, q, nu, beta, measure):
    """Steady incompressible Navier-Stokes weak residual."""
    return (
        momentum_weak_action(u, p, v, nu, beta, measure)
        + continuity_weak_action(u, q, measure)
    )


def navier_stokes_brinkman_weak_form(
    u, p, v, q, nu, beta, alpha, lam, measure
):
    """Variational Navier-Stokes-Brinkman residual."""
    return (
        navier_stokes_weak_form(u, p, v, q, nu, beta, measure)
        + alpha * lam * dot(u, v) * measure
    )
