import numpy as np

from src.stokes_loss_pilot import Loss, build_system, directional_error


def test_all_stokes_pilot_loss_gradients_and_exact_root():
    operator, residual0, norm, _, n_velocity, _ = build_system(4, 2)
    exact = np.linalg.solve(operator.toarray(), -residual0)
    for name in ("raw", "dual", "jacobi_ls", "block", "correction", "oracle"):
        loss = Loss(
            name, operator, residual0, norm,
            n_velocity=n_velocity, exact=exact,
        )
        assert directional_error(loss, np.zeros(operator.shape[1])) < 1.0e-6
        assert loss(exact)[0] < 1.0e-20


def test_exact_correction_equals_oracle_loss_and_gradient():
    operator, residual0, norm, _, n_velocity, _ = build_system(4, 2)
    exact = np.linalg.solve(operator.toarray(), -residual0)
    values = np.linspace(-0.01, 0.02, operator.shape[1])
    correction = Loss(
        "correction", operator, residual0, norm,
        n_velocity=n_velocity, exact=exact,
    )
    oracle = Loss(
        "oracle", operator, residual0, norm,
        n_velocity=n_velocity, exact=exact,
    )
    correction_value, correction_gradient = correction(values)
    oracle_value, oracle_gradient = oracle(values)
    np.testing.assert_allclose(correction_value, oracle_value, rtol=1e-10)
    np.testing.assert_allclose(correction_gradient, oracle_gradient, rtol=1e-10)
