import numpy as np

from src.stokes_loss_pilot import Loss, build_system, directional_error


def test_all_stokes_pilot_loss_gradients_and_exact_root():
    operator, residual0, norm, _, _, _ = build_system(4, 2)
    exact = np.linalg.solve(operator.toarray(), -residual0)
    for name in ("raw", "dual", "correction"):
        loss = Loss(name, operator, residual0, norm)
        assert directional_error(loss, np.zeros(operator.shape[1])) < 1.0e-6
        assert loss(exact)[0] < 1.0e-20
