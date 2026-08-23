import numpy as np
from scipy.sparse import csr_matrix

from src.stokes_loss_pilot import Loss


def test_fixed_richardson_transpose_identity_on_small_spd_blocks():
    matrix = csr_matrix(np.array([
        [4.0, 1.0, 0.2],
        [1.0, 3.0, 0.1],
        [0.2, 0.1, 2.0],
    ]))
    norm = csr_matrix(np.diag([2.0, 2.0, 1.0]))
    loss = Loss(
        "richardson_5", matrix, np.array([1.0, -0.5, 0.25]), norm,
        n_velocity=1, richardson_omega=0.5,
    )
    left = np.array([0.3, -0.2, 0.4])
    right = np.array([-0.1, 0.7, 0.2])
    np.testing.assert_allclose(
        left @ loss._richardson_apply(right),
        loss._richardson_transpose_apply(left) @ right,
        rtol=1e-12,
        atol=1e-12,
    )
