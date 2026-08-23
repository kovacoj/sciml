import numpy as np

from src.brinkman_topologies import (
    PORT_INTERVALS, BrinkmanTopologyGeometry, lambda_values, signed_distance,
    topology,
)


def test_analytic_brinkman_topologies_are_finite_and_ports_are_fluid():
    for name in "ABC":
        geometry = BrinkmanTopologyGeometry(name)
        assert geometry.lambda_field.shape == (64, 64)
        assert np.all(np.isfinite(geometry.lambda_field))
        assert np.all(np.isfinite(geometry.signed_distance))
        assert np.all((geometry.lambda_field >= 0) & (geometry.lambda_field <= 1))
        for lower, upper in PORT_INTERVALS:
            y = 0.5 * (lower + upper)
            assert geometry.interpolate([[0.0, y]], "lambda")[0] < 0.02
            assert geometry.interpolate([[0.128, y]], "lambda")[0] < 0.02


def test_signed_distance_and_lambda_conventions():
    value = topology("A")
    points = np.array([[0.03, 0.024], [0.03, 0.064]])
    distance = signed_distance(points, value)
    assert distance[0] < 0 and distance[1] > 0
    lam = lambda_values(points, value)
    assert lam[0] < 0.5 and lam[1] > 0.5
