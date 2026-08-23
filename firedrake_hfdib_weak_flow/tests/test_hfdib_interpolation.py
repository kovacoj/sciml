import numpy as np
import torch

from src.geometry import TPFMGeometry
from src.hfdib import HFDIB, first_order, second_order
from src.model import CoordinateMLP


DTYPE = torch.float64


def test_first_order_reproduces_manufactured_linear_field():
    x = torch.tensor([-0.003, 0.0005, 0.004], dtype=DTYPE)
    d1 = torch.tensor([0.002, 0.003, 0.006], dtype=DTYPE)
    polynomial = lambda coordinate: 1.25 - 3.5 * coordinate

    result = first_order(polynomial(0.0), polynomial(d1), x, d1)

    assert torch.max(torch.abs(result - polynomial(x))).item() < 1.0e-10


def test_second_order_reproduces_manufactured_quadratic_vector_field():
    x = torch.tensor([-0.002, 0.001, 0.004], dtype=DTYPE)
    d1 = torch.tensor([0.002, 0.003, 0.006], dtype=DTYPE)
    d2 = torch.tensor([0.003, 0.004, 0.005], dtype=DTYPE)

    def polynomial(coordinate):
        coordinate = torch.as_tensor(coordinate, dtype=DTYPE)
        return torch.stack(
            (2.0 + coordinate - 4.0 * coordinate**2,
             -0.5 + 3.0 * coordinate + 2.5 * coordinate**2),
            dim=-1,
        )

    result = second_order(
        polynomial(0.0), polynomial(d1), polynomial(d1 + d2), x, d1, d2
    )

    assert torch.max(torch.abs(result - polynomial(x))).item() < 1.0e-10


def test_geometry_reconstructs_interface_and_searches_outward(tmp_path):
    spacing = 0.01
    axis = (np.arange(20) + 0.5) * spacing
    xx, _ = np.meshgrid(axis, axis)
    exact_sigma = xx - 0.1
    lam = 0.5 * (1.0 - np.tanh(exact_sigma / spacing))
    dataset = tmp_path / "mixer_64.npz"
    np.savez(dataset, inputs=lam[None, None])
    geometry = TPFMGeometry(dataset, spacing=spacing)

    assert geometry.reconstruction_error["max_abs"] < 1.0e-12
    query = np.array([[0.095, 0.075], [0.105, 0.125]])
    assert np.max(
        np.abs(geometry.interpolate(query) - (query[:, 0] - 0.1))
    ) < 1.0e-12

    boundary = torch.tensor([[0.1, 0.075], [0.1, 0.125]], dtype=DTYPE)
    normals = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    points, distances = HFDIB(geometry).outward_points(
        boundary, [0.01, 0.02], normals=normals
    )
    assert points.shape == (2, 2, 2)
    assert torch.allclose(points[..., 0], 0.1 + distances)
    assert np.all(geometry.interpolate(points.numpy(), "signed_distance") > 0.0)


def test_coordinate_mlp_defaults_are_float64_and_near_zero():
    model = CoordinateMLP()
    coordinates = torch.zeros((8, 2), dtype=DTYPE)

    output = model(coordinates)

    assert output.dtype == DTYPE
    assert output.shape == (8, 3)
    assert torch.max(torch.abs(output)).item() < 1.0e-3
