import json
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import torch

from src.forms import FiredrakeContext
from src.geometry import TPFMGeometry
from src.model import CoordinateMLP
from src.residual_bridge import FEGradients, FiredrakeResidualBridge, ResidualMetrics
from src.state import FEFieldMapper


def test_full_external_parameter_direction_finite_difference(synthetic_geometry_path):
    torch.manual_seed(12)
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(geometry, 8, 8)
    mapper = FEFieldMapper(context, geometry)
    model = CoordinateMLP(input_dim=4, width=16, depth=2)
    fields = mapper.evaluate(model)
    bridge = FiredrakeResidualBridge(context)
    bridge.initialize_normalization(fields)
    metrics, fe_gradients = bridge.evaluate_loss_and_gradients(fields, beta=0.25)
    assert isinstance(metrics, ResidualMetrics)
    assert isinstance(fe_gradients, FEGradients)
    assert [field.name for field in dataclass_fields(ResidualMetrics)] == [
        "loss", "loss_x", "loss_y", "loss_continuity", "raw_loss_x",
        "raw_loss_y", "raw_loss_continuity", "residual_x_l2",
        "residual_y_l2", "residual_continuity_l2",
    ]
    assert metrics.loss == metrics.loss_x + metrics.loss_y + metrics.loss_continuity

    model.zero_grad(set_to_none=True)
    torch.autograd.backward(
        (fields.ux, fields.uy, fields.p, fields.uibx, fields.uiby),
        tuple(torch.as_tensor(getattr(fe_gradients, name), dtype=torch.float64)
              for name in ("ux", "uy", "p", "uibx", "uiby")),
    )
    parameter_gradients = tuple(p.grad.detach().numpy().copy() for p in model.parameters())
    assert np.max(np.abs(fe_gradients.ux[context.velocity_boundary_nodes])) == 0.0
    assert np.max(np.abs(fe_gradients.uy[context.velocity_boundary_nodes])) == 0.0
    assert np.max(np.abs(fe_gradients.p[context.pressure_outlet_nodes])) == 0.0

    rng = np.random.default_rng(5)
    directions = tuple(rng.standard_normal(p.shape) for p in model.parameters())
    length = np.sqrt(sum(np.sum(direction**2) for direction in directions))
    directions = tuple(direction / length for direction in directions)
    exact = sum(np.sum(g * d) for g, d in zip(parameter_gradients, directions))
    originals = tuple(parameter.detach().clone() for parameter in model.parameters())
    errors = []
    finite_differences = []
    for eps in (1e-4, 3e-5, 1e-5):
        with torch.no_grad():
            for parameter, value, direction in zip(model.parameters(), originals, directions):
                parameter.copy_(value + eps * torch.as_tensor(direction))
        plus = bridge.evaluate_loss(mapper.evaluate(model), beta=0.25).loss
        with torch.no_grad():
            for parameter, value, direction in zip(model.parameters(), originals, directions):
                parameter.copy_(value - eps * torch.as_tensor(direction))
        minus = bridge.evaluate_loss(mapper.evaluate(model), beta=0.25).loss
        finite_difference = (plus - minus) / (2.0 * eps)
        finite_differences.append(finite_difference)
        errors.append(abs(finite_difference - exact) / max(
            abs(finite_difference), abs(exact), 1e-14
        ))
    with torch.no_grad():
        for parameter, value in zip(model.parameters(), originals):
            parameter.copy_(value)

    output = Path("outputs/gradient_check.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({
        "beta": 0.25,
        "epsilons": [1e-4, 3e-5, 1e-5],
        "exact_directional_derivative": exact,
        "finite_differences": finite_differences,
        "relative_errors": errors,
    }, indent=2) + "\n")
    print("FD relative errors:", errors)
    assert min(errors) < 1e-3
