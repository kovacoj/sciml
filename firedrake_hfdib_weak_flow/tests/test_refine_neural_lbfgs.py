import numpy as np
import torch

from src.model import CoordinateMLP


def test_parameter_flatten_roundtrip_pattern():
    model = CoordinateMLP(input_dim=4, width=4, depth=1)
    parameters = tuple(model.parameters())
    flat = np.concatenate([
        parameter.detach().numpy().reshape(-1) for parameter in parameters
    ])
    replacement = flat + np.linspace(0.0, 1.0e-4, flat.size)
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            parameter.copy_(torch.from_numpy(
                replacement[offset:offset + size].reshape(parameter.shape)
            ))
            offset += size
    actual = np.concatenate([
        parameter.detach().numpy().reshape(-1) for parameter in parameters
    ])
    np.testing.assert_array_equal(actual, replacement)
