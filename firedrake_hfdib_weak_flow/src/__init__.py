"""Lightweight geometry, interpolation, and neural-field components."""

from .geometry import TPFMGeometry
from .hfdib import HFDIB, first_order, second_order
from .model import CoordinateMLP

__all__ = [
    "CoordinateMLP",
    "HFDIB",
    "TPFMGeometry",
    "first_order",
    "second_order",
    "FEFieldMapper",
    "NeuralFields",
    "FiredrakeContext",
    "FEGradients",
    "FiredrakeResidualBridge",
    "ResidualMetrics",
]


def __getattr__(name):
    if name in {"FEFieldMapper", "NeuralFields"}:
        from .state import FEFieldMapper, NeuralFields

        return {"FEFieldMapper": FEFieldMapper, "NeuralFields": NeuralFields}[name]
    if name == "FiredrakeContext":
        from .forms import FiredrakeContext

        return FiredrakeContext
    if name in {"FEGradients", "FiredrakeResidualBridge", "ResidualMetrics"}:
        from .residual_bridge import FEGradients, FiredrakeResidualBridge, ResidualMetrics

        return {
            "FEGradients": FEGradients,
            "FiredrakeResidualBridge": FiredrakeResidualBridge,
            "ResidualMetrics": ResidualMetrics,
        }[name]
    raise AttributeError(name)
