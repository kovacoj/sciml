"""Lightweight geometry, interpolation, and neural-field components."""

from .domain import DomainSpec, RECONSTRUCTED_TPFM_DOMAIN, load_domain_spec
from .geometry import FullDomainGeometry, TPFMGeometry
from .hfdib import HFDIB, first_order, second_order
from .model import CoordinateMLP

__all__ = [
    "CoordinateMLP",
    "DomainSpec",
    "FullDomainGeometry",
    "HFDIB",
    "TPFMGeometry",
    "RECONSTRUCTED_TPFM_DOMAIN",
    "first_order",
    "second_order",
    "load_domain_spec",
    "FEFieldMapper",
    "NeuralFields",
    "enforce_inlet_geometry_compatibility",
    "FiredrakeContext",
    "FEGradients",
    "FiredrakeResidualBridge",
    "ResidualMetrics",
]


def __getattr__(name):
    if name in {"FEFieldMapper", "NeuralFields", "enforce_inlet_geometry_compatibility"}:
        from .state import (
            FEFieldMapper,
            NeuralFields,
            enforce_inlet_geometry_compatibility,
        )

        return {
            "FEFieldMapper": FEFieldMapper,
            "NeuralFields": NeuralFields,
            "enforce_inlet_geometry_compatibility": enforce_inlet_geometry_compatibility,
        }[name]
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
