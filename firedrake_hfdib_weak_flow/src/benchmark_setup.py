"""Shared benchmark configuration and geometry construction."""

from __future__ import annotations

import json
from pathlib import Path

from .domain import load_domain_spec, validate_benchmark_mode
from .geometry import (
    CircularObstacleGeometry, EmptyChannelGeometry, FullDomainGeometry, TPFMGeometry,
)
from .train import resolve_path


def load_config_geometry(config_path: Path):
    """Load config, dataset, domain, Firedrake context, and field mapper."""
    from .forms import FiredrakeContext
    from .state import FEFieldMapper

    config_path = config_path.resolve()
    project_root = Path(__file__).resolve().parents[1]
    config = json.loads(config_path.read_text())
    geometry_kind = config.get("geometry_kind", "tpfm")
    dataset = None
    if geometry_kind == "tpfm":
        dataset = resolve_path(config["dataset"], config_path, project_root)
        if not dataset.exists():
            raise FileNotFoundError(f"dataset not found: {dataset}")
    domain_spec_path = (
        resolve_path(config["domain_spec"], config_path, project_root)
        if config.get("domain_spec") is not None else None
    )
    if geometry_kind in {"manufactured_empty", "manufactured_circle"}:
        if domain_spec_path is not None or "dataset" in config:
            raise ValueError("manufactured geometry does not use dataset or domain_spec")
        geometry_class = (
            EmptyChannelGeometry
            if geometry_kind == "manufactured_empty" else CircularObstacleGeometry
        )
        geometry = geometry_class(spacing=float(config.get("spacing", 0.002)))
        spec = geometry.spec
        validate_benchmark_mode(config.get("benchmark_mode"), spec)
        for name in ("uin", "pout", "nu"):
            if name in config and float(config[name]) != getattr(spec, name):
                raise ValueError(f"manufactured config {name} must equal {getattr(spec, name)}")
        physical = {name: getattr(spec, name) for name in ("uin", "pout", "nu")}
    elif geometry_kind != "tpfm":
        raise ValueError(f"unknown geometry_kind: {geometry_kind}")
    elif domain_spec_path is None:
        validate_benchmark_mode(config.get("benchmark_mode"), None)
        geometry = TPFMGeometry(
            dataset,
            sample_index=int(config["topology_index"]),
            spacing=float(config["spacing"]),
        )
        physical = {name: float(config[name]) for name in ("uin", "pout", "nu")}
        spec = None
    else:
        spec = load_domain_spec(domain_spec_path)
        validate_benchmark_mode(config.get("benchmark_mode"), spec)
        for name in ("uin", "pout", "nu"):
            if name in config:
                raise ValueError(
                    f"config {name} must be absent when domain_spec supplies physical values"
                )
        roi_geometry = TPFMGeometry(
            dataset, sample_index=int(config["topology_index"]), spacing=spec.dx
        )
        geometry = FullDomainGeometry(roi_geometry, spec)
        physical = {name: getattr(spec, name) for name in ("uin", "pout", "nu")}
    context = FiredrakeContext(
        geometry, int(config["nx"]), int(config["ny"]),
        velocity_degree=int(config.get("velocity_degree", 2)),
        pressure_degree=int(config.get("pressure_degree", 1)),
        quadrature_degree=int(config.get("quadrature_degree", 6)),
        nu=physical["nu"], ell=config.get("ell"), uin=physical["uin"],
        pout=physical["pout"], inlet_marker=int(config.get("inlet_marker", 1)),
        outlet_marker=int(config.get("outlet_marker", 2)),
        wall_markers=config.get("wall_markers", [3, 4]),
        formulation=config.get("residual_formulation", "literal_strong_hfdib"),
    )
    mapper = FEFieldMapper(
        context, geometry, u_scale=float(config.get("u_scale", 0.1)),
        p_scale=float(config.get("p_scale", 0.01)),
    )
    return config, dataset, domain_spec_path, spec, geometry, context, mapper
