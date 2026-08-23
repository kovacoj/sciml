"""Evaluate the configured residual on comparable Case0 coefficient fields."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import random
from pathlib import Path

import numpy as np
import torch

from .benchmark_setup import load_config_geometry
from .model import CoordinateMLP
from .residual_bridge import FiredrakeResidualBridge
from .state import NeuralFields
from .train import atomic_json, relative_mass_imbalance, validate_checkpoint_geometry


def _diagnostics(context) -> dict[str, float]:
    from firedrake import FacetNormal, as_vector, assemble, div, dot

    velocity = as_vector((context.ux, context.uy))
    normal = FacetNormal(context.mesh)
    integral_divergence = float(assemble(div(velocity) * context.measure))
    assembled_rc_sum = float(np.sum(assemble(context.rc_form).dat.data_ro))
    inlet_flux = float(assemble(
        dot(velocity, normal) * context.boundary_measure(context.inlet_marker)
    ))
    outlet_flux = float(assemble(
        dot(velocity, normal) * context.boundary_measure(context.outlet_marker)
    ))
    absolute_mass_defect = abs(inlet_flux + outlet_flux)
    return {
        "integral_divergence": integral_divergence,
        "continuity_constant_test": integral_divergence,
        "continuity_constant_test_integral": integral_divergence,
        "continuity_constant_test_assembled_rc_sum": assembled_rc_sum,
        "continuity_constant_test_difference": assembled_rc_sum - integral_divergence,
        "inlet_flux": inlet_flux,
        "outlet_flux": outlet_flux,
        "absolute_mass_defect": absolute_mass_defect,
        "relative_mass_defect": relative_mass_imbalance(inlet_flux, outlet_flux),
    }


def _evaluate(bridge, fields: NeuralFields, beta: float) -> dict:
    metrics, residuals = bridge._evaluate(fields, beta)
    metric_values = asdict(metrics)
    vectors = {
        name: np.asarray(residual.dat.data_ro, dtype=np.float64).tolist()
        for name, residual in zip(("x", "y", "continuity"), residuals)
    }
    return {
        "beta": float(beta),
        "residual_metrics": metric_values,
        "normalized": {
            "momentum": metrics.loss_x + metrics.loss_y,
            "continuity": metrics.loss_continuity,
            "total": metrics.loss,
        },
        "raw_effective_weighted_values": {
            "momentum_x": metrics.raw_loss_x / bridge.Cm,
            "momentum_y": metrics.raw_loss_y / bridge.Cm,
            "momentum": (metrics.raw_loss_x + metrics.raw_loss_y) / bridge.Cm,
            "continuity_before_gamma": metrics.raw_loss_continuity / bridge.Cc,
            "continuity_after_gamma": (
                bridge.gamma * metrics.raw_loss_continuity / bridge.Cc
            ),
            "total": metrics.loss,
        },
        "raw_residual_vectors": vectors,
        "raw_residual_vector_norms": {
            "x": metrics.residual_x_l2,
            "y": metrics.residual_y_l2,
            "continuity": metrics.residual_continuity_l2,
        },
        "diagnostics": _diagnostics(bridge.context),
    }


def _evaluate_betas(bridge, fields: NeuralFields) -> dict:
    return {
        "beta1": _evaluate(bridge, fields, 1.0),
        "beta0": _evaluate(bridge, fields, 0.0),
    }


def _direct_fields(path: Path, mapper) -> tuple[NeuralFields, dict]:
    expected_shapes = {
        "ux": (mapper.context.S.dim(),),
        "uy": (mapper.context.S.dim(),),
        "p": (mapper.context.Q.dim(),),
    }
    with np.load(path) as archive:
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64).copy()
            for name in expected_shapes
        }
        actual_shapes = {name: value.shape for name, value in arrays.items()}
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"direct coefficient shapes {actual_shapes} do not match {expected_shapes}"
            )
        coordinate_checks = {}
        for name, expected in (
            ("s_coordinates", mapper.s_coords),
            ("q_coordinates", mapper.q_coords),
        ):
            supplied = np.asarray(archive[name], dtype=np.float64)
            coordinate_checks[name] = {
                "shape": list(supplied.shape),
                "expected_shape": list(expected.shape),
                "exact_match": bool(np.array_equal(supplied, expected)),
                "max_abs_difference": float(np.max(np.abs(supplied - expected)))
                if supplied.shape == expected.shape else None,
            }
            if supplied.shape != expected.shape or not np.allclose(
                supplied, expected, rtol=0.0, atol=1.0e-14
            ):
                raise ValueError(f"direct {name} do not match the current FE mesh")

    zeros = torch.zeros(expected_shapes["ux"], dtype=torch.float64)
    fields = NeuralFields(
        ux=torch.from_numpy(arrays["ux"]),
        uy=torch.from_numpy(arrays["uy"]),
        p=torch.from_numpy(arrays["p"]),
        uibx=zeros,
        uiby=zeros.clone(),
    )
    validation = {
        "coefficient_shapes": {
            name: list(shape) for name, shape in actual_shapes.items()
        },
        "expected_coefficient_shapes": {
            name: list(shape) for name, shape in expected_shapes.items()
        },
        "coordinate_checks": coordinate_checks,
        "network_masks_applied": False,
        "uib_assignment": "zeros_for_empty_geometry",
    }
    return fields, validation


def run(config_path: Path, direct_path: Path, trained_path: Path, output: Path) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    config_path = config_path.resolve()
    direct_path = direct_path.resolve()
    trained_path = trained_path.resolve()
    config, _, _, _, geometry, context, mapper = load_config_geometry(config_path)

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = CoordinateMLP(
        input_dim=4,
        width=int(config["network_width"]),
        depth=int(config["network_depth"]),
    )
    bridge = FiredrakeResidualBridge(context, gamma=float(config["gamma"]))
    Cm, Cc = bridge.compute_baseline_normalization()

    with torch.no_grad():
        initial_fields = mapper.evaluate(model)
    initial_results = _evaluate_betas(bridge, initial_fields)

    checkpoint = torch.load(trained_path, map_location="cpu", weights_only=False)
    validate_checkpoint_geometry(config, checkpoint)
    model.load_state_dict(checkpoint["model_state"])
    with torch.no_grad():
        trained_fields = mapper.evaluate(model)
    trained_results = _evaluate_betas(bridge, trained_fields)

    direct_fields, direct_validation = _direct_fields(direct_path, mapper)
    direct_results = _evaluate_betas(bridge, direct_fields)

    report = {
        "diagnostic": "residual_consistency",
        "residual_formulation": context.formulation,
        "case": "Case0",
        "config": str(config_path),
        "direct": str(direct_path),
        "trained": str(trained_path),
        "seed": seed,
        "torch_dtype": "float64",
        "normalization": {
            "source": "FiredrakeResidualBridge.compute_baseline_normalization_boundary_lift",
            "Cm": Cm,
            "Cc": Cc,
            "gamma": bridge.gamma,
        },
        "pressure_boundary_conditions": {
            "mismatch": context.formulation != "h1_weak",
            "direct": "natural_zero_traction_outlet",
            "neural": (
                "natural_zero_traction_outlet"
                if context.formulation == "h1_weak"
                else "hard_outlet_pressure_p_equals_pout"
            ),
            "direct_pressure_remasked": False,
        },
        "direct_validation": direct_validation,
        "checkpoint_metadata": {
            name: checkpoint.get(name)
            for name in ("global_step", "stage_index", "stage_step", "beta", "Cm", "Cc")
        },
        "case0_states": {
            "A_initial_model": initial_results,
            "B_trained_model": trained_results,
            "C_direct_fe_coefficients": direct_results,
        },
        "residual_vector_convention": (
            "assembled rx, ry, rc coefficient vectors; essential velocity test entries "
            "are zeroed by the current bridge context"
        ),
    }
    atomic_json(output, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run(arguments.config, arguments.direct, arguments.trained, arguments.output)


if __name__ == "__main__":
    main()
