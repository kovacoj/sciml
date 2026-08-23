import json
from pathlib import Path

import numpy as np
import torch

from src.benchmark_setup import load_config_geometry
from src.direct_reference import run as run_direct_reference
from src.optimize_fe_coefficients import run as optimize_fe_coefficients
from src.residual_bridge import FiredrakeResidualBridge
from src.state import NeuralFields


ROOT = Path(__file__).resolve().parents[1]


def _direct_fields(path, context):
    with np.load(path) as archive:
        zeros = torch.zeros(context.S.dim(), dtype=torch.float64)
        return NeuralFields(
            torch.from_numpy(archive["ux"].copy()),
            torch.from_numpy(archive["uy"].copy()),
            torch.from_numpy(archive["p"].copy()),
            zeros,
            zeros.clone(),
        )


def test_direct_solution_minimizes_neural_weak_residual(tmp_path):
    config_path = ROOT / "configs/manufactured_empty_direct_coarse_geometry.json"
    direct_dir = tmp_path / "direct"
    assert run_direct_reference(config_path, direct_dir)["converged"]
    _, _, _, _, _, context, _ = load_config_geometry(config_path)
    fields = _direct_fields(direct_dir / "direct_reference.npz", context)
    bridge = FiredrakeResidualBridge(context)
    bridge.initialize_normalization(fields)
    metrics = bridge.evaluate_loss(fields, beta=0.0)

    assert metrics.residual_x_l2 < 1.0e-8
    assert metrics.residual_y_l2 < 1.0e-8
    assert metrics.residual_continuity_l2 < 1.0e-8
    assert metrics.loss < 1.0e-12


def test_coarse_fe_coefficient_optimization_decreases_loss_and_direct_error(tmp_path):
    config = json.loads(
        (ROOT / "configs/manufactured_empty_fe_coefficients.json").read_text()
    )
    config.update(nx=8, ny=4, continuation=[{"beta": 0.0, "steps": 20, "lr": 0.01}])
    config_path = tmp_path / "coarse_fe.json"
    config_path.write_text(json.dumps(config))
    direct_dir = tmp_path / "direct"
    assert run_direct_reference(config_path, direct_dir)["converged"]

    result = optimize_fe_coefficients(
        config_path, tmp_path / "optimized", direct_dir / "direct_reference.npz"
    )
    assert result["final"]["loss"] < result["initial"]["loss"]
    assert result["comparison"]["ux_relative_l2"] < 1.0
    assert (tmp_path / "optimized/final_fields.npz").exists()
    assert (tmp_path / "optimized/metrics.json").exists()
