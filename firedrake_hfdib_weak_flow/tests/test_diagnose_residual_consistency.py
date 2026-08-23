import json
from pathlib import Path

import torch

from src.diagnose_residual_consistency import run
from src.direct_reference import run as run_direct_reference
from src.model import CoordinateMLP
from src.residual_bridge import ResidualMetrics


ROOT = Path(__file__).resolve().parents[1]


def test_report_evaluates_nonzero_direct_strong_residual(tmp_path):
    config = json.loads((ROOT / "configs/manufactured_empty_neural.json").read_text())
    config.update(
        nx=12, ny=6, network_width=8, network_depth=1,
        residual_formulation="literal_strong_hfdib",
    )
    config_path = tmp_path / "coarse.json"
    config_path.write_text(json.dumps(config))

    direct_dir = tmp_path / "direct"
    assert run_direct_reference(config_path, direct_dir)["converged"]

    torch.manual_seed(config["seed"])
    checkpoint = tmp_path / "trained.pt"
    torch.save({
        "model_state": CoordinateMLP(input_dim=4, width=8, depth=1).state_dict(),
        "config": {
            "geometry_kind": config["geometry_kind"],
            "benchmark_case": config["benchmark_case"],
        },
    }, checkpoint)
    output = tmp_path / "report.json"
    report = run(config_path, direct_dir / "direct_reference.npz", checkpoint, output)

    assert json.loads(output.read_text())["case"] == "Case0"
    assert report["residual_formulation"] == "literal_strong_hfdib"
    assert report["pressure_boundary_conditions"]["mismatch"] is True
    assert report["direct_validation"]["network_masks_applied"] is False
    assert set(report["case0_states"]) == {
        "A_initial_model", "B_trained_model", "C_direct_fe_coefficients",
    }
    expected_metrics = set(ResidualMetrics.__dataclass_fields__)
    for state in report["case0_states"].values():
        for beta in ("beta1", "beta0"):
            evaluation = state[beta]
            assert set(evaluation["residual_metrics"]) == expected_metrics
            assert set(evaluation["normalized"]) == {"momentum", "continuity", "total"}
            assert set(evaluation["raw_residual_vectors"]) == {"x", "y", "continuity"}
            assert "continuity_constant_test_difference" in evaluation["diagnostics"]

    direct = report["case0_states"]["C_direct_fe_coefficients"]
    assert direct["beta1"]["normalized"]["total"] > 1.0e-12
    assert direct["beta0"]["normalized"]["total"] > 1.0e-12
