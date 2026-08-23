import json
from pathlib import Path


def test_frozen_case0_summary_has_required_claim_levels():
    path = Path(__file__).parents[1] / "outputs/final_case0/case0_summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    assert summary["direct_fe"]["weak_loss"] < 1e-20
    assert summary["fe_coefficients_lbfgs"]["weak_loss"] < 1e-8
    assert summary["fe_coefficients_lbfgs"]["ux_relative_l2"] < 1e-2
    assert summary["coordinate_mlp"]["ux_relative_l2"] < 0.02
    assert summary["mlp_gate"]["passed"] is False
