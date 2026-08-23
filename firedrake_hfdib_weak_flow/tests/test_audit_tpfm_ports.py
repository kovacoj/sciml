import json

import numpy as np
import pytest

from src.audit_tpfm_ports import audit_inputs, contiguous_intervals, main


def synthetic_inputs():
    inputs = np.ones((3, 1, 64, 6))
    fluid = inputs[:, 0]
    fluid[:, 2, 0] = 0.0
    fluid[:, 3, -1] = 0.0

    fluid[0, 8:16, :] = 0.0
    fluid[0, 48:56, :] = 0.0

    fluid[1, 8:12, 0] = 0.0
    fluid[1, 8:16, -1] = 0.0
    fluid[1, 48:52, -1] = 0.0

    fluid[2, 8:12, :] = 0.0
    fluid[2, 48:52, 0] = 0.0
    fluid[2, 48:52, -1] = 0.0
    fluid[2, 20, 1] = 0.5
    return inputs


def test_contiguous_intervals_use_cell_boundary_physical_bounds():
    intervals = contiguous_intervals([False, True, True, False, True])

    assert intervals == [
        {
            "row_start": 1,
            "row_end_exclusive": 3,
            "physical_y_min": pytest.approx(0.002),
            "physical_y_max": pytest.approx(0.006),
        },
        {
            "row_start": 4,
            "row_end_exclusive": 5,
            "physical_y_min": pytest.approx(0.008),
            "physical_y_max": pytest.approx(0.010),
        },
    ]


def test_audit_covers_invariants_closures_components_and_connectivity():
    report, _ = audit_inputs(synthetic_inputs())

    assert report["sample_count"] == 3
    assert report["shape"] == [3, 1, 64, 6]
    assert report["invariant_fluid_rows"] == {
        "left": [2, 8, 9, 10, 11],
        "right": [3, 8, 9, 10, 11, 48, 49, 50, 51],
        "left_right_intersection": [8, 9, 10, 11],
    }
    assert report["threshold"]["fluid_predicate"] == "lambda < 0.5"
    assert report["edge_mask_patterns"]["distinct_count"] == 3
    assert report["expected_port_edge_status"]["left"][1]["closed_sample_indices"] == [1]
    assert report["expected_port_edge_status"]["right"][1]["partial_sample_indices"] == [1, 2]
    assert report["per_sample"]["fluid_component_count"] == [4, 5, 5]
    assert report["per_sample"]["has_any_left_right_path"] == [True, False, True]
    assert report["per_sample"]["expected_left_port_connected_to_any_expected_right_port"] == [
        [True, True], [False, False], [True, False],
    ]
    assert report["connectivity_summary"]["no_left_right_path_sample_indices"] == [1]


def test_cli_writes_json_and_headless_figure(tmp_path):
    dataset = tmp_path / "synthetic.npz"
    output_dir = tmp_path / "audit"
    np.savez(dataset, inputs=synthetic_inputs())

    assert main(["--dataset", str(dataset), "--output-dir", str(output_dir)]) == 0

    report = json.loads((output_dir / "invariant_port_structure.json").read_text())
    assert report["sample_count"] == 3
    assert (output_dir / "invariant_port_structure.png").stat().st_size > 0
