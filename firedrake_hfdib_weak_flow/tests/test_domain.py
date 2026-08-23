import copy
import json

import numpy as np
import pytest

from src.domain import (
    CONTROLLED_TPFM_DERIVED_DOMAIN,
    RECONSTRUCTED_TPFM_DOMAIN,
    load_domain_spec,
    validate_benchmark_mode,
)
from src.geometry import FullDomainGeometry, TPFMGeometry


def _load(tmp_path, data):
    path = tmp_path / "domain.json"
    path.write_text(json.dumps(data))
    return load_domain_spec(path)


def _roi_geometry(tmp_path):
    lam = np.array([
        [0.0, 0.2, 1.0, 1.0],
        [0.0, 0.0, 0.8, 1.0],
        [1.0, 0.5, 0.0, 0.0],
    ])
    path = tmp_path / "roi.npz"
    np.savez(path, inputs=lam[None, None])
    return TPFMGeometry(path, spacing=1.0), lam


def test_valid_spec_loads_frozen_machine_contract(tmp_path, synthetic_domain_data):
    synthetic_domain_data["patches"]["inlet"][0].pop("marker")
    spec = _load(tmp_path, synthetic_domain_data)
    assert spec.classification == RECONSTRUCTED_TPFM_DOMAIN
    assert spec.article_reproduction is True
    assert spec.roi.nx == 4
    assert spec.full_domain.bounds.xmin == -2.0
    assert spec.inlet[0].name == "inlet"
    assert spec.inlet[0].marker is None
    with pytest.raises(Exception):
        spec.dx = 2.0


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.pop("dx"), "missing required keys"),
        (lambda data: data.update(extra=1), "unknown keys"),
        (lambda data: data.update(classification="ROI"), "classification"),
        (lambda data: data.update(dx=0), "dx must be > 0"),
        (lambda data: data["roi"].update(nx=5), "roi x bounds"),
        (lambda data: data["full_domain"].update(ny=4), "full_domain y bounds"),
        (lambda data: data.update(right_extension_cells=-1), "right_extension_cells"),
        (lambda data: data["full_domain"].update(nx=8, bounds={"xmin": -2, "ymin": 10, "xmax": 6, "ymax": 13}), "must equal"),
        (lambda data: data["patches"]["inlet"][0].update(side="front"), "side must"),
        (lambda data: (data["patches"]["inlet"][0].pop("name"), data["patches"]["inlet"][0].pop("marker")), "requires a marker or name"),
        (lambda data: data["patches"]["inlet"][0].update(intervals=[{"min": 9, "max": 11}]), "outside"),
        (lambda data: data["patches"]["inlet"][0].update(intervals=[{"min": 10, "max": 12}, {"min": 11, "max": 13}]), "overlapping"),
        (lambda data: data["patches"]["inlet"][0].update(intervals=[{"min": 11, "max": 13}, {"min": 10, "max": 11}]), "ordered by min"),
        (lambda data: data["roi_cell_indices"][0].__setitem__(1, 2), "unique"),
        (lambda data: data["roi_cell_indices"][0].__setitem__(1, 99), "out-of-range"),
        (lambda data: data.update(roi_cell_indices=[[2, 3]]), "shape"),
        (lambda data: data.update(nu=0), "nu must be > 0"),
    ],
)
def test_invalid_specs_fail_clearly(tmp_path, synthetic_domain_data, mutation, message):
    data = copy.deepcopy(synthetic_domain_data)
    mutation(data)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, data)


def test_extension_construction_carves_only_explicit_channels_and_roundtrips_roi(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 11.0, "max": 12.0}
    ]
    synthetic_domain_data["patches"]["outlet"][0]["intervals"] = [
        {"min": 10.0, "max": 11.0}
    ]
    synthetic_domain_data["patches"]["wall"].extend([
        {
            "name": "left-wall", "side": "left", "marker": 105,
            "intervals": [{"min": 10.0, "max": 11.0}, {"min": 12.0, "max": 13.0}],
        },
        {
            "name": "right-wall", "side": "right", "marker": 106,
            "intervals": [{"min": 11.0, "max": 13.0}],
        },
    ])
    spec = _load(tmp_path, synthetic_domain_data)
    roi_geometry, roi_lambda = _roi_geometry(tmp_path)
    geometry = FullDomainGeometry(roi_geometry, spec)

    np.testing.assert_array_equal(geometry.lambda_field[:, 2:6], roi_lambda)
    np.testing.assert_array_equal(geometry.lambda_field[:, :2], [
        [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]
    ])
    np.testing.assert_array_equal(geometry.lambda_field[:, 6], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(geometry.extract_roi(geometry.lambda_field), roi_lambda)
    np.testing.assert_array_equal(
        geometry.extract_roi(np.arange(21).reshape(3, 7)),
        np.asarray(spec.roi_cell_indices),
    )
    xx, yy = np.meshgrid(geometry.x, geometry.y)
    np.testing.assert_array_equal(
        geometry.interpolate(np.stack((xx, yy), axis=-1), "lambda"),
        geometry.lambda_field,
    )
    assert geometry.signed_distance.shape == geometry.lambda_field.shape
    assert geometry.normals.shape == geometry.lambda_field.shape + (2,)


def test_arbitrary_valid_mapping_is_rejected_by_rectangular_builder(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["roi_cell_indices"][0][0:2] = [3, 2]
    spec = _load(tmp_path, synthetic_domain_data)
    roi_geometry, _ = _roi_geometry(tmp_path)
    with pytest.raises(ValueError, match="arbitrary mappings are not supported"):
        FullDomainGeometry(roi_geometry, spec)


def test_channel_carving_is_half_open_except_at_global_upper_endpoint(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 10.0, "max": 10.5}
    ]
    synthetic_domain_data["patches"]["outlet"][0]["intervals"] = [
        {"min": 12.5, "max": 13.0}
    ]
    synthetic_domain_data["patches"]["wall"].extend([
        {
            "name": "left-wall", "side": "left", "marker": 105,
            "intervals": [{"min": 10.5, "max": 13.0}],
        },
        {
            "name": "right-wall", "side": "right", "marker": 106,
            "intervals": [{"min": 10.0, "max": 12.5}],
        },
    ])
    geometry = FullDomainGeometry(
        _roi_geometry(tmp_path)[0], _load(tmp_path, synthetic_domain_data)
    )
    assert np.all(geometry.lambda_field[:, :2] == 1.0)
    assert geometry.lambda_field[-1, -1] == 0.0
    assert np.all(geometry.lambda_field[:-1, -1] == 1.0)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["patches"].update(wall=data["patches"]["wall"][:1]), "full top side"),
        (lambda data: data["patches"]["inlet"][0].update(intervals=[{"min": 10.5, "max": 13.0}]), "leading gap"),
        (lambda data: data["patches"]["inlet"][0].update(intervals=[{"min": 10.0, "max": 12.5}]), "trailing gap"),
    ],
)
def test_boundary_partition_rejects_missing_coverage(
    tmp_path, synthetic_domain_data, mutation, message
):
    mutation(synthetic_domain_data)
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, synthetic_domain_data)


def test_boundary_partition_rejects_interior_gap(tmp_path, synthetic_domain_data):
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 10.0, "max": 11.0}
    ]
    synthetic_domain_data["patches"]["wall"].append({
        "name": "left-wall", "side": "left", "marker": 105,
        "intervals": [{"min": 11.5, "max": 13.0}],
    })
    with pytest.raises(ValueError, match="interior gap"):
        _load(tmp_path, synthetic_domain_data)


def test_complete_segmented_partition_and_touching_endpoints_are_accepted(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 10.0, "max": 11.0}, {"min": 12.0, "max": 13.0}
    ]
    synthetic_domain_data["patches"]["wall"].append({
        "name": "left-wall", "side": "left", "marker": 105,
        "intervals": [{"min": 11.0, "max": 12.0}],
    })
    spec = _load(tmp_path, synthetic_domain_data)
    assert len(spec.inlet[0].intervals) == 2
    assert spec.inlet[0].intervals[0].maximum == spec.wall[-1].intervals[0].minimum


def test_template_is_valid_json_but_intentionally_rejected():
    path = __import__("pathlib").Path(__file__).parents[1] / "geometry/domain_spec.template.json"
    json.loads(path.read_text())
    with pytest.raises(ValueError, match="dx must be a finite number"):
        load_domain_spec(path)


def test_rejected_dafoam_handoff_cannot_be_loaded_as_domain_spec():
    path = __import__("pathlib").Path(__file__).parents[1] / "geometry/dafoam_handoff_status.json"
    status = json.loads(path.read_text())
    assert status["classification"] == "TPFM_TOPOLOGIES_ONLY"
    assert status["accepted_as_reconstructed_tpfm_domain"] is False
    with pytest.raises(ValueError, match="unknown keys|missing required keys"):
        load_domain_spec(path)


def test_controlled_spec_exact_generator_contract_and_mapping():
    path = __import__("pathlib").Path(__file__).parents[1] / "geometry/controlled_tpfm_32cell.json"
    spec = load_domain_spec(path)
    assert spec.classification == CONTROLLED_TPFM_DERIVED_DOMAIN
    assert spec.article_reproduction is False
    assert spec.dx == 0.002
    assert (spec.roi.nx, spec.roi.ny) == (64, 64)
    assert vars(spec.roi.bounds) == {
        "xmin": 0.0, "ymin": 0.0, "xmax": 0.128, "ymax": 0.128,
    }
    assert (spec.full_domain.nx, spec.full_domain.ny) == (128, 64)
    assert vars(spec.full_domain.bounds) == {
        "xmin": -0.064, "ymin": 0.0, "xmax": 0.192, "ymax": 0.128,
    }
    assert (spec.left_extension_cells, spec.right_extension_cells) == (32, 32)
    assert spec.uin == 0.1 and spec.pout == 0.0 and spec.nu == 0.01
    assert spec.roi_cell_indices == tuple(
        tuple(j * 128 + i for i in range(32, 96)) for j in range(64)
    )
    assert [(i.minimum, i.maximum) for i in spec.inlet[0].intervals] == [
        (0.016, 0.032), (0.096, 0.112),
    ]
    assert [(i.minimum, i.maximum) for i in spec.outlet[0].intervals] == [
        (0.016, 0.032), (0.096, 0.112),
    ]


def test_controlled_provenance_is_required_and_strict(tmp_path):
    source = __import__("pathlib").Path(__file__).parents[1] / "geometry/controlled_tpfm_32cell.json"
    data = json.loads(source.read_text())
    expected = {
        "source_classification": "TPFM_TOPOLOGIES_ONLY",
        "source_branch": "feat/dafoam-tpfm-reproduction",
        "source_gate_passed": False,
        "article_reproduction": False,
    }
    assert data["provenance"]["source"].endswith(
        "dafoam_pytorch_hfdib_demo/python/tpfm_reference/full_domain_case.py"
    )
    assert isinstance(data["provenance"]["extension_selection"], str)
    assert all(data["provenance"][key] == value for key, value in expected.items())
    for mutation, message in (
        (lambda value: value.pop("provenance"), "provenance must be an object"),
        (lambda value: value["provenance"].update(extra=True), "unknown keys"),
        (lambda value: value["provenance"].pop("source_branch"), "missing required"),
        (lambda value: value["provenance"].update(source_gate_passed=0), "must be False"),
        (lambda value: value["provenance"].update(article_reproduction=True), "must be False"),
        (lambda value: value["provenance"].update(source=1), "non-empty string"),
        (lambda value: value["provenance"].update(source_classification="OTHER"), "TPFM_TOPOLOGIES_ONLY"),
        (lambda value: value["provenance"].update(source_branch=1), "feat/dafoam-tpfm-reproduction"),
        (lambda value: value["provenance"].update(extension_selection=False), "non-empty string"),
    ):
        changed = copy.deepcopy(data)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            _load(tmp_path, changed)


def test_benchmark_mode_is_explicit_and_cannot_relabel(tmp_path, synthetic_domain_data):
    authoritative = _load(tmp_path, synthetic_domain_data)
    controlled_path = __import__("pathlib").Path(__file__).parents[1] / "geometry/controlled_tpfm_32cell.json"
    controlled = load_domain_spec(controlled_path)
    assert validate_benchmark_mode("article", authoritative) == "article"
    assert validate_benchmark_mode("controlled", controlled) == "controlled"
    assert validate_benchmark_mode(None, None) == "diagnostic"
    for mode, spec, message in (
        (None, controlled, "required"),
        ("invalid", controlled, "must be one"),
        ("article", controlled, "does not match"),
        ("controlled", authoritative, "does not match"),
        ("controlled", None, "requires domain_spec"),
    ):
        classification = None if spec is None else spec.classification
        with pytest.raises(ValueError, match=message):
            validate_benchmark_mode(mode, spec)
        assert (None if spec is None else spec.classification) == classification
