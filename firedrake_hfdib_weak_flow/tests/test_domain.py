import copy
import json

import numpy as np
import pytest

from src.domain import RECONSTRUCTED_TPFM_DOMAIN, load_domain_spec
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
