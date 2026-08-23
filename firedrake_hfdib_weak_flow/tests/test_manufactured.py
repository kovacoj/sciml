import json
from pathlib import Path

import numpy as np
import pytest
from firedrake import assemble

from src.benchmark_setup import load_config_geometry
from src.compare_reference import compare
from src.direct_reference import main as direct_reference_main, run as run_direct_reference
from src.geometry import CircularObstacleGeometry, EmptyChannelGeometry
from src.geometry_smoke import run as run_geometry_smoke
from src.physical_validation import validate_physical_domain


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=("manufactured_empty", "manufactured_circle"))
def manufactured_setup(request):
    return load_config_geometry(ROOT / f"configs/{request.param}_geometry.json")


def test_empty_channel_geometry_is_finite_and_has_no_hfdib_path():
    geometry = EmptyChannelGeometry()
    points = np.array([[0.0, 0.0], [0.128, 0.064], [0.256, 0.128]])
    assert geometry.lambda_field.shape == (64, 128)
    assert np.count_nonzero(geometry.lambda_field) == 0
    assert np.count_nonzero(geometry.interface) == 0
    assert np.all(np.isfinite(geometry.interpolate(points)))
    assert np.count_nonzero(geometry.interpolate(points, "normals")) == 0
    assert geometry.spec.article_reproduction is False


def test_circle_uses_exact_analytic_sigma_lambda_and_outward_normals():
    geometry = CircularObstacleGeometry()
    points = np.array([
        geometry.center,
        geometry.center + [geometry.radius, 0.0],
        geometry.center + [0.0, 2.0 * geometry.radius],
    ])
    np.testing.assert_allclose(
        geometry.interpolate(points), [-geometry.radius, 0.0, geometry.radius], atol=1e-15
    )
    np.testing.assert_allclose(
        geometry.interpolate(points, "lambda"),
        0.5 * (1.0 - np.tanh(np.array([-geometry.radius, 0.0, geometry.radius]) / 0.002)),
    )
    np.testing.assert_allclose(
        geometry.interpolate(points, "normals"), [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        atol=1e-15,
    )
    assert np.any(geometry.lambda_field > 0.5)
    assert np.any(geometry.interface)


def test_manufactured_physical_gates_connectivity_aspect_and_bcs(manufactured_setup):
    _, dataset, domain_path, spec, geometry, context, mapper = manufactured_setup
    assert dataset is None and domain_path is None
    report = validate_physical_domain(geometry, context, mapper)
    assert report["hard_gate_passed"]
    assert report["fluid_connected"]
    assert report["inlet_geometry_conflicts"] == 0
    assert report["outlet_geometry_conflicts"] == 0
    assert report["article_reproduction"] is False
    assert report["aspect_error"] <= 0.1
    assert len(context.inlet_velocity_nodes) > 0
    assert len(context.pressure_outlet_nodes) > 0
    coordinates = context._coordinates(context.S)
    corners = (
        (np.isclose(coordinates[:, 0], context.xmin)
         | np.isclose(coordinates[:, 0], context.xmax))
        & (np.isclose(coordinates[:, 1], context.ymin)
           | np.isclose(coordinates[:, 1], context.ymax))
    )
    assert np.all(context.ux_lift[corners] == 0.0)
    assert np.all(context.velocity_mask[corners] == 0.0)
    assert spec.inlet[0].side == "left" and spec.outlet[0].side == "right"
    assert (spec.uin, spec.pout, spec.nu) == (0.1, 0.0, 0.01)
    if geometry.classification == "MANUFACTURED_EMPTY_CHANNEL":
        assert np.count_nonzero(context.chi.dat.data_ro) == 0


def test_article_operator_invariants_hold_for_manufactured(manufactured_setup):
    _, _, _, _, _, context, _ = manufactured_setup
    rng = np.random.default_rng(91)
    for field in (context.ux, context.uy, context.p):
        field.dat.data[:] = rng.standard_normal(field.dat.data.shape)
    context.uibx.assign(context.ux)
    context.uiby.assign(context.uy)
    context.chi.assign(1.0)
    context.beta.assign(0.25)
    rx, ry, _ = context.residual_vectors()
    assert np.linalg.norm(rx.dat.data_ro) + np.linalg.norm(ry.dat.data_ro) < 1e-10
    context.chi.assign(0.0)
    before = assemble(context.rx_form).dat.data_ro.copy()
    context.uibx.dat.data[:] = rng.standard_normal(context.uibx.dat.data.shape)
    after = assemble(context.rx_form).dat.data_ro
    assert np.max(np.abs(after - before)) < 1e-14


def test_geometry_smoke_writes_manufactured_labels(tmp_path):
    for case in ("manufactured_empty", "manufactured_circle"):
        output = tmp_path / case
        report = run_geometry_smoke(ROOT / f"configs/{case}_geometry.json", output)
        labels = json.loads((output / "geometry_labels.json").read_text())
        assert report["hard_gate_passed"]
        assert labels["benchmark_mode"] == "manufactured"
        assert labels["article_reproduction"] is False
        assert (output / f"geometry_{case}.png").stat().st_size > 0


def test_empty_direct_reference_coarse_converges_and_writes_cli_artifacts(tmp_path):
    config = json.loads((ROOT / "configs/manufactured_empty_geometry.json").read_text())
    config.update(nx=12, ny=6)
    config_path = tmp_path / "direct_coarse.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "direct"
    metrics = run_direct_reference(config_path, output)
    assert metrics["converged"]
    assert metrics["mass_imbalance"] < 1.0e-6
    with np.load(output / "direct_reference.npz") as archive:
        assert all(np.all(np.isfinite(value)) for value in archive.values())
    assert json.loads((output / "direct_reference_metrics.json").read_text())["converged"]

    cli_output = tmp_path / "direct_cli"
    direct_reference_main([
        "--config", str(config_path), "--output-dir", str(cli_output),
    ])
    assert (cli_output / "direct_reference.npz").stat().st_size > 0
    assert (cli_output / "direct_reference_metrics.json").stat().st_size > 0


def test_circle_direct_reference_is_explicitly_deferred(tmp_path):
    metrics = run_direct_reference(
        ROOT / "configs/manufactured_circle_geometry.json", tmp_path
    )
    assert metrics["status"] == "deferred"
    assert "fixed-point" in metrics["reason"]
    assert not (tmp_path / "direct_reference.npz").exists()


def test_compare_reference_names_simple_norm_and_centers_pressure(tmp_path):
    reference = tmp_path / "reference.npz"
    neural = tmp_path / "neural.npz"
    np.savez(reference, ux=[1.0, 2.0], uy=[0.0, 1.0], p=[1.0, 2.0])
    np.savez(neural, ux=[1.0, 2.0], uy=[0.0, 1.0], p=[11.0, 12.0])
    result = compare(neural, reference)
    assert result["norm"] == "simple_unweighted_FE_coefficient_relative_l2"
    assert result["ux_relative_l2"] == 0.0
    assert result["pressure_gauge_centered_relative_l2"] == 0.0
