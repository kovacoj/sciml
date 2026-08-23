import numpy as np
import pytest
import torch
from firedrake import SpatialCoordinate, assemble

from src.domain import DomainSpec
from src.forms import FiredrakeContext
from src.geometry import FullDomainGeometry, TPFMGeometry
from src.model import CoordinateMLP
from src.state import (
    FEFieldMapper,
    NeuralFields,
    enforce_inlet_geometry_compatibility,
)


def _full_geometry(tmp_path, data, lam=None):
    if lam is None:
        lam = np.zeros((3, 4), dtype=np.float64)
    path = tmp_path / "full_domain_roi.npz"
    np.savez(path, inputs=lam[None, None])
    return FullDomainGeometry(
        TPFMGeometry(path, spacing=1.0), DomainSpec.parse(data)
    )


def test_context_domain_spaces_and_exact_fe_boundary_constraints(synthetic_geometry_path):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(geometry, 6, 5)
    with pytest.warns(RuntimeWarning, match="hard inlet overlaps"):
        mapper = FEFieldMapper(context, geometry)
    fields = mapper.evaluate(CoordinateMLP(input_dim=4, width=8, depth=2))
    assert isinstance(fields, NeuralFields)
    assert context.Lx == geometry.nx * spacing
    assert context.Ly == geometry.ny * spacing

    x, y = mapper.s_coords.T
    inlet_noncorner = np.isclose(x, 0.0) & ~np.isclose(y, 0.0) & ~np.isclose(y, context.Ly)
    walls = np.isclose(y, 0.0) | np.isclose(y, context.Ly)
    outlet = context.pressure_outlet_nodes
    assert np.max(np.abs(fields.ux.detach().numpy()[inlet_noncorner] - 0.1)) < 1e-14
    assert np.max(np.abs(fields.uy.detach().numpy()[inlet_noncorner])) < 1e-14
    assert np.max(np.abs(fields.ux.detach().numpy()[walls])) < 1e-14
    assert np.max(np.abs(fields.uy.detach().numpy()[walls])) < 1e-14
    assert np.max(np.abs(fields.p.detach().numpy()[outlet])) < 1e-14
    assert np.all(context.velocity_mask[context.velocity_boundary_nodes] == 0.0)
    assert set(np.unique(context.chi.dat.data_ro)) <= {0.0, 1.0}
    assert mapper.inlet_geometry_conflicts > 0


def test_uib_graph_explicitly_reaches_network_parameters(synthetic_geometry_path):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(geometry, 6, 6)
    model = CoordinateMLP(input_dim=4, width=8, depth=2)
    fields = FEFieldMapper(context, geometry).evaluate(model)
    interface = np.flatnonzero(
        (context.lam.dat.data_ro > 0.0) & (context.lam.dat.data_ro < 1.0)
    )
    assert fields.uibx.grad_fn is not None
    assert torch.count_nonzero(fields.uibx).item() > 0
    torch.autograd.backward((fields.uibx, fields.uiby), (
        torch.ones_like(fields.uibx), torch.ones_like(fields.uiby)
    ))
    assert len(interface) > 0
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())


def test_uib_is_zero_off_interface(synthetic_geometry_path):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(geometry, 6, 6)
    mapper = FEFieldMapper(context, geometry)
    fields = mapper.evaluate(CoordinateMLP(input_dim=4, width=8, depth=2))
    interface = (mapper.s_lambda > 1e-12) & (mapper.s_lambda < 1.0 - 1e-12)
    assert np.max(np.abs(fields.uibx.detach().numpy()[~interface])) == 0.0
    assert np.max(np.abs(fields.uiby.detach().numpy()[~interface])) == 0.0


def test_configurable_boundary_markers_drive_fe_masks(synthetic_geometry_path):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(
        geometry, 5, 5, inlet_marker=2, outlet_marker=1,
        wall_markers=[3, 4], pout=0.0,
    )
    mapper = FEFieldMapper(context, geometry)
    fields = mapper.evaluate(CoordinateMLP(input_dim=4, width=8, depth=2))
    x, y = mapper.s_coords.T
    inlet = np.isclose(x, context.xmax) & ~np.isclose(y, 0.0) & ~np.isclose(y, context.ymax)
    assert np.max(np.abs(fields.ux.detach().numpy()[inlet] - 0.1)) < 1e-14
    assert np.max(np.abs(fields.p.detach().numpy()[context.pressure_outlet_nodes])) < 1e-14


def test_reconstructed_context_uses_shifted_bounds_and_exact_full_side_nodes(
    tmp_path, synthetic_domain_data
):
    geometry = _full_geometry(tmp_path, synthetic_domain_data)
    context = FiredrakeContext(geometry, 7, 3)
    mapper = FEFieldMapper(context, geometry)
    assert (context.xmin, context.ymin, context.xmax, context.ymax) == (
        -2.0, 10.0, 5.0, 13.0
    )
    assert np.isclose(mapper.s_coords[:, 0].min(), context.xmin)
    assert np.isclose(mapper.s_coords[:, 1].min(), context.ymin)
    assert np.isclose(mapper.s_features[:, 0].min(), -1.0)
    assert np.isclose(mapper.s_features[:, 0].max(), 1.0)
    np.testing.assert_array_equal(
        context.inlet_velocity_nodes,
        np.flatnonzero(np.isclose(mapper.s_coords[:, 0], context.xmin)),
    )
    assert mapper.inlet_geometry_conflicts == 0
    enforce_inlet_geometry_compatibility(mapper)


def test_reconstructed_nonzero_inlet_geometry_conflict_is_a_hard_gate(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["left_extension_cells"] = 0
    synthetic_domain_data["full_domain"] = {
        "nx": 5,
        "ny": 3,
        "bounds": {"xmin": 0.0, "ymin": 10.0, "xmax": 5.0, "ymax": 13.0},
    }
    synthetic_domain_data["patches"]["wall"][0]["intervals"] = [{"min": 0.0, "max": 5.0}]
    synthetic_domain_data["patches"]["wall"][1]["intervals"] = [{"min": 0.0, "max": 5.0}]
    synthetic_domain_data["roi_cell_indices"] = [
        [0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13]
    ]
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 10.0, "max": 10.5},
        {"min": 11.5, "max": 13.0},
    ]
    synthetic_domain_data["patches"]["wall"].append({
        "name": "left-wall", "side": "left", "marker": 105,
        "intervals": [{"min": 10.5, "max": 11.5}],
    })
    lam = np.zeros((3, 4), dtype=np.float64)
    # The lower corner is an explicit wall. Put solid geometry at the middle
    # left-edge cell, where the inlet lift is genuinely nonzero.
    lam[1, 0] = 1.0
    geometry = _full_geometry(tmp_path, synthetic_domain_data, lam)
    mapper = FEFieldMapper(FiredrakeContext(geometry, 5, 3), geometry)
    assert mapper.inlet_geometry_conflicts > 0
    with pytest.raises(RuntimeError, match="nonzero inlet velocity DOFs"):
        enforce_inlet_geometry_compatibility(mapper)
    zero_inlet_context = FiredrakeContext(geometry, 5, 3, uin=0.0)
    zero_inlet_mapper = FEFieldMapper(zero_inlet_context, geometry)
    assert zero_inlet_mapper.inlet_geometry_conflicts == 0
    enforce_inlet_geometry_compatibility(zero_inlet_mapper)


def test_segmented_reconstructed_boundary_uses_exact_interval_nodes(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 10.5, "max": 11.5},
        {"min": 11.5, "max": 13.0},
    ]
    synthetic_domain_data["patches"]["outlet"][0]["intervals"] = [
        {"min": 10.0, "max": 11.5}
    ]
    synthetic_domain_data["patches"]["wall"].extend([
        {
            "name": "left-wall", "side": "left", "marker": 105,
            "intervals": [{"min": 10.0, "max": 10.5}],
        },
        {
            "name": "right-wall", "side": "right", "marker": 106,
            "intervals": [{"min": 11.5, "max": 13.0}],
        },
    ])
    geometry = _full_geometry(tmp_path, synthetic_domain_data)
    context = FiredrakeContext(geometry, 7, 3)
    mapper = FEFieldMapper(context, geometry)
    x, y = mapper.s_coords.T
    external = (
        np.isclose(x, context.xmin) | np.isclose(x, context.xmax)
        | np.isclose(y, context.ymin) | np.isclose(y, context.ymax)
    )
    inlet = np.isclose(x, context.xmin) & (y >= 10.5)
    outlet = np.isclose(x, context.xmax) & (y < 11.5)
    expected_walls = context._patch_selector(
        mapper.s_coords, geometry.spec.wall
    )
    np.testing.assert_array_equal(context.inlet_velocity_nodes, np.flatnonzero(inlet))
    np.testing.assert_array_equal(context.velocity_wall_nodes, np.flatnonzero(expected_walls))
    np.testing.assert_array_equal(
        context.velocity_boundary_nodes, np.flatnonzero(inlet | expected_walls)
    )
    assert np.all(context.ux_lift[inlet & ~expected_walls] == context.uin)
    assert np.all(context.ux_lift[expected_walls] == 0.0)
    assert np.all(context.velocity_mask[outlet & ~expected_walls] == 1.0)

    coordinates = SpatialCoordinate(context.mesh)
    context.p.interpolate(coordinates[0])
    raw = assemble(context.rx_form).dat.data_ro.copy()
    rx, _, _ = context.residual_vectors()
    assert np.max(np.abs(rx.dat.data_ro[context.velocity_boundary_nodes])) == 0.0
    outlet_nodes = np.flatnonzero(outlet)
    assert np.linalg.norm(raw[outlet_nodes]) > 0.0
    np.testing.assert_allclose(rx.dat.data_ro[outlet_nodes], raw[outlet_nodes], atol=1e-14)

    q_coordinates = context.q_coordinates
    q_outlet = (
        np.isclose(q_coordinates[:, 0], context.xmax)
        & (q_coordinates[:, 1] < 11.5)
    )
    np.testing.assert_array_equal(
        context.pressure_outlet_nodes, np.flatnonzero(q_outlet)
    )
    assert context.inlet_marker == 1 and context.outlet_marker == 2
    assert mapper.inlet_geometry_conflicts == 0
    enforce_inlet_geometry_compatibility(mapper)
    expected_first = context.velocity_constraints_at(mapper.first_points)
    expected_second = context.velocity_constraints_at(mapper.second_points)
    np.testing.assert_array_equal(mapper.first_velocity_mask, expected_first[0])
    np.testing.assert_array_equal(mapper.first_ux_lift, expected_first[1])
    np.testing.assert_array_equal(mapper.second_velocity_mask, expected_second[0])
    np.testing.assert_array_equal(mapper.second_ux_lift, expected_second[1])

    points = np.array([
        [context.xmin, 10.0],
        [context.xmin, 11.0],
        [context.xmax, 10.5],
        [context.xmax, 12.0],
    ])
    mask, lift = context.velocity_constraints_at(points)
    np.testing.assert_array_equal(mask, [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_array_equal(lift, [0.0, context.uin, 0.0, 0.0])


def test_reconstructed_patch_collection_must_stay_on_one_side(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["patches"]["inlet"].append({
        "name": "second-inlet", "side": "right", "marker": 105,
        "intervals": [{"min": 12.0, "max": 13.0}],
    })
    synthetic_domain_data["patches"]["outlet"][0]["intervals"] = [
        {"min": 10.0, "max": 11.0}
    ]
    synthetic_domain_data["patches"]["wall"].append({
        "name": "right-wall", "side": "right", "marker": 106,
        "intervals": [{"min": 11.0, "max": 12.0}],
    })
    geometry = _full_geometry(tmp_path, synthetic_domain_data)
    with pytest.raises(ValueError, match="inlet patches must all occupy one"):
        FiredrakeContext(geometry, 7, 3)
