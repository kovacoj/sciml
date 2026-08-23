import numpy as np
import torch

from src.forms import FiredrakeContext
from src.geometry import TPFMGeometry
from src.model import CoordinateMLP
from src.state import FEFieldMapper, NeuralFields


def test_context_domain_spaces_and_exact_fe_boundary_constraints(synthetic_geometry_path):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(geometry, 6, 5)
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
