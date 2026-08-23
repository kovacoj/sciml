import numpy as np
from firedrake import DirichletBC, Function, SpatialCoordinate, assemble

from src.forms import FiredrakeContext
from src.geometry import TPFMGeometry


def _context(synthetic_geometry_path, nx=5, ny=5):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    return FiredrakeContext(geometry, nx, ny)


def _randomize(function, rng):
    function.dat.data[:] = rng.standard_normal(function.dat.data.shape)


def test_literal_momentum_cancellation_and_chi_zero_independence(synthetic_geometry_path):
    context = _context(synthetic_geometry_path)
    rng = np.random.default_rng(4)
    for field in (context.ux, context.uy, context.p):
        _randomize(field, rng)
    context.uibx.assign(context.ux)
    context.uiby.assign(context.uy)
    context.chi.assign(1.0)
    context.beta.assign(0.25)
    rx, ry, _ = context.residual_vectors()
    assert np.linalg.norm(rx.dat.data_ro) + np.linalg.norm(ry.dat.data_ro) < 1e-10

    context.chi.assign(0.0)
    before = [assemble(context.rx_form).dat.data_ro.copy(), assemble(context.ry_form).dat.data_ro.copy()]
    _randomize(context.uibx, rng)
    _randomize(context.uiby, rng)
    after = [assemble(context.rx_form).dat.data_ro, assemble(context.ry_form).dat.data_ro]
    assert max(np.max(np.abs(a - b)) for a, b in zip(after, before)) < 1e-14


def test_continuity_is_independent_of_chi_and_uib(synthetic_geometry_path):
    context = _context(synthetic_geometry_path)
    rng = np.random.default_rng(9)
    _randomize(context.ux, rng)
    _randomize(context.uy, rng)
    before = assemble(context.rc_form).dat.data_ro.copy()
    context.chi.assign(1.0)
    _randomize(context.uibx, rng)
    _randomize(context.uiby, rng)
    after = assemble(context.rc_form).dat.data_ro
    assert np.max(np.abs(after - before)) < 1e-14


def test_momentum_test_bcs_zero_inlet_walls_but_retain_outlet(synthetic_geometry_path):
    context = _context(synthetic_geometry_path)
    x = SpatialCoordinate(context.mesh)
    context.p.interpolate(x[0])
    raw = assemble(context.rx_form).dat.data_ro.copy()
    rx, _, _ = context.residual_vectors()
    outlet = np.asarray(DirichletBC(context.S, 0.0, 2).nodes)
    outlet_noncorner = np.setdiff1d(outlet, context.velocity_boundary_nodes)
    assert np.max(np.abs(rx.dat.data_ro[context.velocity_boundary_nodes])) == 0.0
    assert np.linalg.norm(raw[outlet_noncorner]) > 0.0
    assert np.max(np.abs(rx.dat.data_ro[outlet_noncorner] - raw[outlet_noncorner])) < 1e-14


def test_zero_state_has_zero_residual(synthetic_geometry_path):
    context = _context(synthetic_geometry_path, 3, 3)
    residuals = context.solve_riesz()
    assert context.raw_losses(residuals) == (0.0, 0.0, 0.0)
