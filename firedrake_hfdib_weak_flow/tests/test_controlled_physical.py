import json
from pathlib import Path

import numpy as np
import pytest
from firedrake import Cofunction, Function, assemble

from src.forms import FiredrakeContext
from src.geometry_smoke import run as run_geometry_smoke
from src.model import CoordinateMLP
from src.physical_validation import validate_physical_domain
from src.state import FEFieldMapper


def test_controlled_physical_gate_counts_connectivity_and_aspect(controlled_setup):
    _, _, _, _, geometry, context, mapper = controlled_setup
    report = validate_physical_domain(geometry, context, mapper)
    assert report["domain_classification"] == "CONTROLLED_TPFM_DERIVED_DOMAIN"
    assert report["article_reproduction"] is False
    assert report["inlet_geometry_conflicts"] == 0
    assert report["outlet_geometry_conflicts"] == 0
    assert report["inlet_node_count"] > 0
    assert report["outlet_node_count"] > 0
    assert report["fluid_connected"] is True
    assert report["hx"] == pytest.approx(report["hy"])
    assert report["aspect_error"] <= 0.1
    assert report["hard_gate_passed"] is True


def test_controlled_40_by_20_aspect_passes(controlled_setup):
    _, _, _, _, geometry, _, _ = controlled_setup
    context = FiredrakeContext(
        geometry, 40, 20, nu=geometry.spec.nu,
        uin=geometry.spec.uin, pout=geometry.spec.pout,
    )
    mapper = FEFieldMapper(context, geometry)
    assert validate_physical_domain(geometry, context, mapper)["hard_gate_passed"]


def test_square_mesh_fails_unless_anisotropic_override(controlled_setup):
    _, _, _, _, geometry, _, _ = controlled_setup
    context = FiredrakeContext(geometry, 20, 20)
    mapper = FEFieldMapper(context, geometry)
    with pytest.raises(RuntimeError, match="mesh aspect"):
        validate_physical_domain(geometry, context, mapper)
    report = validate_physical_domain(
        geometry, context, mapper, allow_anisotropic_mesh=True
    )
    assert report["aspect_error"] > 0.1
    assert report["hard_gate_passed"] is True


def test_disconnected_fluid_fails(controlled_setup):
    _, _, _, _, geometry, context, mapper = controlled_setup
    geometry.lambda_field[:, geometry.left_extension_cells + 30] = 1.0
    with pytest.raises(RuntimeError, match="connectivity"):
        validate_physical_domain(geometry, context, mapper)


def test_nonfinite_geometry_fails(controlled_setup):
    _, _, _, _, geometry, context, mapper = controlled_setup
    geometry.normals[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="finite lambda/sigma/normals"):
        validate_physical_domain(geometry, context, mapper)


def test_controlled_operator_invariants_and_hard_boundary_conditions(controlled_setup):
    _, _, _, _, geometry, context, mapper = controlled_setup
    rng = np.random.default_rng(31)
    for field in (context.ux, context.uy, context.p):
        field.dat.data[:] = rng.standard_normal(field.dat.data.shape)
    context.uibx.assign(context.ux)
    context.uiby.assign(context.uy)
    context.chi.assign(1.0)
    context.beta.assign(0.25)
    rx, ry, _ = context.residual_vectors()
    assert np.linalg.norm(rx.dat.data_ro) + np.linalg.norm(ry.dat.data_ro) < 1e-10

    context.chi.assign(0.0)
    before = (assemble(context.rx_form).dat.data_ro.copy(), assemble(context.ry_form).dat.data_ro.copy())
    context.uibx.dat.data[:] = rng.standard_normal(context.uibx.dat.data.shape)
    context.uiby.dat.data[:] = rng.standard_normal(context.uiby.dat.data.shape)
    after = (assemble(context.rx_form).dat.data_ro, assemble(context.ry_form).dat.data_ro)
    assert max(np.max(np.abs(a - b)) for a, b in zip(after, before)) < 1e-14

    context.ux.assign(0.0)
    context.uy.assign(0.0)
    context.p.assign(0.0)
    context.uibx.assign(0.0)
    context.uiby.assign(0.0)
    assert context.raw_losses(context.solve_riesz()) == (0.0, 0.0, 0.0)

    rhs = Cofunction(context.S.dual())
    rhs.dat.data[:] = rng.standard_normal(context.S.dim())
    rhs.dat.data[context.velocity_boundary_nodes] = 0.0
    solution = Function(context.S)
    context.h1_solver.solve(solution, rhs)
    assert np.max(np.abs(solution.dat.data_ro[context.velocity_boundary_nodes])) < 1e-13
    with solution.dat.vec_ro as vector:
        work = context.h1_matrix.petscmat.createVecLeft()
        context.h1_matrix.petscmat.mult(vector, work)
        assert vector.dot(work) > 0.0

    model = CoordinateMLP(input_dim=4, width=8, depth=2)
    fields = mapper.evaluate(model)
    assert np.max(np.abs(
        fields.ux.detach().numpy()[context.inlet_velocity_nodes] - context.uin
    )) < 1e-14
    assert np.max(np.abs(fields.uy.detach().numpy()[context.velocity_boundary_nodes])) < 1e-14
    assert np.max(np.abs(fields.p.detach().numpy()[context.pressure_outlet_nodes])) < 1e-14


def test_geometry_smoke_writes_controlled_artifacts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = run_geometry_smoke(
        root / "configs/controlled_geometry_smoke.json", tmp_path
    )
    assert report == json.loads((tmp_path / "geometry_compatibility.json").read_text())
    assert (tmp_path / "geometry_controlled.png").stat().st_size > 0
    copied = json.loads((tmp_path / "controlled_domain_spec.json").read_text())
    assert copied["classification"] == "CONTROLLED_TPFM_DERIVED_DOMAIN"
