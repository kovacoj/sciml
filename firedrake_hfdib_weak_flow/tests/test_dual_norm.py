import numpy as np
from firedrake import Cofunction, Function

from src.domain import DomainSpec
from src.forms import FiredrakeContext
from src.geometry import FullDomainGeometry, TPFMGeometry
from src.residual_bridge import FiredrakeResidualBridge
from src.state import FEFieldMapper
from src.model import CoordinateMLP


def test_gram_positivity_lu_residual_and_separate_baselines(synthetic_geometry_path):
    path, spacing = synthetic_geometry_path
    geometry = TPFMGeometry(path, spacing=spacing)
    context = FiredrakeContext(geometry, 5, 4)
    rng = np.random.default_rng(8)
    x = Function(context.S)
    x.dat.data[:] = rng.standard_normal(x.dat.data.shape)
    matrix = context.h1_matrix.petscmat
    with x.dat.vec_ro as vector:
        work = matrix.createVecLeft()
        matrix.mult(vector, work)
        assert vector.dot(work) > 0.0

    context.ux.dat.data[:] = rng.standard_normal(context.ux.dat.data.shape)
    rx, _, _ = context.solve_riesz()
    with context.yx.dat.vec_ro as y, rx.dat.vec_ro as b:
        residual = b.duplicate()
        matrix.mult(y, residual)
        residual.axpy(-1.0, b)
        assert residual.norm() / b.norm() < 1e-10

    fields = FEFieldMapper(context, geometry).evaluate(
        CoordinateMLP(input_dim=4, width=8, depth=2)
    )
    bridge = FiredrakeResidualBridge(context, gamma=0.7)
    Cm, Cc = bridge.initialize_normalization(fields)
    assert Cm >= 1e-12 and Cc >= 1e-12
    assert float(context.inv_Cm) == 1.0 / Cm
    assert float(context.inv_Cc) == 1.0 / Cc
    assert float(context.gamma_fd) == 0.7


def test_segmented_gram_eliminates_exact_essential_nodes(
    tmp_path, synthetic_domain_data
):
    synthetic_domain_data["patches"]["inlet"][0]["intervals"] = [
        {"min": 10.5, "max": 13.0}
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
    roi_path = tmp_path / "segmented_gram_roi.npz"
    np.savez(roi_path, inputs=np.zeros((1, 1, 3, 4)))
    geometry = FullDomainGeometry(
        TPFMGeometry(roi_path, spacing=1.0),
        DomainSpec.parse(synthetic_domain_data),
    )
    context = FiredrakeContext(geometry, 7, 3)
    rng = np.random.default_rng(29)
    rhs = Cofunction(context.S.dual())
    rhs.dat.data[:] = rng.standard_normal(context.S.dim())
    rhs.dat.data[context.velocity_boundary_nodes] = 0.0
    solution = Function(context.S)
    context.h1_solver.solve(solution, rhs)

    assert np.max(np.abs(
        solution.dat.data_ro[context.velocity_boundary_nodes]
    )) < 1.0e-13
    matrix = context.h1_matrix.petscmat
    with solution.dat.vec_ro as vector, rhs.dat.vec_ro as right_hand_side:
        residual = right_hand_side.duplicate()
        matrix.mult(vector, residual)
        residual.axpy(-1.0, right_hand_side)
        assert residual.norm() / right_hand_side.norm() < 1.0e-10
