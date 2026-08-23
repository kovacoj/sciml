import numpy as np
from firedrake import Function

from src.forms import FiredrakeContext
from src.geometry import TPFMGeometry
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
