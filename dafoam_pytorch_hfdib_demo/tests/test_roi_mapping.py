import numpy as np

from tpfm_reference.full_domain_case import mesh_metadata
from tpfm_reference.roi_mapping import (
    build_roi_cell_indices,
    extract_roi_pressure,
    extract_roi_velocity,
)


def test_manufactured_cell_ids_extract_exact_roi_order():
    metadata = mesh_metadata(8)
    indices = build_roi_cell_indices(metadata, metadata["roi_bounds"])
    nx = metadata["nx"]
    expected = np.array([[j * nx + i for i in range(8, 72)] for j in range(64)])
    np.testing.assert_array_equal(indices.reshape(64, 64), expected)

    ids = np.arange(metadata["n_cells"], dtype=float)
    velocity = np.column_stack((ids, ids + 0.25, ids + 0.5))
    pressure = ids + 1000
    extracted_u = extract_roi_velocity(velocity, indices)
    extracted_p = extract_roi_pressure(pressure, indices)
    np.testing.assert_array_equal(extracted_u[0], expected)
    np.testing.assert_array_equal(extracted_u[1], expected + 0.25)
    np.testing.assert_array_equal(extracted_p, expected + 1000)
