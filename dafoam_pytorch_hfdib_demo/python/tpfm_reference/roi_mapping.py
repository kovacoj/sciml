"""Coordinate-based mapping from a full CFD mesh to the published TPFM ROI."""
from __future__ import annotations

import numpy as np


def _centres(mesh_metadata) -> np.ndarray:
    if isinstance(mesh_metadata, dict):
        return np.asarray(mesh_metadata["cell_centres"])
    return np.asarray(mesh_metadata.cell_centres)


def build_roi_cell_indices(mesh_metadata, roi_bounds) -> np.ndarray:
    """Return full-state IDs in TPFM row-major (y, then x) array order."""
    centres = _centres(mesh_metadata)
    xmin, xmax, ymin, ymax = map(float, roi_bounds)
    tolerance = 1e-12
    selected = np.flatnonzero(
        (centres[:, 0] >= xmin - tolerance)
        & (centres[:, 0] < xmax - tolerance)
        & (centres[:, 1] >= ymin - tolerance)
        & (centres[:, 1] < ymax - tolerance)
    )
    order = np.lexsort((centres[selected, 0], centres[selected, 1]))
    indices = selected[order]
    if indices.size != 64 * 64:
        raise ValueError(f"ROI has {indices.size} cells, expected 4096")
    xy = centres[indices, :2].reshape(64, 64, 2)
    if not (np.all(np.diff(xy[:, :, 0], axis=1) > 0)
            and np.all(np.diff(xy[:, :, 1], axis=0) > 0)):
        raise ValueError("ROI coordinate ordering is not monotonic")
    return indices


def extract_roi_velocity(full_state, roi_indices) -> np.ndarray:
    state = np.asarray(full_state["U"] if isinstance(full_state, dict) else full_state)
    indices = np.asarray(roi_indices, dtype=np.int64)
    if state.ndim != 2 or state.shape[1] < 2:
        raise ValueError("full_state must be {'U': [nCells,3]} or a velocity array")
    return state[indices, :2].reshape(64, 64, 2).transpose(2, 0, 1)


def extract_roi_pressure(full_state, roi_indices) -> np.ndarray:
    state = np.asarray(full_state["p"] if isinstance(full_state, dict) else full_state)
    indices = np.asarray(roi_indices, dtype=np.int64)
    if state.ndim != 1:
        raise ValueError("full_state must be {'p': [nCells]} or a pressure array")
    return state[indices].reshape(64, 64)
