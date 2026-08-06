"""Convert topology bitmaps to signed-distance fields."""
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt


def mask_to_signed_distance(
    mask: np.ndarray,
    cell_to_grid: np.ndarray,
    grid_shape: tuple[int, int],
    design_bounds: tuple[float, float, float, float],
    domain_bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Convert a topology bitmap to a per-cell signed-distance field.

    mask: [ny_mask, nx_mask] with 1=solid, 0=fluid
    cell_to_grid: [n_cells, 2] = (row, col) into grid_shape
    grid_shape: (H, W) of the CFD grid
    design_bounds: (x_min, x_max, y_min, y_max) of the design region
    domain_bounds: (x_min, x_max, y_min, y_max) of the full domain

    Returns: [n_cells] signed distance (psi > 0 fluid, psi < 0 solid)
    """
    nx, ny = grid_shape[1], grid_shape[0]
    dx = (domain_bounds[1] - domain_bounds[0]) / nx
    dy = (domain_bounds[3] - domain_bounds[2]) / ny

    # Build the solid mask on the CFD grid
    solid_grid = np.zeros(grid_shape, dtype=bool)

    # Map design region to grid indices
    des_x_min, des_x_max, des_y_min, des_y_max = design_bounds
    i_start = int(round((des_x_min - domain_bounds[0]) / dx))
    i_end = int(round((des_x_max - domain_bounds[0]) / dx))
    j_start = int(round((des_y_min - domain_bounds[2]) / dy))
    j_end = int(round((des_y_max - domain_bounds[2]) / dy))

    # Upsample mask to CFD grid
    mask_ny, mask_nx = mask.shape
    cfd_ny = j_end - j_start
    cfd_nx = i_end - i_start
    for j in range(cfd_ny):
        for i in range(cfd_nx):
            mj = min(int(j * mask_ny / cfd_ny), mask_ny - 1)
            mi = min(int(i * mask_nx / cfd_nx), mask_nx - 1)
            solid_grid[j_start + j, i_start + i] = (mask[mj, mi] == 1)

    # Initialize psi to a large positive value (deep fluid) everywhere,
    # so cells outside the design region are classified as pure fluid.
    h_inplane = np.sqrt(dx * dy)
    psi_grid = np.full(grid_shape, 10.0 * h_inplane, dtype=np.float64)

    # Compute signed distance only within the design region
    # (psi_grid already initialized to large positive value outside)
    des_slice = (slice(j_start, j_end), slice(i_start, i_end))
    dist_to_solid = distance_transform_edt(~solid_grid[des_slice], sampling=(dy, dx))
    dist_to_fluid = distance_transform_edt(solid_grid[des_slice], sampling=(dy, dx))
    psi_grid[des_slice] = dist_to_solid - dist_to_fluid

    # Flatten to OpenFOAM cell ordering
    n_cells = cell_to_grid.shape[0]
    psi_cells = np.empty(n_cells)
    for cell_id in range(n_cells):
        row, col = cell_to_grid[cell_id]
        psi_cells[cell_id] = psi_grid[row, col]

    return psi_cells


def compute_geometry_fields(psi: np.ndarray, h: float) -> dict:
    """Compute lambda, chi, interface mask from signed distance."""
    chi = (psi < 0.0).astype(np.float64)
    interface = (np.abs(psi) <= 1.5 * h).astype(np.float64)
    lam = 0.5 * (1.0 - np.tanh(psi / (1.5 * h)))
    return {"psi": psi, "chi": chi, "lambda": lam, "interface": interface}


def write_openfoam_scalar_list(path: Path, values: np.ndarray,
                                object_name: str = "signedDistance") -> None:
    """Write values as an OpenFOAM scalarList file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       scalarList;\n")
        f.write(f"    object      {object_name};\n")
        f.write("}\n\n")
        f.write(f"{len(values)}\n(\n")
        for v in values:
            f.write(f"{v:.16e}\n")
        f.write(")\n")
