"""Physical checks for Gate G: solid no-slip, source activation, mass
conservation, flow deflection, residual convergence.
"""
from __future__ import annotations

import numpy as np

from .geometry_manifest import GeometryManifest, solid_cell_ids, interface_cell_ids


def check_solid_noslip(state, manifest: GeometryManifest, u_in: float,
                       layout) -> dict:
    """max|U| in pure-solid cells must be < 1e-3 * U_in."""
    u_idx = layout.indices("U")
    u_vec = state[u_idx].reshape(-1, 3)  # cell-major [Ux,Uy,Uz] per cell
    speeds = np.linalg.norm(u_vec[:, :2], axis=1)  # 2D speed (skip z)
    solid_ids = solid_cell_ids(manifest)
    if not solid_ids:
        return {"max_solid_speed": -1.0, "pass": False,
                "error": "no solid cells"}
    max_solid = float(np.max(speeds[solid_ids]))
    return {
        "max_solid_speed": max_solid,
        "threshold": 1e-3 * u_in,
        "pass": max_solid < 1e-3 * u_in,
    }


def check_source_norm(r_hfdib, r_base, manifest: GeometryManifest,
                      layout) -> dict:
    """Source activation: |f_ib|_2 = |R_H - R_0|_2 restricted to U rows > 0."""
    u_idx = layout.indices("U")
    delta = r_hfdib - r_base
    source_l2 = float(np.linalg.norm(delta[u_idx]))
    # also check source is zero outside chi=1 cells
    chi_cells = [c for c in manifest.cells if c["chi"] > 0.0]
    non_chi_ids = [c["cell_id"] for c in manifest.cells if c["chi"] == 0.0]
    # per-cell U residual delta (3 per cell)
    delta_u = delta[u_idx].reshape(-1, 3)
    if non_chi_ids:
        max_nonchi = float(np.max(np.linalg.norm(
            delta_u[non_chi_ids], axis=1)))
    else:
        max_nonchi = 0.0
    return {
        "source_l2": source_l2,
        "max_source_outside_chi": max_nonchi,
        "pass": source_l2 > 0.0 and max_nonchi < 1e-10,
    }


def check_mass_imbalance(state, layout, n_internal_faces: int,
                          n_inlet_faces: int) -> dict:
    """Mass imbalance using boundary face fluxes from phi state block."""
    phi_idx = layout.indices("phi")
    phi = state[phi_idx]
    # boundary faces start after internal faces, patches in blockMesh order:
    # inlet, outlet, walls, frontAndBack
    inlet_start = n_internal_faces
    outlet_start = n_internal_faces + n_inlet_faces
    n_outlet = n_inlet_faces  # same count for this duct
    phi_in = float(np.sum(phi[inlet_start:inlet_start + n_inlet_faces]))
    phi_out = float(np.sum(phi[outlet_start:outlet_start + n_outlet]))
    imbalance = abs(phi_in + phi_out) / max(abs(phi_in), abs(phi_out), 1e-12)
    return {
        "phi_inlet": phi_in,
        "phi_outlet": phi_out,
        "mass_imbalance": imbalance,
        "pass": imbalance < 1e-6,
    }


def check_flow_deflection(state_hfdib, state_base, layout) -> dict:
    """HFDIB solution must differ from baseline."""
    u_idx = layout.indices("U")
    u_hfdib = state_hfdib[u_idx]
    u_base = state_base[u_idx]
    diff = float(np.linalg.norm(u_hfdib - u_base))
    base_norm = float(np.linalg.norm(u_base))
    rel_diff = diff / max(base_norm, 1e-12)
    # max cross-stream velocity (Uy)
    n_cells = u_hfdib.size // 3
    uy_hfdib = u_hfdib[1::3]  # Uy components (cell-major: Ux,Uy,Uz)
    max_uy = float(np.max(np.abs(uy_hfdib)))
    return {
        "relative_difference": rel_diff,
        "max_cross_stream_velocity": max_uy,
        "pass": rel_diff > 1e-3 and max_uy > 0.0,
    }


def check_residuals(r_hfdib, layout) -> dict:
    """Per-block residual norms."""
    u_idx = layout.indices("U")
    p_idx = layout.indices("p")
    phi_idx = layout.indices("phi")
    r_u = float(np.linalg.norm(r_hfdib[u_idx]))
    r_p = float(np.linalg.norm(r_hfdib[p_idx]))
    r_phi = float(np.linalg.norm(r_hfdib[phi_idx]))
    r_total = float(np.linalg.norm(r_hfdib))
    return {
        "residual_U": r_u,
        "residual_p": r_p,
        "residual_phi": r_phi,
        "residual_l2": r_total,
        "all_finite": bool(np.all(np.isfinite(r_hfdib))),
        "pass": bool(np.all(np.isfinite(r_hfdib))) and r_total < 1e-3,
    }


def run_all(state_hfdib, state_base, r_hfdib, r_base,
            manifest: GeometryManifest, layout,
            u_in: float, n_internal_faces: int, n_inlet_faces: int) -> dict:
    return {
        "solid_noslip": check_solid_noslip(state_hfdib, manifest, u_in, layout),
        "source_activation": check_source_norm(r_hfdib, r_base, manifest, layout),
        "mass_imbalance": check_mass_imbalance(state_hfdib, layout,
                                                n_internal_faces, n_inlet_faces),
        "flow_deflection": check_flow_deflection(state_hfdib, state_base, layout),
        "residuals": check_residuals(r_hfdib, layout),
    }
