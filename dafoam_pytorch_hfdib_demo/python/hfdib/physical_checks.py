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


def check_source_activation(bridge_hfdib, bridge_base, w, manifest, layout):
    """Source activation: evaluate R_H(W) - R_0(W) at the SAME state."""
    r_h = bridge_hfdib.residual(w)
    r_0 = bridge_base.residual(w)
    u_idx = layout.indices("U")
    delta = r_h - r_0
    source_l2 = float(np.linalg.norm(delta[u_idx]))
    # check source is zero outside chi>0 cells
    non_chi = [c["cell_id"] for c in manifest.cells if c["chi"] == 0.0]
    if non_chi:
        delta_u = delta[u_idx].reshape(-1, 3)
        max_nonchi = float(np.max(np.linalg.norm(delta_u[non_chi], axis=1)))
    else:
        max_nonchi = 0.0
    return {
        "source_l2": source_l2,
        "max_source_outside_chi": max_nonchi,
        "pass": source_l2 > 0.0 and max_nonchi < 1e-10,
    }


def check_interface_velocity(state, manifest, layout, u_in):
    """Compare U at interface cells against the HFDIB-imposed Uib.

    Uib is computed from the manifest stencils: u_ib = coeff * Σ w * U[src]
    """
    u_idx = layout.indices("U")
    u_vec = state[u_idx].reshape(-1, 3)  # cell-major
    iface_ids = [c["cell_id"] for c in manifest.cells if c["is_interface"]]
    if not iface_ids:
        return {"error": 0.0, "pass": False, "reason": "no interface cells"}

    total_err_sq = 0.0
    for s in manifest.stencils:
        cid = s["cell_id"]
        cell_geom = manifest.cells[cid]
        coeff = cell_geom["coeff"]
        u_ib = np.zeros(2)  # 2D (x,y)
        for j, src in enumerate(s["source_cells"]):
            w = s["source_weights"][j]
            u_ib[:2] += w * u_vec[src, :2]
        u_ib *= coeff
        u_cell = u_vec[cid, :2]
        total_err_sq += np.sum((u_cell - u_ib) ** 2)

    error = float(np.sqrt(total_err_sq) / (u_in * np.sqrt(len(iface_ids)) + 1e-30))
    return {"error": error, "pass": error < 1e-3}


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


def run_all(state_hfdib, state_base, r_hfdib_at_whfdib, r_base_at_whfdib,
            manifest: GeometryManifest, layout,
            u_in: float, n_internal_faces: int, n_inlet_faces: int) -> dict:
    """Physical checks using arrays only (no bridge objects needed).

    r_hfdib_at_whfdib: R_H(W_H) evaluated at the HFDIB converged state
    r_base_at_whfdib: R_0(W_H) evaluated at the SAME state (source isolation)
    """
    return {
        "solid_noslip": check_solid_noslip(state_hfdib, manifest, u_in, layout),
        "interface_velocity": check_interface_velocity(state_hfdib, manifest, layout, u_in),
        "source_activation": {
            "source_l2": float(np.linalg.norm(
                (r_hfdib_at_whfdib - r_base_at_whfdib)[layout.indices("U")])),
            "pass": float(np.linalg.norm(
                (r_hfdib_at_whfdib - r_base_at_whfdib)[layout.indices("U")])) > 0.0,
        },
        "mass_imbalance": check_mass_imbalance(state_hfdib, layout,
                                                n_internal_faces, n_inlet_faces),
        "flow_deflection": check_flow_deflection(state_hfdib, state_base, layout),
        "residuals": check_residuals(r_hfdib_at_whfdib, layout),
    }
