"""Shared construction and field metrics for DAFoam warm-start studies."""
from __future__ import annotations

import time

import numpy as np


def build_state_assembler(mesh_meta, layout, base_state):
    import torch
    from pinn.flux_assembly import FluxAssembler
    from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler

    phi_trainable = np.zeros(mesh_meta.n_faces, dtype=bool)
    phi_trainable[:mesh_meta.n_internal_faces] = True
    patch_names = list(mesh_meta.patch_names)
    for name in ("outletLower", "outletUpper"):
        index = patch_names.index(name)
        start = int(mesh_meta.patch_start_faces[index])
        count = int(mesh_meta.patch_face_counts[index])
        phi_trainable[start:start + count] = True
    indices = np.flatnonzero(phi_trainable)
    flux_assembler = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=mesh_meta.n_cells,
        n_faces=mesh_meta.n_faces,
    )
    assembler = IndependentPhiStateAssembler(
        torch.from_numpy(base_state), layout, flux_assembler,
        mesh_meta.n_cells, mesh_meta.n_faces, indices,
    )
    return assembler, indices


def predict_full_state(model, topology, state_assembler):
    """Use the exact projection and state assembly from held-out evaluation."""
    import torch
    from unet.train import project_solid_velocity

    lam_t = torch.from_numpy(topology).unsqueeze(0).unsqueeze(0)
    started = time.perf_counter()
    with torch.no_grad():
        cell_pred, phi_pred = model(lam_t)
        cell_pred = project_solid_velocity(cell_pred, lam_t)
        corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        state = state_assembler.assemble(
            corrections, phi_pred.squeeze(0)).detach().numpy()
    return np.ascontiguousarray(state, dtype=np.float64), time.perf_counter() - started


def relative_field_errors(state, reference, n_cells):
    n_u = 3 * n_cells
    u = state[:n_u].reshape(n_cells, 3)[:, :2]
    u_ref = reference[:n_u].reshape(n_cells, 3)[:, :2]
    p = state[n_u:n_u + n_cells]
    p_ref = reference[n_u:n_u + n_cells]
    p = p - np.mean(p)
    p_ref = p_ref - np.mean(p_ref)
    return {
        "rel_u": float(np.linalg.norm(u - u_ref) / (np.linalg.norm(u_ref) + 1e-30)),
        "rel_p": float(np.linalg.norm(p - p_ref) / (np.linalg.norm(p_ref) + 1e-30)),
    }
