"""Sanity check: verify IndependentPhiStateAssembler scaling is correct.

Tests:
  1. Zero network output => state == W0 (machine precision)
  2. Unit dimensionless output => physical corrections match expected scales
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

import torch
torch.set_default_dtype(torch.float64)

from common import PROJECT_ROOT  # noqa: E402
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.mesh_metadata import MeshMetadata  # noqa: E402
from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler  # noqa: E402
from state_layout import build_isothermal_layout  # noqa: E402


def main():
    ds_dir = Path(PROJECT_ROOT) / "datasets" / "four_port_64"
    mesh_meta = MeshMetadata.load(
        str(ds_dir / "shared" / "mesh_metadata.npz"),
        str(ds_dir / "shared" / "mesh_metadata.json"),
    )
    layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)

    n_cells = mesh_meta.n_cells
    n_faces = mesh_meta.n_faces
    n_internal = mesh_meta.n_internal_faces
    n_u = 3 * n_cells
    n_p = n_cells

    w0 = np.load(ds_dir / "shared" / "base_state_k0.npy")

    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=n_cells,
        n_faces=n_faces,
    )

    # Trainable phi: internal + outlets
    patch_names = list(mesh_meta.patch_names)
    phi_trainable = np.zeros(n_faces, dtype=bool)
    phi_trainable[:n_internal] = True
    for pname in ["outletLower", "outletUpper"]:
        if pname in patch_names:
            idx = patch_names.index(pname)
            start = int(mesh_meta.patch_start_faces[idx])
            count = int(mesh_meta.patch_face_counts[idx])
            phi_trainable[start:start + count] = True
    phi_trainable_indices = np.flatnonzero(phi_trainable)

    base_state = torch.from_numpy(w0)
    asm = IndependentPhiStateAssembler(
        base_state, layout, flux_asm, n_cells, n_faces,
        phi_trainable_indices)

    # ---- Test 1: Zero output => W0 ----
    cell_zero = torch.zeros(n_cells, 3, dtype=torch.float64)
    phi_zero = torch.zeros(len(phi_trainable_indices), dtype=torch.float64)
    w_zero = asm.assemble(cell_zero, phi_zero).detach().numpy()

    err_zero = np.linalg.norm(w_zero - w0) / (np.linalg.norm(w0) + 1e-30)
    print(f"Test 1 (zero output => W0):")
    print(f"  ||W - W0|| / ||W0|| = {err_zero:.2e}")
    assert err_zero < 1e-14, "Zero output must reproduce W0 to machine precision"
    print("  PASS")
    print()

    # ---- Test 2: Unit output => correct physical scales ----
    cell_unit = torch.zeros(n_cells, 3, dtype=torch.float64)
    cell_unit[:, 0] = 1.0  # q_Ux = 1
    cell_unit[:, 1] = 1.0  # q_Uy = 1
    cell_unit[:, 2] = 1.0  # q_p = 1
    phi_unit = torch.ones(len(phi_trainable_indices), dtype=torch.float64)

    w_unit = asm.assemble(cell_unit, phi_unit).detach().numpy()

    # Extract corrections
    delta_u = w_unit[:n_u] - w0[:n_u]
    delta_p = w_unit[n_u:n_u + n_p] - w0[n_u:n_u + n_p]
    delta_phi = w_unit[n_u + n_p:] - w0[n_u + n_p:]

    # Ux: should be U_SCALE * 1 = 0.1
    ux_corr = delta_u[0::3]
    uy_corr = delta_u[1::3]
    uz_corr = delta_u[2::3]

    print(f"Test 2 (unit output => physical scales):")
    print(f"  dUx: mean={ux_corr.mean():.6e}, expected=1.0e-01")
    print(f"  dUy: mean={uy_corr.mean():.6e}, expected=1.0e-01")
    print(f"  dUz: mean={uz_corr.mean():.6e}, expected=0.0e+00")
    print(f"  dp:  mean={delta_p.mean():.6e}, expected=1.0e-02")

    assert abs(ux_corr.mean() - 0.1) < 1e-10, f"dUx mean {ux_corr.mean()} != 0.1"
    assert abs(uy_corr.mean() - 0.1) < 1e-10, f"dUy mean {uy_corr.mean()} != 0.1"
    assert abs(uz_corr.mean()) < 1e-15, f"dUz mean {uz_corr.mean()} != 0"
    assert abs(delta_p.mean() - 0.01) < 1e-12, f"dp mean {delta_p.mean()} != 0.01"
    print("  Cell corrections PASS")

    # Phi: the independent correction part should be PHI_SCALE * 1 = 4e-7
    # on trainable faces. The interpolation part (A_phi * dU) is also present
    # on internal faces, so we check the outlet faces where only the
    # independent correction applies.
    for pname in ["outletLower", "outletUpper"]:
        if pname not in patch_names:
            continue
        idx = patch_names.index(pname)
        start = int(mesh_meta.patch_start_faces[idx])
        count = int(mesh_meta.patch_face_counts[idx])
        phi_corr_outlet = delta_phi[start:start + count]
        print(f"  dphi[{pname}]: mean={phi_corr_outlet.mean():.6e}, "
              f"expected=4.0e-07")
        assert abs(phi_corr_outlet.mean() - 4e-7) < 1e-12, \
            f"phi correction {phi_corr_outlet.mean()} != 4e-7"

    print("  Phi corrections PASS")
    print()
    print("All sanity checks PASSED")


if __name__ == "__main__":
    main()
