"""Generate comparison plots: HFDIB vs supervised U-Net vs physics NN.

Uses IndependentPhiStateAssembler for physics model inference —
the same assembly path as training. Metrics are vector L2 errors
matching the trainer's [field] diagnostic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import PROJECT_ROOT  # noqa: E402


def build_physics_assembler(dataset_dir, w0, mesh_meta):
    """Build IndependentPhiStateAssembler matching training."""
    import torch
    from pinn.flux_assembly import FluxAssembler
    from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
    from state_layout import build_isothermal_layout

    layout = build_isothermal_layout(mesh_meta.n_cells, mesh_meta.n_faces)

    patch_names = list(mesh_meta.patch_names)
    phi_trainable = np.zeros(mesh_meta.n_faces, dtype=bool)
    phi_trainable[:mesh_meta.n_internal_faces] = True
    for pname in ["outletLower", "outletUpper"]:
        if pname in patch_names:
            idx = patch_names.index(pname)
            start = int(mesh_meta.patch_start_faces[idx])
            count = int(mesh_meta.patch_face_counts[idx])
            phi_trainable[start:start + count] = True
    phi_trainable_indices = np.flatnonzero(phi_trainable)

    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=mesh_meta.n_cells,
        n_faces=mesh_meta.n_faces,
    )

    base_state = torch.from_numpy(w0)
    asm = IndependentPhiStateAssembler(
        base_state, layout, flux_asm,
        mesh_meta.n_cells, mesh_meta.n_faces,
        phi_trainable_indices)

    return asm


def predict_physics_fields(model, lam, state_asm, n_cells):
    """Run physics model through proper state assembly."""
    import torch
    from unet.train import project_solid_velocity

    with torch.no_grad():
        cell_pred, phi_pred = model(lam)
        cell_pred = project_solid_velocity(cell_pred, lam)
        corrections = cell_pred.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        phi_corr = phi_pred.squeeze(0)
        w = state_asm.assemble(corrections, phi_corr)

    n_u = 3 * n_cells
    u_cells = w[:n_u].reshape(n_cells, 3).cpu().numpy()
    p_cells = w[n_u:n_u + n_cells].cpu().numpy()

    return {
        "ux": u_cells[:, 0].reshape(64, 64),
        "uy": u_cells[:, 1].reshape(64, 64),
        "p": p_cells.reshape(64, 64),
    }


def predict_supervised_fields(model, lam):
    """Run supervised model (single output tensor)."""
    import torch
    with torch.no_grad():
        pred = model(lam).squeeze(0).numpy()
    return {"ux": pred[0], "uy": pred[1], "p": pred[2]}


def compute_vector_rel_u(pred, ref_ux, ref_uy):
    """Vector L2 relative error for velocity."""
    du = pred["ux"] - ref_ux
    dv = pred["uy"] - ref_uy
    return float(np.sqrt(
        np.sum(du**2) + np.sum(dv**2)
    ) / (
        np.sqrt(np.sum(ref_ux**2) + np.sum(ref_uy**2)) + 1e-30
    ))


def compute_rel_p(pred, ref_p):
    """L2 relative error for pressure."""
    return float(np.sqrt(np.sum((pred["p"] - ref_p)**2)) / (
        np.sqrt(np.sum(ref_p**2)) + 1e-30
    ))


def load_fields(dataset_dir, topology_id):
    """Load HFDIB reference fields."""
    topo_dir = Path(dataset_dir) / topology_id
    return {
        "lambda": np.load(topo_dir / "lambda.npy"),
        "ux_hfdib": np.load(topo_dir / "ux_hfdib.npy"),
        "uy_hfdib": np.load(topo_dir / "uy_hfdib.npy"),
        "p_hfdib": np.load(topo_dir / "pressure_hfdib.npy"),
    }


def plot_comparison(hfdib_fields, sup_pred, phys_pred, topology_id, output_dir):
    """Generate side-by-side comparison figure."""
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle(f"Topology: {topology_id}", fontsize=16)

    hfdib_speed = np.sqrt(hfdib_fields["ux_hfdib"]**2 + hfdib_fields["uy_hfdib"]**2)
    sup_speed = np.sqrt(sup_pred["ux"]**2 + sup_pred["uy"]**2)
    phys_speed = np.sqrt(phys_pred["ux"]**2 + phys_pred["uy"]**2)

    vmax = max(hfdib_speed.max(), sup_speed.max(), phys_speed.max())

    axes[0, 0].imshow(hfdib_fields["lambda"], origin="lower", cmap="gray_r", vmin=0, vmax=1)
    axes[0, 0].set_title("Topology (lambda)")
    axes[0, 1].imshow(hfdib_speed, origin="lower", cmap="coolwarm", vmin=0, vmax=vmax)
    axes[0, 1].set_title("HFDIB |U|")
    axes[0, 2].imshow(sup_speed, origin="lower", cmap="coolwarm", vmin=0, vmax=vmax)
    axes[0, 2].set_title("Supervised |U|")
    axes[0, 3].imshow(phys_speed, origin="lower", cmap="coolwarm", vmin=0, vmax=vmax)
    axes[0, 3].set_title("Physics |U|")
    axes[0, 4].imshow(np.abs(sup_speed - hfdib_speed), origin="lower", cmap="hot")
    axes[0, 4].set_title("Supervised |U| error")
    axes[0, 5].imshow(np.abs(phys_speed - hfdib_speed), origin="lower", cmap="hot")
    axes[0, 5].set_title("Physics |U| error")

    pmax = max(abs(hfdib_fields["p_hfdib"]).max(),
              abs(sup_pred["p"]).max(), abs(phys_pred["p"]).max())

    axes[1, 0].axis("off")
    im1 = axes[1, 1].imshow(hfdib_fields["p_hfdib"], origin="lower", cmap="RdBu_r",
                             vmin=-pmax, vmax=pmax)
    axes[1, 1].set_title("HFDIB p")
    axes[1, 2].imshow(sup_pred["p"], origin="lower", cmap="RdBu_r", vmin=-pmax, vmax=pmax)
    axes[1, 2].set_title("Supervised p")
    axes[1, 3].imshow(phys_pred["p"], origin="lower", cmap="RdBu_r", vmin=-pmax, vmax=pmax)
    axes[1, 3].set_title("Physics p")
    axes[1, 4].imshow(np.abs(sup_pred["p"] - hfdib_fields["p_hfdib"]), origin="lower", cmap="hot")
    axes[1, 4].set_title("Supervised p error")
    axes[1, 5].imshow(np.abs(phys_pred["p"] - hfdib_fields["p_hfdib"]), origin="lower", cmap="hot")
    axes[1, 5].set_title("Physics p error")

    plt.tight_layout()
    out_path = Path(output_dir) / f"comparison_{topology_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--supervised-ckpt", required=True)
    ap.add_argument("--physics-ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()

    import torch
    torch.set_default_dtype(torch.float64)
    from unet.factory import load_model_from_checkpoint
    from pinn.mesh_metadata import MeshMetadata

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(PROJECT_ROOT) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir

    # Load shared data for physics model assembly
    mesh_meta = MeshMetadata.load(
        str(dataset_dir / "shared" / "mesh_metadata.npz"),
        str(dataset_dir / "shared" / "mesh_metadata.json"),
    )
    w0 = np.load(dataset_dir / "shared" / "base_state_k0.npy")
    state_asm = build_physics_assembler(dataset_dir, w0, mesh_meta)
    n_cells = mesh_meta.n_cells

    # Load models
    sup_model = load_model_from_checkpoint(args.supervised_ckpt)
    phys_model = load_model_from_checkpoint(args.physics_ckpt)
    phys_model.eval()
    sup_model.eval()

    # Find topologies
    splits_path = dataset_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
        valid_ids = set(splits.get(args.split, []))
        topo_dirs = sorted([d for d in dataset_dir.iterdir()
                           if d.is_dir() and d.name.startswith("topology_")
                           and d.name in valid_ids])
    else:
        topo_dirs = sorted([d for d in dataset_dir.iterdir()
                           if d.is_dir() and d.name.startswith("topology_")])

    all_metrics = []
    for topo_dir in topo_dirs:
        tid = topo_dir.name
        fields = load_fields(str(dataset_dir), tid)

        lam = torch.from_numpy(fields["lambda"]).unsqueeze(0).unsqueeze(0)

        # Supervised model (single output, no state assembly)
        sup_pred = predict_supervised_fields(sup_model, lam)

        # Physics model (through IndependentPhiStateAssembler)
        phys_pred = predict_physics_fields(phys_model, lam, state_asm, n_cells)

        plot_comparison(fields, sup_pred, phys_pred, tid, output_dir)

        sup_rel_u = compute_vector_rel_u(sup_pred, fields["ux_hfdib"], fields["uy_hfdib"])
        sup_rel_p = compute_rel_p(sup_pred, fields["p_hfdib"])
        phys_rel_u = compute_vector_rel_u(phys_pred, fields["ux_hfdib"], fields["uy_hfdib"])
        phys_rel_p = compute_rel_p(phys_pred, fields["p_hfdib"])

        all_metrics.append({
            "topology_id": tid,
            "supervised": {"rel_velocity_l2": sup_rel_u, "rel_pressure_l2": sup_rel_p},
            "physics": {"rel_velocity_l2": phys_rel_u, "rel_pressure_l2": phys_rel_p},
        })

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[plot] {len(all_metrics)} comparisons saved to {output_dir}")

    print("\nTopology | Sup rel_U | Sup rel_p | Phys rel_U | Phys rel_p")
    for m in all_metrics:
        print(f"{m['topology_id']} | {m['supervised']['rel_velocity_l2']:.4e} | "
              f"{m['supervised']['rel_pressure_l2']:.4e} | "
              f"{m['physics']['rel_velocity_l2']:.4e} | "
              f"{m['physics']['rel_pressure_l2']:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
