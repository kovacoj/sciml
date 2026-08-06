"""Generate comparison plots: HFDIB vs supervised U-Net vs physics U-Net."""
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


def load_fields(dataset_dir: str, topology_id: str):
    """Load HFDIB reference fields."""
    topo_dir = Path(dataset_dir) / topology_id
    return {
        "lambda": np.load(topo_dir / "lambda.npy"),
        "ux_hfdib": np.load(topo_dir / "ux_hfdib.npy"),
        "uy_hfdib": np.load(topo_dir / "uy_hfdib.npy"),
        "p_hfdib": np.load(topo_dir / "pressure_hfdib.npy"),
    }


def predict_fields(model, lam, device="cpu"):
    """Run model and extract predicted fields."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x).squeeze(0).cpu().numpy()
    return {"ux": pred[0], "uy": pred[1], "p": pred[2]}


def plot_comparison(hfdib_fields, sup_pred, phys_pred, topology_id, output_dir):
    """Generate side-by-side comparison figure."""
    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle(f"Topology: {topology_id}", fontsize=16)

    # Row 0: velocity magnitude
    hfdib_speed = np.sqrt(hfdib_fields["ux_hfdib"]**2 + hfdib_fields["uy_hfdib"]**2)
    sup_speed = np.sqrt(sup_pred["ux"]**2 + sup_pred["uy"]**2)
    phys_speed = np.sqrt(phys_pred["ux"]**2 + phys_pred["uy"]**2)

    vmax = max(hfdib_speed.max(), sup_speed.max(), phys_speed.max())

    axes[0, 0].imshow(hfdib_fields["lambda"], origin="lower", cmap="gray", vmin=0, vmax=1)
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

    # Row 1: pressure
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


def compute_metrics(hfdib_fields, pred):
    """Compute error metrics."""
    hfdib_speed = np.sqrt(hfdib_fields["ux_hfdib"]**2 + hfdib_fields["uy_hfdib"]**2)
    pred_speed = np.sqrt(pred["ux"]**2 + pred["uy"]**2)

    rel_u = np.linalg.norm(pred_speed - hfdib_speed) / (np.linalg.norm(hfdib_speed) + 1e-12)
    rel_p = np.linalg.norm(pred["p"] - hfdib_fields["p_hfdib"]) / (np.linalg.norm(hfdib_fields["p_hfdib"]) + 1e-12)

    return {
        "rel_velocity_l2": float(rel_u),
        "rel_pressure_l2": float(rel_p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--supervised-ckpt", required=True)
    ap.add_argument("--physics-ckpt", required=True)
    ap.add_argument("--output", default="outputs/comparison")
    args = ap.parse_args()

    import torch
    from unet.factory import load_model_from_checkpoint
    from unet.boundary import BoundaryEnforcer

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = Path(PROJECT_ROOT) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models from checkpoints (architecture-aware)
    sup_model = load_model_from_checkpoint(args.supervised_ckpt)
    phys_model = load_model_from_checkpoint(args.physics_ckpt)

    bc = BoundaryEnforcer(64, 64)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir

    # Find all topology directories
    topo_dirs = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("topology_")])

    all_metrics = []
    for topo_dir in topo_dirs:
        tid = topo_dir.name
        fields = load_fields(str(dataset_dir), tid)

        lam = torch.from_numpy(fields["lambda"]).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            sup_pred_raw = sup_model(lam).squeeze(0).numpy()
            phys_pred_raw = phys_model(lam).squeeze(0).numpy()

        sup_pred = {"ux": sup_pred_raw[0], "uy": sup_pred_raw[1], "p": sup_pred_raw[2]}
        phys_pred = {"ux": phys_pred_raw[0], "uy": phys_pred_raw[1], "p": phys_pred_raw[2]}

        # Apply BCs
        sup_pred_t = torch.from_numpy(np.stack([sup_pred["ux"], sup_pred["uy"], sup_pred["p"]])).unsqueeze(0)
        phys_pred_t = torch.from_numpy(np.stack([phys_pred["ux"], phys_pred["uy"], phys_pred["p"]])).unsqueeze(0)
        lam_t = torch.from_numpy(fields["lambda"]).unsqueeze(0).unsqueeze(0)
        sup_pred_t = bc.apply(sup_pred_t, lam_t).squeeze(0).numpy()
        phys_pred_t = bc.apply(phys_pred_t, lam_t).squeeze(0).numpy()
        sup_pred = {"ux": sup_pred_t[0], "uy": sup_pred_t[1], "p": sup_pred_t[2]}
        phys_pred = {"ux": phys_pred_t[0], "uy": phys_pred_t[1], "p": phys_pred_t[2]}

        plot_comparison(fields, sup_pred, phys_pred, tid, output_dir)

        sup_metrics = compute_metrics(fields, sup_pred)
        phys_metrics = compute_metrics(fields, phys_pred)

        all_metrics.append({
            "topology_id": tid,
            "supervised": sup_metrics,
            "physics": phys_metrics,
        })

    # Save summary
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[plot] {len(all_metrics)} comparisons saved to {output_dir}")

    # Print summary table
    print("\nTopology | Sup rel_U | Sup rel_p | Phys rel_U | Phys rel_p")
    for m in all_metrics:
        print(f"{m['topology_id']} | {m['supervised']['rel_velocity_l2']:.4e} | "
              f"{m['supervised']['rel_pressure_l2']:.4e} | "
              f"{m['physics']['rel_velocity_l2']:.4e} | "
              f"{m['physics']['rel_pressure_l2']:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
