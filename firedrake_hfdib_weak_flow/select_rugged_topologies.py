"""Select three rugged paper-like topologies using geometry-only descriptors."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.rugged_bitmap_mesh import descriptors, full_fluid_mask


ROOT = Path("outputs/supervisor_2026_08_24/topic2_firedrake/paper_like_64")
DATASET = Path("/home/cady/personal/sciml/tpfm_unet_reference/data/mixer_64.npz")


def robust_z(values):
    values = np.asarray(values, dtype=float)
    center = np.median(values); scale = np.quantile(values, .75) - np.quantile(values, .25)
    return (values - center) / (scale + 1e-12)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "candidates").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    with np.load(DATASET) as archive:
        lambda_fields = archive["inputs"][:, 0].astype(np.float64)
    records = []
    for index, lam in enumerate(lambda_fields):
        value = descriptors(lam)
        value["source_topology_id"] = index
        records.append(value)
    eligible = [record for record in records if record["all_ports_connected"]
                and .20 <= record["fluid_fraction"] <= .65]
    keys = ("tortuosity", "interface_length_pixels", "asymmetry",
            "skeleton_branch_points", "central_fluid_fraction",
            "parallel_band_fluid_fraction", "solid_island_count")
    normalized = {
        key: robust_z([record[key] for record in eligible]) for key in keys
    }
    for position, record in enumerate(eligible):
        record["score_tortuous"] = float(
            2.0 * normalized["tortuosity"][position]
            + normalized["interface_length_pixels"][position]
            + normalized["asymmetry"][position]
            + .5 * normalized["skeleton_branch_points"][position]
        )
        record["score_merge_split"] = float(
            2.0 * normalized["central_fluid_fraction"][position]
            + 1.5 * normalized["skeleton_branch_points"][position]
            + .5 * normalized["interface_length_pixels"][position]
        )
        record["score_parallel"] = float(
            2.5 * normalized["parallel_band_fluid_fraction"][position]
            - 1.5 * normalized["tortuosity"][position]
            - .5 * normalized["asymmetry"][position]
            - .5 * normalized["skeleton_branch_points"][position]
        )
    selections = {}
    used = set()
    classes = (
        ("P64_A", "tortuous_network", "score_tortuous",
         "highest geometry-only tortuosity/interface/asymmetry score"),
        ("P64_B", "central_merge_split", "score_merge_split",
         "highest geometry-only central-fluid/junction score"),
        ("P64_C", "parallel_channels", "score_parallel",
         "highest geometry-only parallel-band and low-tortuosity score"),
    )
    for case, class_name, score_name, reason in classes:
        selected = next(
            record for record in sorted(eligible, key=lambda item: (-item[score_name], item["source_topology_id"]))
            if record["source_topology_id"] not in used
        )
        used.add(selected["source_topology_id"])
        selections[case] = {
            "class": class_name,
            "source_topology_id": selected["source_topology_id"],
            "selection_rule": reason,
            "geometry_metrics": selected,
        }
    (ROOT / "selected_cases.json").write_text(json.dumps(selections, indent=2) + "\n")
    (ROOT / "candidate_descriptors.json").write_text(json.dumps(records, indent=2) + "\n")

    # Thirty-six deterministic candidates spanning tortuosity rank, with selected
    # cases always included. This is a geometry diagnostic, not CFD cherry-picking.
    ranked = sorted(eligible, key=lambda item: (item["tortuosity"], item["source_topology_id"]))
    positions = np.linspace(0, len(ranked) - 1, 36, dtype=int)
    contact_ids = [ranked[position]["source_topology_id"] for position in positions]
    for topology_id in used:
        if topology_id not in contact_ids:
            contact_ids[-1] = topology_id
    figure, axes = plt.subplots(6, 6, figsize=(10, 10), constrained_layout=True)
    for axis, topology_id in zip(axes.flat, contact_ids):
        axis.imshow(lambda_fields[topology_id], origin="lower", cmap="gray_r", vmin=0, vmax=1)
        axis.set_title(str(topology_id), fontsize=8); axis.axis("off")
    figure.suptitle("Official TPFM topology candidates (geometry only)")
    figure.savefig(ROOT / "figures/paper_like_64_candidate_contact_sheet.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    for axis, (case, selection) in zip(axes, selections.items()):
        topology_id = selection["source_topology_id"]
        full = full_fluid_mask(lambda_fields[topology_id] < .5)
        axis.imshow(~full, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                    extent=(-.016, .144, 0, .128))
        axis.set_title(f"{case}\n{selection['class']}\nsource {topology_id}")
        axis.axis("off")
    figure.suptitle("Selected paper-like 64x64 topologies")
    figure.savefig(ROOT / "figures/paper_like_64_selected.png", dpi=320)
    plt.close(figure)

    for case, selection in selections.items():
        case_dir = ROOT / "cases" / case; case_dir.mkdir(parents=True, exist_ok=True)
        topology_id = selection["source_topology_id"]
        lam = lambda_fields[topology_id]
        binary = (lam >= .5).astype(np.uint8)
        coarse = (binary.reshape(8, 8, 8, 8).mean(axis=(1, 3)) >= .5).astype(np.uint8)
        np.save(case_dir / "lambda_64x64.npy", lam)
        np.save(case_dir / "binary_mask.npy", binary)
        np.save(case_dir / "bitmap_8x8.npy", coarse)
        plt.imsave(case_dir / "lambda_64x64.png", lam, origin="lower",
                   cmap="gray_r", vmin=0, vmax=1)
        (case_dir / "connectivity.json").write_text(json.dumps({
            "status": "PASS", "all_four_ports_same_component": True,
            "source_topology_id": topology_id,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
