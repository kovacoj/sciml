"""Generate final figures, transfer comparison, and a file-derived report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.geometry import TPFMGeometry


TRANSFER_COLUMNS = [
    "loss_threshold", "scratch_global_step", "transfer_global_step", "speedup"
]


def read_history(run: Path) -> list[dict[str, float | str]]:
    path = run / "history.csv"
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{
        key: float(value) if value not in ("", "nan") else value
        for key, value in row.items()
    } for row in rows]


def latest(run: Path, pattern: str) -> Path | None:
    paths = sorted(run.glob(pattern))
    return paths[-1] if paths else None


def plot_training(history: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    steps = [row["global_step"] for row in history]
    for column, label in (
        ("loss_total", "total"),
        ("loss_momentum", "momentum"),
        ("loss_continuity", "continuity"),
    ):
        axis.semilogy(steps, [row[column] for row in history], label=label)
    for previous, current in zip(history, history[1:]):
        if current["beta"] != previous["beta"]:
            axis.axvline(current["global_step"], color="0.5", linestyle="--", linewidth=0.8)
    axis.set_xlabel("global step")
    axis.set_ylabel("loss")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_transfer(scratch: list[dict], transfer: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for history, label in ((scratch, "scratch"), (transfer, "transfer")):
        if history:
            axis.semilogy(
                [row["global_step"] for row in history],
                [row["loss_total"] for row in history], label=label,
            )
    axis.set_xlabel("global step")
    axis.set_ylabel("total loss")
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def first_threshold_step(history: list[dict], threshold: float) -> int | str:
    for row in history:
        if row["loss_total"] <= threshold:
            return int(row["global_step"])
    return ""


def write_transfer_csv(scratch: list[dict], transfer: list[dict], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRANSFER_COLUMNS)
        writer.writeheader()
        if not scratch or not transfer:
            return
        combined = [float(row["loss_total"]) for row in scratch + transfer]
        upper = int(np.ceil(np.log10(max(combined))))
        lower = int(np.floor(np.log10(max(min(combined), 1.0e-300))))
        for exponent in range(upper, lower - 1, -1):
            threshold = 10.0**exponent
            scratch_step = first_threshold_step(scratch, threshold)
            transfer_step = first_threshold_step(transfer, threshold)
            speedup = (
                float(scratch_step) / float(transfer_step)
                if scratch_step != "" and transfer_step not in ("", 0) else ""
            )
            writer.writerow({
                "loss_threshold": threshold,
                "scratch_global_step": scratch_step,
                "transfer_global_step": transfer_step,
                "speedup": speedup,
            })


def resolve_dataset(config: dict, project_root: Path) -> Path:
    path = Path(config["dataset"])
    return path if path.is_absolute() else (project_root / path).resolve()


def plot_hfdib_geometry(config: dict, project_root: Path, path: Path) -> None:
    geometry = TPFMGeometry(
        resolve_dataset(config, project_root),
        sample_index=int(config["topology_index"]),
        spacing=float(config["spacing"]),
    )
    chi = (geometry.lambda_field > 1.0e-12).astype(np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (geometry.lambda_field, chi, geometry.signed_distance),
        ("lambda", "chi", "signed distance"),
    ):
        image = axis.imshow(values, origin="lower")
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_checkpoint_metadata(run: Path) -> dict:
    path = run / "checkpoint_latest.pt"
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "git_sha": checkpoint["git_sha"],
        "config": checkpoint["config"],
        "global_step": checkpoint["global_step"],
        "beta": checkpoint["beta"],
        "Cm": checkpoint["Cm"],
        "Cc": checkpoint["Cc"],
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }


def write_report(
    histories: dict[str, list[dict]], metadata: dict[str, dict],
    gradient_check: dict, transfer_rows: list[dict], path: Path,
) -> None:
    lines = [
        "# Presentation Results", "",
        "Every value in this report is read from a checkpoint, history, or gradient-check artifact.", "",
    ]
    for name, history in histories.items():
        lines.extend((f"## {name}", ""))
        run_metadata = metadata[name]
        if run_metadata:
            lines.extend((f"- Git SHA: `{run_metadata['git_sha']}`", "- Checkpoint configuration:"))
            lines.extend(("```json", json.dumps(run_metadata["config"], indent=2), "```", ""))
        if not history:
            lines.extend(("No history artifact is available.", ""))
            continue
        lines.extend(("| Metric | Final value |", "|---|---:|"))
        for key, value in history[-1].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
    lines.extend(("## Gradient Check", "", "```json", json.dumps(gradient_check, indent=2), "```", ""))
    lines.extend(("## Transfer Thresholds", "", "| " + " | ".join(TRANSFER_COLUMNS) + " |"))
    lines.append("|" + "---|" * len(TRANSFER_COLUMNS))
    for row in transfer_rows:
        lines.append("| " + " | ".join(str(row[column]) for column in TRANSFER_COLUMNS) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--final-dir", type=Path, default=Path("outputs/final"))
    arguments = parser.parse_args()
    arguments.final_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parent
    runs = {
        "Topology A": arguments.outputs / "topoA",
        "Topology B scratch": arguments.outputs / "topoB_scratch",
        "Topology B transfer": arguments.outputs / "topoB_transfer",
    }
    histories = {name: read_history(run) for name, run in runs.items()}
    metadata = {name: load_checkpoint_metadata(run) for name, run in runs.items()}

    preview = latest(runs["Topology A"], "preview_*.png")
    if preview:
        shutil.copyfile(preview, arguments.final_dir / "figure_topology_A.png")
    if histories["Topology A"]:
        plot_training(
            histories["Topology A"],
            arguments.final_dir / "figure_topology_A_training.png",
        )
    scratch, transfer = histories["Topology B scratch"], histories["Topology B transfer"]
    if scratch or transfer:
        plot_transfer(scratch, transfer, arguments.final_dir / "figure_transfer.png")
    transfer_csv = arguments.final_dir / "transfer_comparison.csv"
    write_transfer_csv(scratch, transfer, transfer_csv)
    with transfer_csv.open(newline="") as stream:
        transfer_rows = list(csv.DictReader(stream))

    topology_config = metadata["Topology A"].get("config")
    if topology_config is None:
        config_path = project_root / "configs/tpfm_topology_a.json"
        topology_config = json.loads(config_path.read_text()) if config_path.exists() else None
    if topology_config is not None and resolve_dataset(topology_config, project_root).exists():
        plot_hfdib_geometry(
            topology_config, project_root,
            arguments.final_dir / "figure_hfdib_geometry.png",
        )
    gradient_path = arguments.outputs / "gradient_check.json"
    gradient_check = json.loads(gradient_path.read_text()) if gradient_path.exists() else {}
    write_report(
        histories, metadata, gradient_check, transfer_rows,
        arguments.final_dir / "PRESENTATION_RESULTS.md",
    )


if __name__ == "__main__":
    main()
