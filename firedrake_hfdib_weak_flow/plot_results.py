"""Generate final figures, transfer comparison, and a file-derived report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.geometry import TPFMGeometry


TRANSFER_COLUMNS = [
    "step", "scratch_loss", "transfer_loss", "scratch_momentum",
    "transfer_momentum", "scratch_continuity", "transfer_continuity",
    "scratch_divergence", "transfer_divergence", "scratch_mass_imbalance",
    "transfer_mass_imbalance", "scratch_wall_seconds", "transfer_wall_seconds",
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


def write_transfer_csv(scratch: list[dict], transfer: list[dict], path: Path) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRANSFER_COLUMNS)
        writer.writeheader()
        scratch_by_step = {int(row["global_step"]): row for row in scratch}
        transfer_by_step = {int(row["global_step"]): row for row in transfer}
        for step in sorted(scratch_by_step.keys() & transfer_by_step.keys()):
            scratch_row = scratch_by_step[step]
            transfer_row = transfer_by_step[step]
            writer.writerow({
                "step": step,
                "scratch_loss": scratch_row["loss_total"],
                "transfer_loss": transfer_row["loss_total"],
                "scratch_momentum": scratch_row["loss_momentum"],
                "transfer_momentum": transfer_row["loss_momentum"],
                "scratch_continuity": scratch_row["loss_continuity"],
                "transfer_continuity": transfer_row["loss_continuity"],
                "scratch_divergence": scratch_row["divergence_l2"],
                "transfer_divergence": transfer_row["divergence_l2"],
                "scratch_mass_imbalance": scratch_row["mass_imbalance"],
                "transfer_mass_imbalance": transfer_row["mass_imbalance"],
                "scratch_wall_seconds": scratch_row["elapsed_seconds"],
                "transfer_wall_seconds": transfer_row["elapsed_seconds"],
            })


def resolve_dataset(config: dict, project_root: Path) -> Path:
    path = Path(config["dataset"]).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        project_root / path,
        project_root.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def plot_hfdib_geometry(config: dict, project_root: Path, path: Path) -> None:
    geometry = TPFMGeometry(
        resolve_dataset(config, project_root),
        sample_index=int(config["topology_index"]),
        spacing=float(config["spacing"]),
    )
    chi = (geometry.lambda_field > 1.0e-12).astype(np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    extent = (
        geometry.x[0], geometry.x[-1], geometry.y[0], geometry.y[-1],
    )
    for axis, values, title in zip(
        axes,
        (geometry.lambda_field, chi, geometry.signed_distance),
        (r"$\lambda$", r"$\chi$", "signed distance"),
    ):
        image = axis.imshow(values, origin="lower", extent=extent)
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046)

    interface_y, interface_x = np.nonzero(geometry.interface)
    if len(interface_x):
        selected = np.linspace(0, len(interface_x) - 1, min(8, len(interface_x)), dtype=int)
        points = np.column_stack((geometry.x[interface_x[selected]], geometry.y[interface_y[selected]]))
        normals = geometry.normals[interface_y[selected], interface_x[selected]]
        sigma = geometry.signed_distance[interface_y[selected], interface_x[selected]]
        boundary = points - sigma[:, None] * normals
        first = boundary + geometry.spacing * normals
        second = boundary + 2.0 * geometry.spacing * normals
        axes[2].quiver(
            boundary[:, 0], boundary[:, 1], geometry.spacing * normals[:, 0],
            geometry.spacing * normals[:, 1], angles="xy", scale_units="xy", scale=1,
            color="white", width=0.006,
        )
        for values, marker, label in (
            (boundary, "o", r"$x_B$"),
            (first, "x", r"$x_1$"),
            (second, "+", r"$x_2$"),
        ):
            axes[2].scatter(
                values[:, 0], values[:, 1], marker=marker, s=20, label=label,
            )
        axes[2].legend(loc="lower right", fontsize=7, framealpha=0.8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_checkpoint_metadata(run: Path) -> dict:
    path = run / "checkpoint_latest.pt"
    if not path.exists():
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "git_sha": checkpoint.get("git_sha", "unknown"),
        "config": checkpoint["config"],
        "global_step": checkpoint["global_step"],
        "beta": checkpoint["beta"],
        "Cm": checkpoint["Cm"],
        "Cc": checkpoint["Cc"],
        "elapsed_seconds": checkpoint["elapsed_seconds"],
        "parameter_count": sum(value.numel() for value in checkpoint["model_state"].values()),
    }


def current_git_sha(project_root: Path) -> str:
    repo_root = project_root.parent.resolve()
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def row_at_step(history: list[dict], step: int) -> dict | None:
    return next((row for row in history if int(row["global_step"]) == step), None)


def threshold_result(history: list[dict], threshold: float) -> str:
    for row in history:
        if float(row["loss_total"]) <= threshold:
            return f"{int(row['global_step'])} / {float(row['elapsed_seconds']):.2f} s"
    return "not reached"


def format_number(value: float) -> str:
    return f"{float(value):.6g}"


def write_report(
    histories: dict[str, list[dict]], metadata: dict[str, dict],
    gradient_check: dict, project_root: Path, path: Path,
) -> None:
    topology_a = histories["Topology A"]
    scratch = histories["Topology B scratch"]
    transfer = histories["Topology B transfer"]
    implementation = metadata["Topology A"]
    config = implementation.get("config", {})
    git_sha = implementation.get("git_sha", "unknown")
    if not git_sha or git_sha == "unknown":
        git_sha = current_git_sha(project_root)

    lines = [
        "# Presentation Results", "",
        "All values are derived from completed checkpoints, histories, configuration, gradient-check, or Git artifacts.", "",
        "## Implementation", "",
        f"- Git SHA: `{git_sha}`",
        f"- Mesh: {config.get('nx', 'unknown')} x {config.get('ny', 'unknown')}",
        f"- Finite elements: P{config.get('velocity_degree', '?')} velocity / P{config.get('pressure_degree', '?')} pressure",
        f"- Network: 4 inputs, {config.get('network_depth', '?')} hidden tanh layers x {config.get('network_width', '?')}, 3 outputs",
        f"- Parameters: {implementation.get('parameter_count', 'unknown')}",
        "- HFDIB interpolation: quadratic (`xB`, `x1`, `x2`)",
        f"- Viscosity `nu`: {config.get('nu', 'unknown')}",
        f"- Inlet velocity `uin`: {config.get('uin', 'unknown')}",
        "- Supervised labels: NO", "",
    ]

    lines.extend(("## Gradient Validation", ""))
    if gradient_check.get("relative_errors"):
        best_index = int(np.argmin(gradient_check["relative_errors"]))
        lines.extend((
            f"- Best epsilon: {format_number(gradient_check['epsilons'][best_index])}",
            f"- AD directional derivative: {format_number(gradient_check['exact_directional_derivative'])}",
            f"- FD directional derivative: {format_number(gradient_check['finite_differences'][best_index])}",
            f"- Relative error: {format_number(gradient_check['relative_errors'][best_index])}", "",
        ))
    else:
        lines.extend(("No gradient-check artifact is available.", ""))

    lines.extend(("## Topology A", ""))
    if topology_a:
        initial, final = topology_a[0], topology_a[-1]
        initial_loss = float(initial["loss_total"])
        final_loss = float(final["loss_total"])
        wall = implementation.get("elapsed_seconds", final["elapsed_seconds"])
        lines.extend((
            f"- Steps: {int(final['global_step'])}",
            f"- Wall time: {float(wall):.2f} s",
            f"- Total loss: {format_number(initial_loss)} at first log -> {format_number(final_loss)} final",
            f"- Reduction: {(1.0 - final_loss / initial_loss) * 100.0:.2f}% ({initial_loss / final_loss:.2f}x)",
            f"- Final divergence L2: {format_number(final['divergence_l2'])}",
            f"- Final mass imbalance: {format_number(final['mass_imbalance'])}", "",
        ))
    else:
        lines.extend(("No Topology A history artifact is available.", ""))

    lines.extend((
        "## Topology B Transfer", "",
        "| Run | Loss @ 50 | Loss @ 100 | Loss @ 200 | Best loss |",
        "|---|---:|---:|---:|---:|",
    ))
    for label, history in (("Scratch", scratch), ("Transfer", transfer)):
        values = []
        for step in (50, 100, 200):
            row = row_at_step(history, step)
            values.append(format_number(row["loss_total"]) if row else "not logged")
        best = min((float(row["loss_total"]) for row in history), default=float("nan"))
        lines.append(f"| {label} | {' | '.join(values)} | {format_number(best)} |")

    lines.extend((
        "", "### Threshold Steps / Wall Time", "",
        "| Loss threshold | Scratch | Transfer |",
        "|---:|---:|---:|",
    ))
    for threshold in (1.0, 0.5, 0.25):
        lines.append(
            f"| {threshold:g} | {threshold_result(scratch, threshold)} | "
            f"{threshold_result(transfer, threshold)} |"
        )
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
        histories, metadata, gradient_check, project_root,
        arguments.final_dir / "PRESENTATION_RESULTS.md",
    )


if __name__ == "__main__":
    main()
