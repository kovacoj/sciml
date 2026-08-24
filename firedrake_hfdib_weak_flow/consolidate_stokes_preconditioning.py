"""Build paper-quality figures from validated Stokes pilot artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VALIDATED_COEFFICIENT_METHODS = (
    "raw", "dual", "jacobi_ls", "block", "correction", "oracle",
)
VALIDATED_MLP_METHODS = ("raw", "dual", "correction")


def load_coefficient(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return [
            {
                key: (float(value) if key not in {"loss"} else value)
                for key, value in row.items()
            }
            for row in csv.DictReader(stream)
        ]


def generate(project: Path) -> Path:
    source = project / "outputs/stokes_preconditioning_research"
    destination = source / "final_figures"
    destination.mkdir(parents=True, exist_ok=True)
    coefficient = load_coefficient(
        source / "stokes_loss_pilot_expanded/pilot_summary.csv"
    )
    mlp = json.loads(
        (source / "stokes_mlp_loss_pilot/mlp_summary.json").read_text()
    )
    coefficient = [
        row for row in coefficient if row["loss"] in VALIDATED_COEFFICIENT_METHODS
    ]
    mlp = [row for row in mlp if row["loss"] in VALIDATED_MLP_METHODS]

    with (destination / "validated_coefficient_table.csv").open("w", newline="") as stream:
        columns = ["nx", "ny", "loss", "relative_velocity_error",
                   "relative_pressure_gauge_error", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(coefficient)
    with (destination / "validated_mlp_table.csv").open("w", newline="") as stream:
        columns = ["nx", "ny", "loss", "velocity_median", "velocity_q1",
                   "velocity_q3", "pressure_median", "time_median_seconds"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(mlp)

    colors = {"raw": "#c43c39", "dual": "#2878b5", "correction": "#27864a"}
    labels = {"raw": "Raw residual", "dual": "Dual residual", "correction": "Exact correction"}
    meshes = ((16, 8), (32, 16), (64, 32))
    figure, axis = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
    for method in VALIDATED_MLP_METHODS:
        rows = [row for row in coefficient if row["loss"] == method]
        axis.semilogy(
            [f"{int(row['nx'])}x{int(row['ny'])}" for row in rows],
            [row["relative_velocity_error"] for row in rows],
            "o-", lw=2, ms=6, color=colors[method], label=labels[method],
        )
    axis.set_xlabel("Taylor-Hood mesh")
    axis.set_ylabel("relative velocity error")
    axis.set_title("Coefficient optimization: fixed 300-step budget")
    axis.grid(alpha=.25); axis.legend()
    figure.savefig(destination / "coefficient_mesh_sensitivity.png", dpi=220)
    figure.savefig(destination / "coefficient_mesh_sensitivity.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.3), constrained_layout=True)
    for method in VALIDATED_MLP_METHODS:
        rows = [row for row in mlp if row["loss"] == method]
        median = np.asarray([row["velocity_median"] for row in rows])
        lower = median - np.asarray([row["velocity_q1"] for row in rows])
        upper = np.asarray([row["velocity_q3"] for row in rows]) - median
        axis.errorbar(
            [f"{row['nx']}x{row['ny']}" for row in rows], median,
            yerr=np.vstack((lower, upper)), marker="o", capsize=4, lw=2,
            color=colors[method], label=labels[method],
        )
    axis.set_yscale("log")
    axis.set_xlabel("Taylor-Hood mesh")
    axis.set_ylabel("median relative velocity error (IQR)")
    axis.set_title("Coordinate MLP: five matched seeds")
    axis.grid(alpha=.25); axis.legend()
    figure.savefig(destination / "mlp_mesh_sensitivity.png", dpi=220)
    figure.savefig(destination / "mlp_mesh_sensitivity.pdf")
    plt.close(figure)

    rows64 = {
        row["loss"]: row for row in coefficient
        if int(row["nx"]) == 64 and int(row["ny"]) == 32
    }
    ordered = ("raw", "dual", "block", "correction", "oracle")
    display = {
        "raw": "Raw", "dual": "Dual", "block": "Block",
        "correction": "Correction", "oracle": "Oracle",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    bar_colors = ["#c43c39", "#2878b5", "#8a5ca8", "#27864a", "#333333"]
    axes[0].bar(
        [display[name] for name in ordered],
        [rows64[name]["relative_velocity_error"] for name in ordered],
        color=bar_colors,
    )
    axes[0].set_yscale("log"); axes[0].set_ylabel("relative velocity error")
    axes[0].set_title("64x32, 300 iterations")
    ratio = rows64["correction"]["relative_velocity_error"] / rows64["oracle"]["relative_velocity_error"]
    axes[1].bar(["Correction", "Oracle"], [
        rows64["correction"]["relative_velocity_error"],
        rows64["oracle"]["relative_velocity_error"],
    ], color=["#27864a", "#333333"])
    axes[1].set_yscale("log"); axes[1].set_ylabel("relative velocity error")
    axes[1].set_title(f"Correction / oracle = {ratio:.3f}")
    figure.suptitle("Physics-only correction tracks unavailable FE-error oracle")
    figure.savefig(destination / "oracle_equivalence.png", dpi=220)
    figure.savefig(destination / "oracle_equivalence.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    method_colors = {
        "raw": "#c43c39", "dual": "#2878b5", "jacobi_ls": "#e89c31",
        "block": "#8a5ca8", "correction": "#27864a", "oracle": "#333333",
    }
    for method in VALIDATED_COEFFICIENT_METHODS:
        row = rows64[method]
        axis.scatter(
            row["elapsed_seconds"], row["relative_velocity_error"], s=75,
            color=method_colors[method], label=method,
        )
        axis.annotate(method, (row["elapsed_seconds"], row["relative_velocity_error"]),
                      xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_yscale("log")
    axis.set_xlabel("total optimizer wall time [s]")
    axis.set_ylabel("relative velocity error")
    axis.set_title("64x32 accuracy-cost Pareto view")
    axis.grid(alpha=.25)
    figure.savefig(destination / "accuracy_vs_cost.png", dpi=220)
    figure.savefig(destination / "accuracy_vs_cost.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 2.7), constrained_layout=True)
    axis.axis("off")
    boxes = (
        (0.02, "Raw residual\n$\\|AU-b\\|^2$\nill-conditioned"),
        (0.27, "Dual residual\n$r^T G^{-1}r$\nbetter scaling"),
        (0.52, "PDE correction\n$A\\delta=-(AU-b)$\n$\\delta=U_h-U$"),
        (0.77, "Correction loss\n$\\|\\delta\\|_X^2$\n≈ oracle, no labels"),
    )
    for x, text in boxes:
        axis.text(
            x, .5, text, transform=axis.transAxes, ha="left", va="center",
            fontsize=12, bbox=dict(boxstyle="round,pad=.6", fc="#eef4f8", ec="#355c75"),
        )
    for x in (.235, .485, .735):
        axis.annotate("", xy=(x + .02, .5), xytext=(x - .02, .5),
                      xycoords=axis.transAxes,
                      arrowprops=dict(arrowstyle="->", lw=2, color="#355c75"))
    axis.set_title("Operator preconditioning changes the neural optimization problem", fontsize=15)
    figure.savefig(destination / "operator_preconditioning_concept.png", dpi=220)
    figure.savefig(destination / "operator_preconditioning_concept.pdf")
    plt.close(figure)

    report = {
        "validated_only": True,
        "richardson_included": False,
        "coefficient_64x32": {name: {
            "velocity_error": rows64[name]["relative_velocity_error"],
            "pressure_error": rows64[name]["relative_pressure_gauge_error"],
            "wall_seconds": rows64[name]["elapsed_seconds"],
        } for name in VALIDATED_COEFFICIENT_METHODS},
        "correction_oracle_velocity_error_ratio": ratio,
        "mlp": mlp,
    }
    (destination / "validated_results.json").write_text(json.dumps(report, indent=2) + "\n")
    return destination


def hash_manifest(directory: Path) -> None:
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            })
    (directory / "manifest.json").write_text(json.dumps({
        "validated_only": True,
        "conflicting_richardson_work_excluded": True,
        "files": files,
    }, indent=2) + "\n")


if __name__ == "__main__":
    destination = generate(Path(__file__).resolve().parent)
    hash_manifest(destination)
