"""Aggregate matched-initialization MLP residual-metric pilots."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    source = Path("outputs/stokes_mlp_loss_pilot")
    destination = source / "final"
    destination.mkdir(parents=True, exist_ok=True)
    meshes = ((16, 8), (32, 16))
    seeds = (11, 22, 33, 44, 55)
    methods = ("raw", "dual", "correction")
    rows = []
    reports = {}
    for mesh in meshes:
        for seed in seeds:
            report = json.loads((source / f"mlp_{mesh[0]}x{mesh[1]}_seed{seed}.json").read_text())
            reports[mesh, seed] = {result["loss"]: result for result in report["results"]}
            for result in report["results"]:
                rows.append({key: value for key, value in result.items() if key != "history_samples"})
    columns = ["nx", "ny", "seed", "loss", "normalized_final",
               "relative_coefficient_error", "relative_velocity_error",
               "relative_pressure_gauge_error", "iterations", "evaluations",
               "elapsed_seconds", "preconditioner_solve_seconds"]
    with (destination / "mlp_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for mesh in meshes:
        for method in methods:
            values = np.asarray([
                reports[mesh, seed][method]["relative_velocity_error"] for seed in seeds
            ])
            pressure = np.asarray([
                reports[mesh, seed][method]["relative_pressure_gauge_error"] for seed in seeds
            ])
            times = np.asarray([
                reports[mesh, seed][method]["elapsed_seconds"] for seed in seeds
            ])
            summary.append({
                "nx": mesh[0], "ny": mesh[1], "loss": method,
                "velocity_median": float(np.median(values)),
                "velocity_q1": float(np.quantile(values, .25)),
                "velocity_q3": float(np.quantile(values, .75)),
                "pressure_median": float(np.median(pressure)),
                "time_median_seconds": float(np.median(times)),
            })
    (destination / "mlp_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {"raw": "tab:red", "dual": "tab:blue", "correction": "tab:green"}
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, mesh in zip(axes, meshes):
        positions = np.arange(len(methods))
        data = [[reports[mesh, seed][method]["relative_velocity_error"] for seed in seeds] for method in methods]
        plot = axis.boxplot(
            data, positions=positions, patch_artist=True, tick_labels=methods
        )
        for patch, method in zip(plot["boxes"], methods):
            patch.set_facecolor(colors[method]); patch.set_alpha(.55)
        axis.set_yscale("log"); axis.set_title(f"{mesh[0]}x{mesh[1]}, five seeds")
        axis.set_ylabel("relative velocity error"); axis.grid(alpha=.25)
    figure.savefig(destination / "figure_mlp_velocity_boxplots.png", dpi=190)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, mesh in zip(axes, meshes):
        for method in methods:
            histories = [reports[mesh, seed][method]["history_samples"] for seed in seeds]
            common = min(len(history) for history in histories)
            iterations = [histories[0][index]["iteration"] for index in range(common)]
            losses = np.asarray([[history[index]["loss"] for index in range(common)] for history in histories])
            median = np.median(losses, axis=0)
            q1, q3 = np.quantile(losses, [.25, .75], axis=0)
            axis.semilogy(iterations, median, color=colors[method], label=method)
            axis.fill_between(iterations, q1, q3, color=colors[method], alpha=.18)
        axis.set_title(f"{mesh[0]}x{mesh[1]}"); axis.set_xlabel("L-BFGS iteration")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("normalized training loss"); axes[0].legend()
    figure.savefig(destination / "figure_mlp_loss_convergence.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
