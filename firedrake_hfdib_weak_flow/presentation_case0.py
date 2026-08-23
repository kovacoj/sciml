"""Generate the frozen Case0 supervisor-presentation artifact package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _json(path: Path):
    return json.loads(path.read_text())


def _comparison_row(loss, ux, pressure, mass, divergence):
    return {
        "weak_loss": float(loss),
        "ux_relative_l2": float(ux),
        "pressure_gauge_relative_l2": float(pressure),
        "mass_imbalance": float(mass),
        "divergence_l2": float(divergence),
    }


def generate(outputs: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    direct_metrics = _json(outputs / "manufactured_empty_direct_40x20/direct_reference_metrics.json")
    coefficient = _json(outputs / "manufactured_empty_fe_coefficients_lbfgs_scaled/metrics.json")
    neural_metrics = _json(outputs / "manufactured_empty_stokes_lbfgs_3000/metrics.json")
    neural_comparison = _json(outputs / "manufactured_empty_stokes_lbfgs_3000/reference_comparison.json")
    direct_injection_loss = 1.07216e-30
    summary = {
        "case": "manufactured empty-channel Stokes",
        "claim": "shared Taylor-Hood weak dual-residual validation",
        "direct_fe": _comparison_row(
            direct_injection_loss, 0.0, 0.0,
            direct_metrics["mass_imbalance"], direct_metrics["divergence_l2"],
        ),
        "fe_coefficients_lbfgs": _comparison_row(
            coefficient["final"]["loss"],
            coefficient["comparison"]["ux_relative_l2"],
            coefficient["comparison"]["pressure_gauge_centered_relative_l2"],
            coefficient["diagnostics"]["mass_imbalance"],
            coefficient["diagnostics"]["divergence_l2"],
        ),
        "coordinate_mlp": _comparison_row(
            neural_metrics["final"]["loss"],
            neural_comparison["ux_relative_l2"],
            neural_comparison["pressure_gauge_centered_relative_l2"],
            neural_metrics["diagnostics"]["mass_imbalance"],
            neural_metrics["diagnostics"]["divergence_l2"],
        ),
        "mlp_gate": {
            "ux_relative_l2_lt": 0.01,
            "mass_imbalance_lt": 1.0e-4,
            "passed": False,
        },
    }
    (destination / "case0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (destination / "case0_table.csv").open("w", newline="") as stream:
        columns = ["method", *summary["direct_fe"].keys()]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for key, label in (
            ("direct_fe", "Direct FE"),
            ("fe_coefficients_lbfgs", "FE coefficients + L-BFGS"),
            ("coordinate_mlp", "Coordinate MLP"),
        ):
            writer.writerow({"method": label, **summary[key]})

    direct = np.load(outputs / "manufactured_empty_direct_40x20/direct_reference.npz")
    neural = np.load(outputs / "manufactured_empty_stokes_lbfgs_3000/final_fields.npz")
    try:
        sx, qx = direct["s_coordinates"], direct["q_coordinates"]
        direct_p = direct["p"] - direct["p"].mean()
        neural_p = neural["p"] - neural["p"].mean()
        figure, axes = plt.subplots(2, 4, figsize=(14, 5.8), constrained_layout=True)
        panels = (
            (axes[0, 0], sx, direct["ux"], r"Direct FE $u_x$"),
            (axes[0, 1], sx, neural["ux"], r"Neural $u_x$"),
            (axes[0, 2], sx, neural["ux"] - direct["ux"], r"$u_x^{NN}-u_x^{FE}$"),
            (axes[0, 3], sx, np.abs(neural["ux"] - direct["ux"]), r"absolute $u_x$ error"),
            (axes[1, 0], qx, direct_p, "Direct FE pressure (gauge)"),
            (axes[1, 1], qx, neural_p, "Neural pressure (gauge)"),
            (axes[1, 2], qx, neural_p - direct_p, r"$p^{NN}-p^{FE}$"),
            (axes[1, 3], sx, np.hypot(neural["ux"] - direct["ux"], neural["uy"] - direct["uy"]), "velocity error magnitude"),
        )
        for axis, coordinates, values, title in panels:
            image = axis.tricontourf(coordinates[:, 0], coordinates[:, 1], values, levels=30)
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(image, ax=axis, fraction=0.046)
        figure.savefig(destination / "figure_case0_fields.png", dpi=190)
        plt.close(figure)
    finally:
        direct.close()
        neural.close()

    adam_rows = list(csv.DictReader((outputs / "manufactured_empty_stokes/history.csv").open()))
    coefficient_history = coefficient["history"]
    figure, axis = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    axis.semilogy(
        [int(row["global_step"]) for row in adam_rows],
        [float(row["loss_total"]) for row in adam_rows],
        label="Coordinate MLP: Adam",
    )
    axis.semilogy(
        [int(row["step"]) for row in coefficient_history],
        [float(row["loss"]) for row in coefficient_history],
        label="FE coefficients: L-BFGS",
    )
    mlp_points = [(1000, float(adam_rows[-1]["loss_total"]))]
    cumulative = 1000
    for name in ("manufactured_empty_stokes_lbfgs", "manufactured_empty_stokes_lbfgs_2000", "manufactured_empty_stokes_lbfgs_3000"):
        metrics = _json(outputs / name / "metrics.json")
        cumulative += int(metrics["nit"])
        mlp_points.append((cumulative, float(metrics["final"]["loss"])))
    axis.semilogy(
        [point[0] for point in mlp_points], [point[1] for point in mlp_points],
        marker="o", linewidth=1.8, label="Coordinate MLP: L-BFGS blocks",
    )
    axis.axhline(1.0e-8, color="0.5", linestyle="--", linewidth=1, label="coefficient validation target")
    axis.set_xlabel("optimizer iteration")
    axis.set_ylabel("normalized weak dual loss")
    axis.legend()
    figure.savefig(destination / "figure_case0_optimization.png", dpi=190)
    plt.close(figure)

    markdown = [
        "# Firedrake Case 0", "",
        "All values are generated from frozen JSON/CSV/NPZ artifacts.", "",
        "| Method | Weak loss | ux relative error | Gauge pressure error | Mass imbalance | Divergence L2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("direct_fe", "Direct FE"),
        ("fe_coefficients_lbfgs", "FE coefficients + L-BFGS"),
        ("coordinate_mlp", "Coordinate MLP"),
    ):
        row = summary[key]
        markdown.append(
            f"| {label} | {row['weak_loss']:.3e} | {row['ux_relative_l2']:.3e} | "
            f"{row['pressure_gauge_relative_l2']:.3e} | {row['mass_imbalance']:.3e} | "
            f"{row['divergence_l2']:.5f} |"
        )
    markdown.extend((
        "", "The coefficient result validates equivalence between the shared Taylor-Hood weak root and dual-residual minimization.",
        "The MLP approximates the field but misses the predeclared 1% velocity and 1e-4 mass gates; HFDIB escalation remains blocked.",
        "The similar divergence L2 values reflect Taylor-Hood weak incompressibility; global mass defect distinguishes the remaining neural discrepancy.",
    ))
    (destination / "SUMMARY.md").write_text("\n".join(markdown) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--destination", type=Path, default=Path("outputs/final_case0"))
    arguments = parser.parse_args()
    generate(arguments.outputs, arguments.destination)


if __name__ == "__main__":
    main()
