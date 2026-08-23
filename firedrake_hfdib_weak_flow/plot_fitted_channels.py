"""Plot the fitted Topology A direct Stokes result."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    source = Path("outputs/fitted_topology_A_direct")
    destination = Path("outputs/fitted_topology_A_presentation")
    destination.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((source / "metrics.json").read_text())
    with np.load(source / "direct_fields.npz") as data:
        velocity = data["velocity"]
        speed = np.linalg.norm(velocity, axis=1)
        pressure = data["pressure"] - data["pressure"].mean()
        vxy, pxy = data["velocity_coordinates"], data["pressure_coordinates"]
        figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
        panels = (
            (axes[0], vxy, speed, "Velocity magnitude"),
            (axes[1], pxy, pressure, "Pressure (gauge centered)"),
            (axes[2], vxy, velocity[:, 0], r"$u_x$"),
        )
        for axis, coordinates, values, title in panels:
            image = axis.tricontourf(coordinates[:, 0], coordinates[:, 1], values, levels=30)
            axis.set_title(title); axis.set_aspect("equal")
            axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]")
            figure.colorbar(image, ax=axis, fraction=0.046)
        figure.suptitle("Fitted level-set-derived Topology A: direct Stokes")
        figure.savefig(destination / "figure_fitted_topology_A_direct.png", dpi=190)
        plt.close(figure)
    (destination / "summary.json").write_text(json.dumps({
        **metrics,
        "neural_training_launched": False,
        "reason": "bounded time gate: variable fitted-mesh neural bridge deferred",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
