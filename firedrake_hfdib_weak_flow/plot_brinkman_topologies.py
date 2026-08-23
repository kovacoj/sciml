"""Plot analytic Brinkman topology A/B/C fields."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.brinkman_topologies import BrinkmanTopologyGeometry


def main() -> None:
    output = Path("outputs/brinkman_topologies")
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for axis, name in zip(axes, "ABC"):
        geometry = BrinkmanTopologyGeometry(name)
        image = axis.imshow(
            geometry.lambda_field, origin="lower", extent=(0, .128, 0, .128),
            cmap="gray_r", vmin=0, vmax=1,
        )
        axis.set_title(f"Topology {name}: {geometry.topology.name.split('_', 1)[1]}")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal")
    figure.colorbar(image, ax=axes, label=r"Brinkman topology $\lambda$")
    figure.savefig(output / "figure_brinkman_topologies.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
