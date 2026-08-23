"""Geometry-only physical benchmark smoke CLI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from .benchmark_setup import load_config_geometry
from .domain import CONTROLLED_TPFM_DERIVED_DOMAIN
from .physical_validation import validate_physical_domain
from .train import atomic_json


def _draw_patch(axis, geometry, patch, color: str, label: str) -> None:
    first = True
    for interval in patch.intervals:
        if patch.side in {"left", "right"}:
            x = geometry.xmin if patch.side == "left" else geometry.xmax
            axis.plot(
                [x, x], [interval.minimum, interval.maximum], color=color,
                linewidth=3, label=label if first else None,
            )
        else:
            y = geometry.ymin if patch.side == "bottom" else geometry.ymax
            axis.plot(
                [interval.minimum, interval.maximum], [y, y], color=color,
                linewidth=3, label=label if first else None,
            )
        first = False


def run(config_path: Path, output_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    config, _, domain_path, spec, geometry, context, mapper = load_config_geometry(
        config_path
    )
    if spec is None:
        raise ValueError("geometry smoke requires physical boundary metadata")
    report = validate_physical_domain(
        geometry,
        context,
        mapper,
        allow_anisotropic_mesh=config.get("allow_anisotropic_mesh", False),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "geometry_compatibility.json", report)
    atomic_json(output_dir / "geometry_labels.json", {
        "benchmark_mode": config["benchmark_mode"],
        "benchmark_case": config.get("benchmark_case"),
        "classification": geometry.classification,
        "article_reproduction": False,
        "tpfm_or_article_claim": False,
    })
    if spec.classification == CONTROLLED_TPFM_DERIVED_DOMAIN:
        shutil.copyfile(domain_path, output_dir / "controlled_domain_spec.json")

    chi = (geometry.lambda_field > mapper.interface_tolerance).astype(np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    extent = (geometry.xmin, geometry.xmax, geometry.ymin, geometry.ymax)
    for axis, values, title in zip(axes, (geometry.lambda_field, chi), ("lambda", "chi")):
        image = axis.imshow(values, origin="lower", extent=extent, aspect="equal")
        roi = geometry.roi_bounds
        if roi is not None:
            axis.add_patch(Rectangle(
                (roi.xmin, roi.ymin), roi.xmax - roi.xmin, roi.ymax - roi.ymin,
                fill=False, edgecolor="white", linewidth=1.5, linestyle="--", label="ROI",
            ))
        for patches, color, label in (
            (spec.inlet, "#16c784", "inlet"),
            (spec.outlet, "#ffb000", "outlet"),
            (spec.wall, "#d62728", "wall"),
        ):
            for patch in patches:
                _draw_patch(axis, geometry, patch, color, label)
                label = "_nolegend_"
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, fraction=0.046)
    axes[0].legend(loc="upper center", ncol=4, fontsize=8)
    figure.suptitle(geometry.classification)
    image_name = (
        "geometry_controlled.png"
        if spec.classification == CONTROLLED_TPFM_DERIVED_DOMAIN
        else f"geometry_{config['benchmark_case']}.png"
    )
    figure.savefig(output_dir / image_name, dpi=180)
    plt.close(figure)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    main()
