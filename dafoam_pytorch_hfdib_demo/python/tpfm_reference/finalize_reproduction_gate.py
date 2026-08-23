"""Apply the frozen TPFM reproduction thresholds and emit gate artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())["results"]
    validation = json.loads(args.validation.read_text())["results"]
    rows = calibration + validation
    strong = all(
        row["velocity_rel_l2"] < 0.10
        and row["pressure_gauge_centered_rel_l2"] < 0.25
        and 0.75 < row["pressure_range_ratio"] < 1.33
        for row in rows
    )
    approximate = all(
        row["velocity_rel_l2"] < 0.15
        and row["pressure_gauge_centered_rel_l2"] < 0.35
        for row in rows
    )
    classification = (
        "EXACT_BENCHMARK_REPRODUCED" if strong else
        "APPROXIMATE_PHYSICAL_REPRODUCTION" if approximate else
        "TPFM_TOPOLOGIES_ONLY"
    )
    result = {
        "classification": classification,
        "geometry_selection": {
            "calibration_sample": 0,
            "extension_candidates_cells": [8, 16, 32],
            "selected_extension_cells_each_side": 32,
            "validation_samples": [274, 549],
        },
        "thresholds_predeclared": {
            "strong": {"velocity_rel_l2_lt": 0.10, "pressure_gauge_rel_l2_lt": 0.25, "pressure_range_ratio": [0.75, 1.33]},
            "approximate": {"velocity_rel_l2_lt": 0.15, "pressure_gauge_rel_l2_lt": 0.35},
            "historical_velocity_gate": 0.20,
        },
        "results": rows,
        "strong_reproduction_pass": strong,
        "approximate_physical_reproduction_pass": approximate,
        "wording": "TPFM topology distribution under our DAFoam/HFDIB discretization.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reproduction_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output_dir / "reproduction_gate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    samples = [str(row["sample_index"]) for row in rows]
    x = range(len(rows))
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar([value - 0.18 for value in x], [row["velocity_rel_l2"] for row in rows], 0.36, label="velocity")
    axis.bar([value + 0.18 for value in x], [row["pressure_gauge_centered_rel_l2"] for row in rows], 0.36, label="pressure")
    axis.axhline(0.15, color="k", linestyle="--", label="approx velocity gate")
    axis.set_xticks(list(x), samples)
    axis.set_xlabel("TPFM sample")
    axis.set_ylabel("relative L2 error")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "reproduction_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
