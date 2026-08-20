"""Probe whether the published 64x64 TPFM arrays are an interior CFD crop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    return [group for group in np.split(indices, np.where(np.diff(indices) > 1)[0] + 1) if group.size]


def values_stats(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def edge_openings(lam: np.ndarray, column: int) -> list[np.ndarray]:
    # The archive itself confirms lambda near 0 is fluid and lambda near 1 is solid.
    return contiguous_groups(np.flatnonzero(lam[:, column] < 0.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--indices", default="0,274,549")
    parser.add_argument("--local-cross-validation", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-figure", required=True, type=Path)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",")]
    with np.load(args.reference_root / "data" / "mixer_64.npz") as archive:
        inputs = archive["inputs"]
        outputs = archive["outputs"]
        speed = np.sqrt(outputs[:, 0] ** 2 + outputs[:, 1] ** 2 + outputs[:, 2] ** 2)
        lambda_zero = speed[inputs[:, 0] < 1e-10]
        lambda_one = speed[inputs[:, 0] > 1 - 1e-10]
        cases = []
        for index in indices:
            lam = inputs[index, 0]
            ux, uy, uz, pressure = outputs[index]
            case = {"sample_index": index, "edges": {}}
            for side, column, normal in (("left", 0, -1.0), ("right", -1, 1.0)):
                openings = edge_openings(lam, column)
                side_rows = []
                for opening_index, rows in enumerate(openings):
                    # Two-column values expose whether only the boundary-adjacent
                    # cell centers happen to resemble a prescribed condition.
                    columns = [0, 1] if side == "left" else [-1, -2]
                    side_rows.append({
                        "opening": opening_index,
                        "row_start": int(rows[0]),
                        "row_end": int(rows[-1]),
                        "outer_column": {
                            "ux": values_stats(ux[rows, column]),
                            "uy": values_stats(uy[rows, column]),
                            "max_abs_uy": float(np.max(np.abs(uy[rows, column]))),
                            "p": values_stats(pressure[rows, column]),
                            "normal_flux_m2_s": float(
                                normal * ux[rows, column].sum() * 0.002 * 0.002
                            ),
                        },
                        "two_column": {
                            "ux": values_stats(ux[np.ix_(rows, columns)]),
                            "uy": values_stats(uy[np.ix_(rows, columns)]),
                            "p": values_stats(pressure[np.ix_(rows, columns)]),
                        },
                    })
                case["edges"][side] = side_rows
            cases.append(case)

    local = json.loads(args.local_cross_validation.read_text())
    local_by_index = {item["sample_index"]: item for item in local["results"]}
    pressure_ratios = [{
        "sample_index": index,
        "local_range": local_by_index[index]["local_pressure_range"],
        "published_range": local_by_index[index]["reference_pressure_range_pa"],
        "local_over_published": (
            local_by_index[index]["local_pressure_range"]
            / local_by_index[index]["reference_pressure_range_pa"]
        ),
    } for index in indices]
    result = {
        "indices": indices,
        "lambda_convention": {
            "lambda_near_0_mean_speed": float(lambda_zero.mean()),
            "lambda_near_1_mean_speed": float(lambda_one.mean()),
            "conclusion": "lambda=0 is fluid; lambda=1 is solid",
        },
        "channel_semantics": {
            "channel_0": "Ux: dominant positive streamwise component",
            "channel_1": "Uy: signed transverse component",
            "channel_2": "Uz: identically zero",
            "channel_3": "pressure: decreases in the streamwise direction",
        },
        "cases": cases,
        "pressure_range_ratios": pressure_ratios,
        "roi_boundary_test": {
            "inlet_expected": "Ux=0.1 and Uy=0 on fluid openings",
            "outlet_expected": "p=0 on fluid openings",
            "interpretation": (
                "Nonconstant inlet Ux or nonzero outlet p at the outermost cell centers "
                "supports, but alone does not prove, an interior observation window."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 3.2 * len(indices)), squeeze=False)
    with np.load(args.reference_root / "data" / "mixer_64.npz") as archive:
        for row, index in enumerate(indices):
            lam = archive["inputs"][index, 0]
            ux, uy, _, pressure = archive["outputs"][index]
            axes[row, 0].imshow(lam, origin="lower", vmin=0, vmax=1, cmap="gray_r")
            axes[row, 0].set_title(f"sample {index}: lambda")
            axes[row, 1].plot(ux[:, 0], np.arange(64), label="left Ux")
            axes[row, 1].axvline(0.1, color="k", linestyle="--", label="BC 0.1")
            axes[row, 1].set_title("Published left-edge Ux")
            axes[row, 1].legend(fontsize=8)
            axes[row, 2].plot(pressure[:, -1], np.arange(64), label="right p")
            axes[row, 2].axvline(0.0, color="k", linestyle="--", label="BC 0")
            axes[row, 2].set_title("Published right-edge pressure")
            axes[row, 2].legend(fontsize=8)
    fig.tight_layout()
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_figure, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
