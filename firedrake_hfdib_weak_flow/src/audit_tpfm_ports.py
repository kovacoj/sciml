"""Audit invariant edge ports and fluid connectivity in a TPFM dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label


THRESHOLD = 0.5
SPACING = 0.002
EXPECTED_PORT_BANDS = ((8, 16), (48, 56))
FOUR_NEIGHBOR_STRUCTURE = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
)


def contiguous_intervals(mask: np.ndarray, spacing: float = SPACING) -> list[dict]:
    """Convert a row mask to half-open row and cell-boundary y intervals."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return []
    groups = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    return [
        {
            "row_start": int(group[0]),
            "row_end_exclusive": int(group[-1] + 1),
            "physical_y_min": float(group[0] * spacing),
            "physical_y_max": float((group[-1] + 1) * spacing),
        }
        for group in groups
    ]


def _band_status(mask: np.ndarray, start: int, end: int) -> str:
    count = int(mask[start:end].sum())
    if count == 0:
        return "closed"
    if count == end - start:
        return "full"
    return "partial"


def _summarize_band_status(statuses: list[list[list[str]]]) -> dict:
    result = {}
    for side_index, side in enumerate(("left", "right")):
        bands = []
        for band_index, (start, end) in enumerate(EXPECTED_PORT_BANDS):
            values = [sample[side_index][band_index] for sample in statuses]
            bands.append(
                {
                    "rows": [start, end],
                    "full_count": values.count("full"),
                    "partial_count": values.count("partial"),
                    "closed_count": values.count("closed"),
                    "partial_sample_indices": [
                        i for i, value in enumerate(values) if value == "partial"
                    ],
                    "closed_sample_indices": [
                        i for i, value in enumerate(values) if value == "closed"
                    ],
                }
            )
        result[side] = bands
    return result


def _edge_patterns(left: np.ndarray, right: np.ndarray) -> list[dict]:
    packed = np.concatenate((left, right), axis=1)
    unique, first, inverse, counts = np.unique(
        packed, axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.argsort(-counts, kind="stable")
    patterns = []
    ny = left.shape[1]
    for pattern_index in order:
        representative = int(first[pattern_index])
        patterns.append(
            {
                "count": int(counts[pattern_index]),
                "representative_sample_index": representative,
                "sample_indices": np.flatnonzero(inverse == pattern_index).astype(int).tolist(),
                "left_mask_row_0_to_ny_minus_1": "".join(
                    unique[pattern_index, :ny].astype(np.uint8).astype(str)
                ),
                "right_mask_row_0_to_ny_minus_1": "".join(
                    unique[pattern_index, ny:].astype(np.uint8).astype(str)
                ),
            }
        )
    return patterns


def audit_inputs(inputs: np.ndarray) -> tuple[dict, dict]:
    """Compute the complete audit from an archive ``inputs`` array."""
    inputs = np.asarray(inputs)
    if inputs.ndim != 4 or inputs.shape[1] != 1:
        raise ValueError("inputs must have shape (samples, 1, ny, nx)")
    if inputs.shape[0] == 0:
        raise ValueError("inputs must contain at least one sample")
    ny, nx = inputs.shape[2:]
    if ny < max(end for _, end in EXPECTED_PORT_BANDS):
        raise ValueError("inputs must have at least 56 rows for expected port bands")

    fluid = inputs[:, 0] < THRESHOLD
    left = fluid[:, :, 0]
    right = fluid[:, :, -1]
    left_frequency = left.mean(axis=0)
    right_frequency = right.mean(axis=0)
    invariant_left = left.all(axis=0)
    invariant_right = right.all(axis=0)
    invariant_both = invariant_left & invariant_right

    component_counts = []
    left_right_paths = []
    left_port_connectivity = []
    statuses: list[list[list[str]]] = []
    expected_right_rows = np.zeros(ny, dtype=bool)
    for start, end in EXPECTED_PORT_BANDS:
        expected_right_rows[start:end] = True

    for sample in fluid:
        components, component_count = label(sample, structure=FOUR_NEIGHBOR_STRUCTURE)
        component_counts.append(int(component_count))
        left_labels = set(components[:, 0][sample[:, 0]].tolist()) - {0}
        right_labels = set(components[:, -1][sample[:, -1]].tolist()) - {0}
        left_right_paths.append(bool(left_labels & right_labels))
        expected_right_labels = set(
            components[:, -1][sample[:, -1] & expected_right_rows].tolist()
        ) - {0}
        sample_connectivity = []
        for start, end in EXPECTED_PORT_BANDS:
            left_band_labels = set(
                components[start:end, 0][sample[start:end, 0]].tolist()
            ) - {0}
            sample_connectivity.append(bool(left_band_labels & expected_right_labels))
        left_port_connectivity.append(sample_connectivity)
        statuses.append(
            [
                [_band_status(sample[:, column], start, end) for start, end in EXPECTED_PORT_BANDS]
                for column in (0, -1)
            ]
        )

    patterns = _edge_patterns(left, right)
    component_values, component_frequencies = np.unique(
        component_counts, return_counts=True
    )
    failed_paths = [i for i, connected in enumerate(left_right_paths) if not connected]
    failed_ports = [
        [i for i, connected in enumerate(row) if not connected]
        for row in np.asarray(left_port_connectivity, dtype=bool).T
    ]
    report = {
        "sample_count": int(inputs.shape[0]),
        "shape": list(inputs.shape),
        "threshold": {
            "value": THRESHOLD,
            "fluid_predicate": "lambda < 0.5",
            "semantics": (
                "The diffuse field is binarized exactly: every value below 0.5 is fluid, "
                "including intermediate diffuse-interface values; lambda == 0.5 and values "
                "above it are non-fluid for this audit."
            ),
        },
        "spacing": SPACING,
        "edge_fluid_frequency_by_row": {
            "left": left_frequency.tolist(),
            "right": right_frequency.tolist(),
        },
        "invariant_fluid_rows": {
            "left": np.flatnonzero(invariant_left).astype(int).tolist(),
            "right": np.flatnonzero(invariant_right).astype(int).tolist(),
            "left_right_intersection": np.flatnonzero(invariant_both).astype(int).tolist(),
        },
        "invariant_fluid_intervals": {
            "left": contiguous_intervals(invariant_left),
            "right": contiguous_intervals(invariant_right),
            "left_right_intersection": contiguous_intervals(invariant_both),
        },
        "edge_mask_patterns": {
            "distinct_count": len(patterns),
            "exact_match_definition": "concatenated thresholded left and right edge row masks",
            "patterns": patterns,
        },
        "expected_dafoam_port_bands": [
            {
                "rows": [start, end],
                "physical_y_bounds": [start * SPACING, end * SPACING],
            }
            for start, end in EXPECTED_PORT_BANDS
        ],
        "expected_port_edge_status": _summarize_band_status(statuses),
        "per_sample": {
            "fluid_component_count": component_counts,
            "has_any_left_right_path": left_right_paths,
            "expected_left_port_connected_to_any_expected_right_port": left_port_connectivity,
        },
        "connectivity_summary": {
            "component_count_histogram": {
                str(int(value)): int(count)
                for value, count in zip(component_values, component_frequencies)
            },
            "left_right_path_count": int(sum(left_right_paths)),
            "no_left_right_path_count": len(failed_paths),
            "no_left_right_path_sample_indices": failed_paths,
            "expected_left_port_connected_counts": [
                int(np.asarray(left_port_connectivity)[:, band].sum()) for band in range(2)
            ],
            "expected_left_port_not_connected_counts": [len(values) for values in failed_ports],
            "expected_left_port_not_connected_sample_indices": failed_ports,
        },
    }
    plot_data = {
        "left_frequency": left_frequency,
        "right_frequency": right_frequency,
        "invariants": np.stack((invariant_left, invariant_right, invariant_both)),
        "component_counts": np.asarray(component_counts),
        "left_right_paths": np.asarray(left_right_paths),
        "left_port_connectivity": np.asarray(left_port_connectivity),
        "patterns": patterns,
        "ny": ny,
    }
    return report, plot_data


def save_figure(plot_data: dict, path: Path) -> None:
    """Write the four-panel audit summary."""
    ny = plot_data["ny"]
    rows = np.arange(ny)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(plot_data["left_frequency"], rows, label="left")
    axes[0, 0].plot(plot_data["right_frequency"], rows, label="right", linestyle="--")
    axes[0, 0].set(xlim=(-0.02, 1.02), ylim=(-0.5, ny - 0.5), xlabel="fluid frequency", ylabel="row")
    axes[0, 0].set_title("Edge fluid frequency by row")
    axes[0, 0].legend()

    axes[0, 1].imshow(
        plot_data["invariants"], aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1
    )
    axes[0, 1].set_yticks(range(3), labels=["left", "right", "intersection"])
    axes[0, 1].set(xlabel="row", title="Invariant fluid masks across all samples")

    components = plot_data["component_counts"]
    bins = np.arange(components.min(), components.max() + 2) - 0.5
    axes[1, 0].hist(components, bins=bins, color="#52789c", edgecolor="white")
    path_count = int(plot_data["left_right_paths"].sum())
    port_counts = plot_data["left_port_connectivity"].sum(axis=0)
    axes[1, 0].set(xlabel="4-neighbor fluid component count", ylabel="samples")
    axes[1, 0].set_title(
        f"Components; L-R paths {path_count}/{len(components)}; port links {port_counts.tolist()}"
    )

    patterns = plot_data["patterns"]
    shown = patterns[: min(20, len(patterns))]
    pattern_image = np.array(
        [
            [int(value) for value in item["left_mask_row_0_to_ny_minus_1"]]
            + [0]
            + [int(value) for value in item["right_mask_row_0_to_ny_minus_1"]]
            for item in shown
        ]
    ).T
    axes[1, 1].imshow(pattern_image, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].axhline(ny - 0.5, color="black", linewidth=1)
    axes[1, 1].set_xticks(
        range(len(shown)), labels=[str(item["count"]) for item in shown], rotation=45
    )
    axes[1, 1].set_yticks([ny // 2, ny + 1 + ny // 2], labels=["left edge", "right edge"])
    axes[1, 1].set(xlabel="pattern count (up to 20 most common)", title=f"Exact edge masks: {len(patterns)} patterns")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(dataset: Path, output_dir: Path) -> dict:
    with np.load(dataset) as archive:
        if "inputs" not in archive:
            raise ValueError("dataset archive does not contain 'inputs'")
        report, plot_data = audit_inputs(archive["inputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "invariant_port_structure.json"
    figure_path = output_dir / "invariant_port_structure.png"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    save_figure(plot_data, figure_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.dataset, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
