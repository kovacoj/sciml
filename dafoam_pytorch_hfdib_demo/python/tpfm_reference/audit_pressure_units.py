"""Report pressure scales without assuming a Pa/kinematic conversion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--cross-validation", required=True, type=Path)
    parser.add_argument("--indices", default="0,274,549")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",")]
    local = json.loads(args.cross_validation.read_text())
    by_index = {entry["sample_index"]: entry for entry in local["results"]}
    with np.load(args.reference_root / "data/mixer_64.npz") as archive:
        pressure = archive["outputs"][indices, 3]
    cases = []
    for position, index in enumerate(indices):
        published = pressure[position]
        published_range = float(np.ptp(published))
        local_range = float(by_index[index]["local_pressure_range"])
        cases.append({
            "sample_index": index,
            "published_pressure_range": published_range,
            "local_openfoam_kinematic_pressure_range": local_range,
            "ratio_direct": local_range / published_range,
            "ratio_x1000": local_range / (published_range * 1000.0),
            "ratio_div1000": local_range / (published_range / 1000.0),
            "streamwise_pressure_trend": float(published[:, -1].mean() - published[:, 0].mean()),
            "outlet_adjacent_values": {
                "mean": float(published[:, -1].mean()),
                "min": float(published[:, -1].min()),
                "max": float(published[:, -1].max()),
            },
            "gauge_centered_shape_error_direct": by_index[index]["pressure_rel_l2_direct_gauge_centered"],
        })
    result = {
        "default_interpretation": "Published p tilde is kinematic pressure, following article terminology.",
        "unit_conversion_applied_to_range": None,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
