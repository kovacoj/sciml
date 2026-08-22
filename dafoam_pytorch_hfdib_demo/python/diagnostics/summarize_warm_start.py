"""Summarize warm-start evaluation metrics across held-out topologies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    topology_ids = sorted(metrics)
    field_u = [metrics[tid]["rel_errors"]["rel_u"] for tid in topology_ids]
    field_p = [metrics[tid]["rel_errors"]["rel_p"] for tid in topology_ids]
    pressure_ratio = [
        metrics[tid]["article_metrics"]["pressure_drop_ratio"]
        for tid in topology_ids
    ]

    steps = sorted({
        row["k"]
        for tid in topology_ids
        for row in metrics[tid]["warm_start"]
    })
    warm = {}
    for step in steps:
        cold = []
        neural = []
        for tid in topology_ids:
            rows = metrics[tid]["warm_start"]
            cold.append(next(row["rel_u"] for row in rows
                             if row["start"] == "W0" and row["k"] == step))
            neural_label = "W_NN" if any(r["start"] == "W_NN" for r in rows) else "W_ENS"
            neural.append(next(row["rel_u"] for row in rows
                               if row["start"] == neural_label and row["k"] == step))
        warm[str(step)] = {
            "cold_rel_u": stats(cold),
            "neural_rel_u": stats(neural),
            "neural_better_count": int(np.sum(np.asarray(neural) < np.asarray(cold))),
            "median_cold_over_neural": float(np.median(
                np.asarray(cold) / (np.asarray(neural) + 1e-30))),
        }

    result = {
        "n_topologies": len(topology_ids),
        "topology_ids": topology_ids,
        "network_field_rel_u": stats(field_u),
        "network_field_rel_p": stats(field_p),
        "pressure_drop_ratio": stats(pressure_ratio),
        "warm_start": warm,
        "worst_initial_topology": topology_ids[int(np.argmax(field_u))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
