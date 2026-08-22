"""Cross-seed comparison of warm-start evaluation results.

Loads all seed warm-start JSONs and computes per-step statistics
across seeds, identifies consistently hard/easy topologies, and
compares against an optional ensemble result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_seed_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_per_step(data: dict, step: int, start_label: str) -> list[float]:
    vals = []
    for tid in sorted(data):
        for row in data[tid]["warm_start"]:
            if row["start"] == start_label and row["k"] == step:
                vals.append(row["rel_u"])
                break
    return vals


def extract_field_errors(data: dict) -> tuple[list[float], list[float]]:
    rel_us = [data[tid]["rel_errors"]["rel_u"] for tid in sorted(data)]
    rel_ps = [data[tid]["rel_errors"]["rel_p"] for tid in sorted(data)]
    return rel_us, rel_ps


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path,
                        help="Output JSON path for cross-seed summary")
    parser.add_argument("--seeds", nargs="+", required=True,
                        help="Pairs of label:path, e.g. seed11:path.json")
    parser.add_argument("--ensemble", type=Path, default=None,
                        help="Optional ensemble warm-start JSON")
    args = parser.parse_args()

    seed_data = {}
    for spec in args.seeds:
        label, path_str = spec.split(":", 1)
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        seed_data[label] = load_seed_metrics(path)

    topology_ids = sorted(next(iter(seed_data.values())).keys())
    simple_steps = sorted({
        row["k"]
        for data in seed_data.values()
        for tid in topology_ids
        for row in data[tid]["warm_start"]
        if row["start"] == "W_NN" or row["start"] == "W_ENS"
    })

    # Per-seed field error statistics
    per_seed = {}
    for label, data in seed_data.items():
        rel_us, rel_ps = extract_field_errors(data)
        per_seed[label] = {
            "field_rel_u": stats(rel_us),
            "field_rel_p": stats(rel_ps),
        }

    # Per-step cross-seed statistics (neural start)
    cross_step = {}
    for step in simple_steps:
        seed_means = []
        per_seed_step = {}
        for label, data in seed_data.items():
            vals = extract_per_step(data, step, "W_NN")
            if vals:
                s = stats(vals)
                per_seed_step[label] = s
                seed_means.append(s["mean"])

        cross_step[str(step)] = {
            "per_seed": per_seed_step,
            "cross_seed_mean_of_means": float(np.mean(seed_means)) if seed_means else None,
            "cross_seed_std_of_means": float(np.std(seed_means)) if seed_means else None,
        }

    # Per-topology difficulty ranking (average across seeds at k=0)
    topo_difficulty = {}
    for tid in topology_ids:
        vals = []
        for label, data in seed_data.items():
            if tid in data:
                vals.append(data[tid]["rel_errors"]["rel_u"])
        topo_difficulty[tid] = {
            "mean_rel_u": float(np.mean(vals)),
            "std_rel_u": float(np.std(vals)),
            "max_rel_u": float(np.max(vals)),
        }

    # Worst topologies
    worst = sorted(topo_difficulty.items(),
                   key=lambda x: x[1]["mean_rel_u"], reverse=True)[:5]
    best = sorted(topo_difficulty.items(),
                  key=lambda x: x[1]["mean_rel_u"])[:5]

    # Ensemble comparison
    ensemble = None
    if args.ensemble and args.ensemble.exists():
        ens_data = load_seed_metrics(args.ensemble)
        ens_rel_us, ens_rel_ps = extract_field_errors(ens_data)
        ensemble = {
            "field_rel_u": stats(ens_rel_us),
            "field_rel_p": stats(ens_rel_ps),
        }
        ens_steps = {}
        for step in simple_steps:
            ens_vals = extract_per_step(ens_data, step, "W_ENS")
            if ens_vals:
                ens_steps[str(step)] = stats(ens_vals)
        ensemble["per_step"] = ens_steps

    result = {
        "n_seeds": len(seed_data),
        "seed_labels": list(seed_data.keys()),
        "n_topologies": len(topology_ids),
        "simple_steps": simple_steps,
        "per_seed_summary": per_seed,
        "cross_step_summary": cross_step,
        "topology_difficulty": topo_difficulty,
        "worst_topologies": [
            {"topology_id": tid, **d} for tid, d in worst
        ],
        "best_topologies": [
            {"topology_id": tid, **d} for tid, d in best
        ],
        "ensemble": ensemble,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"CROSS-SEED SUMMARY ({len(seed_data)} seeds, "
          f"{len(topology_ids)} topologies)")
    print(f"{'='*60}")

    print(f"\n{'seed':<10} {'mean rel_U':>12} {'std':>8} "
          f"{'median':>8} {'max':>8}")
    for label in seed_data:
        s = per_seed[label]["field_rel_u"]
        print(f"{label:<10} {s['mean']:>12.4f} {s['std']:>8.4f} "
              f"{s['median']:>8.4f} {s['max']:>8.4f}")

    if ensemble:
        es = ensemble["field_rel_u"]
        print(f"{'ENSEMBLE':<10} {es['mean']:>12.4f} {es['std']:>8.4f} "
              f"{es['median']:>8.4f} {es['max']:>8.4f}")

    print(f"\nWarm-start rel_U by step (mean across topologies, "
          f"mean across seeds):")
    print(f"{'step':<6} {'mean':>10} {'std':>8}")
    for step in simple_steps:
        s = cross_step[str(step)]
        m = s["cross_seed_mean_of_means"]
        sd = s["cross_seed_std_of_means"]
        if m is not None:
            print(f"k={step:<4} {m:>10.4f} {sd:>8.4f}")

    print(f"\nWorst topologies (by mean rel_U across seeds):")
    for entry in worst[:3]:
        print(f"  {entry[0]}: mean={entry[1]['mean_rel_u']:.4f} "
              f"std={entry[1]['std_rel_u']:.4f}")

    print(f"\nBest topologies:")
    for entry in best[:3]:
        print(f"  {entry[0]}: mean={entry[1]['mean_rel_u']:.4f} "
              f"std={entry[1]['std_rel_u']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
