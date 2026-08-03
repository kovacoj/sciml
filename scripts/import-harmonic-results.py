#!/usr/bin/env python3
"""Validate and compact a precomputed harmonic-oscillator web dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--points", type=int, default=801)
    parser.add_argument("--source-script", type=Path)
    parser.add_argument(
        "--published-script",
        type=Path,
        default=Path("notebooks/harmonic_oscillator_precompute.py"),
    )
    args = parser.parse_args()

    with args.source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    coordinates = np.asarray(payload["coordinates"], dtype=float)
    if coordinates.ndim != 1 or len(coordinates) < args.points:
        raise ValueError("Source coordinate grid is smaller than requested output.")

    indices = np.unique(
        np.linspace(0, len(coordinates) - 1, args.points).round().astype(int)
    )
    payload["coordinates"] = np.round(coordinates[indices], 8).tolist()
    for method in payload["methods"].values():
        for state in method["states"]:
            values = np.asarray(state["values"], dtype=float)
            if values.shape != coordinates.shape:
                raise ValueError("Eigenfunction values do not match coordinate grid.")
            state["values"] = np.round(values[indices], 8).tolist()

    payload["provenance"]["import"] = {
        "script": "scripts/import-harmonic-results.py",
        "sourcePoints": int(len(coordinates)),
        "publishedPoints": int(len(indices)),
    }
    payload["provenance"]["source"] = str(args.published_script)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))

    if args.source_script is not None:
        args.published_script.parent.mkdir(parents=True, exist_ok=True)
        args.published_script.write_text(
            args.source_script.read_text(encoding="utf-8"), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
