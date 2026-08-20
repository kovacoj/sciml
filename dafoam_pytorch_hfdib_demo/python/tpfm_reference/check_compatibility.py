"""Check TPFM mesh coordinates and lambda/signed-distance reconstruction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def signed_distance(mask: np.ndarray, spacing: float) -> np.ndarray:
    # lambda=1 lies on the negative side of sigma in lambda=(1-tanh(sigma/h))/2.
    return (distance_transform_edt(~mask) - distance_transform_edt(mask)) * spacing


def reconstruct(lam: np.ndarray, spacing: float, width_factor: float) -> dict:
    mask = lam > 0.5
    sigma_mask = signed_distance(mask, spacing)
    sigma = sigma_mask.copy()
    interface = (lam > 1e-10) & (lam < 1 - 1e-10)
    sigma[interface] = width_factor * spacing * np.arctanh(1 - 2 * lam[interface])
    regenerated = 0.5 * (1 - np.tanh(sigma / (width_factor * spacing)))
    return {
        "width_factor": width_factor,
        "interface_cells": int(interface.sum()),
        "relative_lambda_l2": float(
            np.linalg.norm(regenerated - lam) / (np.linalg.norm(lam) + 1e-30)
        ),
        "max_lambda_error": float(np.max(np.abs(regenerated - lam))),
        "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", default="0,274,549")
    args = parser.parse_args()
    with (args.reference_root / "data" / "coordinates_64.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    reference_xyz = np.array(
        [[float(row[axis]) for axis in ("x", "y", "z")] for row in rows]
    )
    mesh = np.load(args.mesh)
    local_xyz = mesh["cell_centres"]
    spacing = float(np.median(np.diff(np.unique(reference_xyz[:, 0]))))
    with np.load(args.reference_root / "data" / "mixer_64.npz") as data:
        inputs = data["inputs"]
        reconstructions = []
        for index in [int(value) for value in args.samples.split(",")]:
            lam = inputs[index, 0]
            reconstructions.append({
                "sample_index": index,
                "paper_width_h": reconstruct(lam, spacing, 1.0),
                "local_width_1_5h": reconstruct(lam, spacing, 1.5),
            })
    max_error = float(np.max(np.abs(local_xyz - reference_xyz)))
    result = {
        "same_n_cells": len(local_xyz) == len(reference_xyz),
        "same_dx": bool(np.isclose(spacing, 0.002)),
        "same_dy": bool(np.isclose(np.median(np.diff(np.unique(reference_xyz[:, 1]))), 0.002)),
        "same_domain_extent": bool(
            np.allclose(local_xyz.min(axis=0), reference_xyz.min(axis=0))
            and np.allclose(local_xyz.max(axis=0), reference_xyz.max(axis=0))
        ),
        "coordinate_max_error": max_error,
        "coordinates_match": bool(max_error < 1e-12),
        "lambda_reconstruction": reconstructions,
        "warning": "Local HFDIB uses a 1.5*h tanh width; the reference paper states h.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["coordinates_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
