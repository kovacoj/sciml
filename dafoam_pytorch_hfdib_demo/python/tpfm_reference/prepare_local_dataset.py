"""Prepare official TPFM lambdas for the explicitly local DAFoam formulation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT
from tpfm_reference.cross_validate_cfd import reconstruct_sigma
from unet.generate_case import write_signed_distance_file


def ids(start: int, count: int, excluded: set[int]) -> list[int]:
    result = []
    current = start
    while len(result) < count:
        if current not in excluded:
            result.append(current)
        current += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", default="datasets/tpfm_64_local_dafoam", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    project = Path(PROJECT_ROOT)
    output = args.output if args.output.is_absolute() else project / args.output
    template = project / "cases/four_port_64x64"
    excluded = {0, 274, 549}
    split_indices = {
        "train": ids(1, 256, excluded),
        "validation": ids(257, 32, excluded),
        "test": ids(290, 16, excluded),
    }
    selected = split_indices["train"] + split_indices["validation"] + split_indices["test"]
    with np.load(args.reference_root / "data/mixer_64.npz") as archive:
        lambdas = archive["inputs"][selected, 0].copy()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project / "datasets/four_port_64/shared", output / "shared", dirs_exist_ok=True)
    for position, index in enumerate(selected):
        topology_id = f"topology_{index:04d}"
        directory = output / topology_id
        if args.skip_existing and (directory / "case/constant/hfdibGeometry/signedDistance").is_file():
            continue
        case = directory / "case"
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(template, case)
        shutil.rmtree(case / "constant/polyMesh", ignore_errors=True)
        subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True)
        lam = lambdas[position]
        sigma = reconstruct_sigma(lam, 0.002, 1.0)
        write_signed_distance_file(str(case), sigma)
        np.save(directory / "lambda.npy", lam)
        np.save(directory / "signed_distance.npy", sigma)
        (directory / "source.json").write_text(json.dumps({
            "tpfm_sample_index": index,
            "uses_published_flow_labels": False,
            "formulation": "local 64x64 DAFoam/HFDIB",
        }, indent=2) + "\n")
    splits = {name: [f"topology_{index:04d}" for index in values] for name, values in split_indices.items()}
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    manifest = {
        "source": "techMathGroup/tpfm_unet mixer_64.npz inputs only",
        "classification": "TPFM_TOPOLOGIES_ONLY",
        "train_count": 256,
        "validation_count": 32,
        "test_count": 16,
        "excluded_reproduction_indices": sorted(excluded),
        "sigma_width_factor": 1.0,
        "uses_converged_cfd_labels_for_training": False,
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
