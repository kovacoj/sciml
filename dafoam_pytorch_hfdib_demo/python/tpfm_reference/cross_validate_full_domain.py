"""Cross-validate a controlled full-domain reconstruction against TPFM ROI fields."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import PROJECT_ROOT, hfdib_signed_distance_options  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from tpfm_reference.full_domain_case import create_case, full_sigma, write_sigma  # noqa: E402
from tpfm_reference.roi_mapping import (  # noqa: E402
    build_roi_cell_indices,
    extract_roi_pressure,
    extract_roi_velocity,
)


def relative(actual: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(actual - reference) / (np.linalg.norm(reference) + 1e-30))


def save_figure(path: Path, index: int, lam, reference_u, local_u, reference_p, local_p) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fields = (
        (lam, "lambda"),
        (np.linalg.norm(reference_u, axis=0), "published |U|"),
        (np.linalg.norm(local_u, axis=0), "DAFoam |U|"),
        (local_u[0] - reference_u[0], "Ux error"),
        (reference_p - reference_p.mean(), "published p'"),
        (local_p - local_p.mean(), "DAFoam p'"),
    )
    for axis, (field, title) in zip(axes.flat, fields):
        image = axis.imshow(field, origin="lower")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle(f"TPFM sample {index}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--indices", default="0,274,549")
    parser.add_argument("--extension-cells", required=True, type=int, choices=(8, 16, 32))
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sigma-width-factor", default=1.0, type=float)
    args = parser.parse_args()
    project = Path(PROJECT_ROOT)
    work = args.work_dir if args.work_dir.is_absolute() else project / args.work_dir
    output = args.output if args.output.is_absolute() else project / args.output
    indices = [int(value) for value in args.indices.split(",")]
    with np.load(args.reference_root / "data/mixer_64.npz") as archive:
        inputs = archive["inputs"][indices].copy()
        outputs = archive["outputs"][indices].copy()

    results = []
    for position, index in enumerate(indices):
        sample = work / f"sample_{index:03d}_e{args.extension_cells:02d}"
        case = sample / "case"
        metadata = create_case(project / "cases/four_port_64x64", case, args.extension_cells)
        subprocess.run(["blockMesh", "-case", str(case)], check=True, capture_output=True, text=True)
        sigma = full_sigma(inputs[position, 0], args.extension_cells, args.sigma_width_factor)
        write_sigma(case, sigma)
        sample.mkdir(parents=True, exist_ok=True)
        np.save(sample / "lambda_roi.npy", inputs[position, 0])
        np.save(sample / "signed_distance_full.npy", sigma)
        roi_indices = build_roi_cell_indices(metadata, metadata["roi_bounds"])
        np.save(sample / "roi_cell_indices.npy", roi_indices)

        previous = Path.cwd()
        start = time.perf_counter()
        try:
            os.chdir(case)
            from mpi4py import MPI
            bridge = DAFoamResidualBridge(
                str(case),
                hfdib_signed_distance_options(
                    str(case),
                    inlet_patches=["inletLower", "inletUpper"],
                    outlet_patches=["outletLower", "outletUpper"],
                ),
                comm=MPI.COMM_SELF,
            )
            bridge.solver()
            state = np.asarray(bridge.solver.getStates().copy(), dtype=np.float64)
        finally:
            os.chdir(previous)
        elapsed = time.perf_counter() - start
        n_cells = metadata["n_cells"]
        velocity = state[:3 * n_cells].reshape(n_cells, 3)
        pressure = state[3 * n_cells:4 * n_cells]
        local_u = extract_roi_velocity({"U": velocity}, roi_indices)
        local_p = extract_roi_pressure({"p": pressure}, roi_indices)
        reference_u = outputs[position, :2]
        reference_p = outputs[position, 3]
        local_range = float(np.ptp(local_p))
        reference_range = float(np.ptp(reference_p))
        inlet_ids = np.r_[np.arange(8, 16), np.arange(48, 56)] * metadata["nx"]
        outlet_ids = inlet_ids + metadata["nx"] - 1
        inlet_flux = float(velocity[inlet_ids, 0].sum() * 0.002 * 0.002)
        outlet_flux = float(velocity[outlet_ids, 0].sum() * 0.002 * 0.002)
        result = {
            "sample_index": index,
            "velocity_rel_l2": relative(local_u, reference_u),
            "pressure_gauge_centered_rel_l2": relative(local_p - local_p.mean(), reference_p - reference_p.mean()),
            "local_velocity_max": float(np.linalg.norm(local_u, axis=0).max()),
            "reference_velocity_max": float(np.linalg.norm(reference_u, axis=0).max()),
            "local_pressure_range": local_range,
            "reference_pressure_range": reference_range,
            "pressure_range_ratio": local_range / reference_range,
            "inlet_flux": inlet_flux,
            "outlet_flux": outlet_flux,
            "simple_iterations": 5000,
            "wall_time_s": elapsed,
            "orientation": "identity",
        }
        results.append(result)
        save_figure(sample / "comparison.png", index, inputs[position, 0], reference_u, local_u, reference_p, local_p)
        print(json.dumps(result, indent=2), flush=True)
        del bridge

    summary = {
        "classification_pending": True,
        "extension_cells_each_side": args.extension_cells,
        "sigma_width_factor": args.sigma_width_factor,
        "roi_bounds": [0.0, 0.128, 0.0, 0.128],
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
