"""Cross-validate local HFDIB solves against three published TPFM fields."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import PROJECT_ROOT, hfdib_signed_distance_options  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from unet.generate_case import write_signed_distance_file  # noqa: E402


def reconstruct_sigma(lam: np.ndarray, h: float, width_factor: float) -> np.ndarray:
    solid = lam > 0.5
    sigma = (
        distance_transform_edt(~solid, sampling=(h, h))
        - distance_transform_edt(solid, sampling=(h, h))
    )
    interface = (lam > 1e-10) & (lam < 1 - 1e-10)
    sigma[interface] = width_factor * h * np.arctanh(1 - 2 * lam[interface])
    return sigma


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def transformed_fields(field: np.ndarray):
    yield "identity", field
    yield "transpose", field.T
    yield "flip_x", field[:, ::-1]
    yield "flip_y", field[::-1, :]
    yield "transpose_flip_x", field.T[:, ::-1]
    yield "transpose_flip_y", field.T[::-1, :]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--indices", default="0,274,549")
    parser.add_argument("--work-dir", default="datasets/tpfm_64_crossval", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sigma-width-factor", type=float, default=1.0)
    args = parser.parse_args()
    project = Path(PROJECT_ROOT)
    work_dir = args.work_dir if args.work_dir.is_absolute() else project / args.work_dir
    template = project / "cases" / "four_port_64x64"
    indices = [int(value) for value in args.indices.split(",")]
    with np.load(args.reference_root / "data" / "mixer_64.npz") as archive:
        inputs = archive["inputs"][indices].copy()
        outputs = archive["outputs"][indices].copy()

    results = []
    h = 0.002
    for position, index in enumerate(indices):
        sample_dir = work_dir / f"sample_{index:03d}"
        case_dir = sample_dir / "case"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        shutil.copytree(template, case_dir)
        shutil.rmtree(case_dir / "constant" / "polyMesh", ignore_errors=True)
        subprocess.run(["blockMesh", "-case", str(case_dir)], check=True, capture_output=True)
        lam = inputs[position, 0]
        sigma = reconstruct_sigma(lam, h, args.sigma_width_factor)
        write_signed_distance_file(str(case_dir), sigma)
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "lambda_reference.npy", lam)
        np.save(sample_dir / "signed_distance.npy", sigma)

        previous = Path.cwd()
        os.chdir(case_dir)
        from mpi4py import MPI
        start = time.perf_counter()
        bridge = DAFoamResidualBridge(
            str(case_dir),
            hfdib_signed_distance_options(
                str(case_dir),
                inlet_patches=["inletLower", "inletUpper"],
                outlet_patches=["outletLower", "outletUpper"],
            ),
            comm=MPI.COMM_SELF,
        )
        bridge.solver()
        elapsed = time.perf_counter() - start
        state = np.asarray(bridge.solver.getStates().copy(), dtype=np.float64)
        os.chdir(previous)
        del bridge

        n_cells = 4096
        velocity = state[:3 * n_cells].reshape(n_cells, 3).reshape(64, 64, 3)
        pressure = state[3 * n_cells:4 * n_cells].reshape(64, 64)
        reference_u = outputs[position, :2]
        reference_p_pa = outputs[position, 3]

        orientation_errors = []
        for name, ux in transformed_fields(reference_u[0]):
            uy = dict(transformed_fields(reference_u[1]))[name]
            orientation_errors.append({
                "orientation": name,
                "rel_u": relative(
                    np.stack([velocity[:, :, 0], velocity[:, :, 1]]),
                    np.stack([ux, uy]),
                ),
            })
        best = min(orientation_errors, key=lambda item: item["rel_u"])

        # OpenFOAM incompressible p is kinematic pressure. Compare the published
        # pressure both directly and after the standard rho=1000 Pa conversion,
        # with each field gauge-centered.
        p_local = pressure - pressure.mean()
        p_ref = reference_p_pa - reference_p_pa.mean()
        result = {
            "sample_index": index,
            "wall_time_s": elapsed,
            "velocity_identity_rel_l2": orientation_errors[0]["rel_u"],
            "velocity_best_orientation": best,
            "pressure_rel_l2_direct_gauge_centered": relative(p_local, p_ref),
            "pressure_rel_l2_rho1000_gauge_centered": relative(p_local, p_ref / 1000.0),
            "local_velocity_max": float(np.linalg.norm(velocity[:, :, :2], axis=2).max()),
            "reference_velocity_max": float(np.linalg.norm(reference_u, axis=0).max()),
            "local_pressure_range": float(pressure.max() - pressure.min()),
            "reference_pressure_range_pa": float(reference_p_pa.max() - reference_p_pa.min()),
            "orientation_errors": orientation_errors,
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "indices": indices,
        "geometry_reconstruction": (
            "mask EDT plus "
            f"{args.sigma_width_factor}*h*atanh(1-2lambda) in interface cells"
        ),
        "results": results,
        "mean_velocity_identity_rel_l2": float(np.mean([
            item["velocity_identity_rel_l2"] for item in results
        ])),
        "passes_velocity_20_percent": all(
            item["velocity_identity_rel_l2"] < 0.2 for item in results
        ),
        "decision": "PASS" if all(
            item["velocity_identity_rel_l2"] < 0.2 for item in results
        ) else "STOP_PDE_MISMATCH",
    }
    output = args.output if args.output.is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
