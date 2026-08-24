"""Measure SIMPLE work to a common residual tolerance from five initial states."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import PROJECT_ROOT, hfdib_signed_distance_options  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from diagnostics.warmstart_common import (  # noqa: E402
    build_state_assembler, predict_full_state, relative_field_errors)
from state_layout import build_isothermal_layout  # noqa: E402
from unet.generate_case import write_signed_distance_file  # noqa: E402

METHODS = ("cold", "teacher_k20", "neural", "neural_t1", "neural_t5")
EPS = 1e-30


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def residual_metrics(residual, cold_norm, n_cells):
    n_u = 3 * n_cells
    blocks = {
        "rho_residual": np.linalg.norm(residual),
        "rho_u": np.linalg.norm(residual[:n_u]),
        "rho_p": np.linalg.norm(residual[n_u:n_u + n_cells]),
        "rho_phi": np.linalg.norm(residual[n_u + n_cells:]),
    }
    return {name: float(value / (cold_norm + EPS)) for name, value in blocks.items()}


def run_to_convergence(bridge, initial_state, cold_norm, threshold, reference,
                       n_cells, max_steps, save_every):
    state = initial_state.copy()
    history = []
    simple_time = 0.0
    status = "MAX_STEPS"
    for step in range(max_steps + 1):
        residual = bridge.residual(state)
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(state)):
            status = "NONFINITE"
            break
        rmetrics = residual_metrics(residual, cold_norm, n_cells)
        fields = relative_field_errors(state, reference, n_cells)
        if step % save_every == 0 or rmetrics["rho_residual"] <= threshold or step == max_steps:
            history.append({"simple_steps": step, **rmetrics, **fields})
        if rmetrics["rho_residual"] <= threshold:
            status = "CONVERGED"
            break
        if rmetrics["rho_residual"] > 1e8:
            status = "DIVERGED"
            break
        if step < max_steps:
            started = time.perf_counter()
            state = bridge.simple_step(state)
            simple_time += time.perf_counter() - started
    final = history[-1] if history else {**rmetrics, **relative_field_errors(state, reference, n_cells)}
    return status, step, simple_time, final, history


def load_model(checkpoint: Path):
    import torch
    from unet.factory import build_model
    torch.set_default_dtype(torch.float64)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(payload.get("architecture", "simple"), **payload.get("model_kwargs", {}))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def selected_cases(ds_dir: Path, value: str):
    splits = json.loads((ds_dir / "splits.json").read_text())
    frozen = splits["test"]
    if not value:
        return frozen
    requested = [part.strip() for part in value.split(",") if part.strip()]
    normalized = [part if part.startswith("topology_") else f"topology_{int(part):04d}" for part in requested]
    if any(case not in frozen for case in normalized):
        raise ValueError(f"test-indices must be within frozen cases {frozen}")
    return normalized


def aggregate(output_dir: Path) -> None:
    rows = []
    for case_dir in sorted(output_dir.glob("topology_*")):
        for method in METHODS:
            path = case_dir / f"{method}.json"
            if path.is_file():
                payload = json.loads(path.read_text())
                rows.append({key: value for key, value in payload.items() if key != "history"})
    if not rows:
        return
    columns = ["case", "method", "initial_rho", "threshold", "status", "pre_steps",
               "remaining_steps", "total_simple_steps", "inference_time_s",
               "precorrection_time_s", "remaining_simple_time_s", "total_time_s",
               "final_rho", "final_rel_u", "final_rel_p"]
    with (output_dir / "warm_start_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    cold = {row["case"]: row for row in rows if row["method"] == "cold" and row["status"] == "CONVERGED"}
    summary = {"n_rows": len(rows), "n_complete_cases": sum((output_dir / case / "DONE.json").is_file() for case in cold)}
    for method in METHODS:
        matching = [row for row in rows if row["method"] == method and row["case"] in cold and row["status"] == "CONVERGED"]
        iteration_savings = [1 - row["total_simple_steps"] / cold[row["case"]]["total_simple_steps"] for row in matching]
        wall_speedups = [cold[row["case"]]["total_time_s"] / row["total_time_s"] for row in matching if row["total_time_s"] > 0]
        summary[method] = {
            "converged": len(matching),
            "mean_iteration_saving": float(np.mean(iteration_savings)) if iteration_savings else None,
            "median_iteration_saving": float(np.median(iteration_savings)) if iteration_savings else None,
            "mean_wall_speedup": float(np.mean(wall_speedups)) if wall_speedups else None,
            "median_wall_speedup": float(np.median(wall_speedups)) if wall_speedups else None,
            "min_wall_speedup": float(np.min(wall_speedups)) if wall_speedups else None,
            "max_wall_speedup": float(np.max(wall_speedups)) if wall_speedups else None,
            "wall_speedup_cases": sum(value > 1 for value in wall_speedups),
        }
    atomic_json(output_dir / "warm_start_summary.json", summary)
    threshold_cases = {}
    for path in sorted(output_dir.glob("topology_*/threshold.json")):
        payload = json.loads(path.read_text())
        threshold_cases[payload["case"]] = {
            "rho_reference": payload["rho_reference"],
            "threshold": payload["threshold"],
        }
    atomic_json(Path(PROJECT_ROOT) / "outputs/final_dafoam/convergence_thresholds.json", {
        "definition": "max(10*rho_reference, residual_rel_tol)",
        "residual_rel_tol": 1e-6,
        "cases": threshold_cases,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tpfm_64_local_dafoam")
    parser.add_argument("--checkpoint", default="outputs/tpfm_local_distill_seed22/checkpoint.pt")
    parser.add_argument("--test-indices", default="")
    parser.add_argument("--output-dir", default="outputs/warmstart_convergence")
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--residual-rel-tol", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    absolute = lambda value: Path(value) if Path(value).is_absolute() else Path(PROJECT_ROOT) / value
    ds_dir, checkpoint, output_dir = map(absolute, (args.dataset, args.checkpoint, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint)
    from pinn.mesh_metadata import MeshMetadata
    mesh = MeshMetadata.load(str(ds_dir / "shared/mesh_metadata.npz"), str(ds_dir / "shared/mesh_metadata.json"))
    w0 = np.load(ds_dir / "shared/base_state_k0.npy")
    assembler, _ = build_state_assembler(mesh, build_isothermal_layout(mesh.n_cells, mesh.n_faces), w0)
    previous = json.loads((Path(PROJECT_ROOT) / "outputs/final_dafoam/warm_start_raw.json").read_text())
    frozen_cases = json.loads((ds_dir / "splits.json").read_text())["test"]

    from mpi4py import MPI
    for case in selected_cases(ds_dir, args.test_indices):
        case_output = output_dir / case
        if args.resume and (case_output / "DONE.json").is_file():
            continue
        topology_dir = ds_dir / case
        case_dir = topology_dir / "case"
        subprocess.run(["blockMesh", "-case", str(case_dir)], check=True, capture_output=True)
        write_signed_distance_file(str(case_dir), np.load(topology_dir / "signed_distance.npy"))
        os.chdir(case_dir)
        bridge = DAFoamResidualBridge(str(case_dir), hfdib_signed_distance_options(
            str(case_dir), inlet_patches=["inletLower", "inletUpper"],
            outlet_patches=["outletLower", "outletUpper"]), comm=MPI.COMM_SELF)
        bridge.residual(w0)  # untimed warm-up
        bridge.solver()
        reference = np.ascontiguousarray(bridge.solver.getStates().copy(), dtype=np.float64)
        cold_norm = float(np.linalg.norm(bridge.residual(w0)))
        rho_reference = float(np.linalg.norm(bridge.residual(reference)) / (cold_norm + EPS))
        threshold = max(10 * rho_reference, args.residual_rel_tol)
        atomic_json(case_output / "threshold.json", {
            "case": case, "rho_reference": rho_reference, "threshold": threshold,
            "definition": "max(10*rho_reference, residual_rel_tol)",
            "residual_rel_tol": args.residual_rel_tol,
        })

        neural, inference_time = predict_full_state(model, np.load(topology_dir / "lambda.npy"), assembler)
        actual = relative_field_errors(neural, reference, mesh.n_cells)["rel_u"]
        if case in previous:
            expected = previous[case]["rel_errors"]["rel_u"]
            if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
                raise AssertionError(f"{case}: neural assembly regression {actual} != {expected}")
        teacher = np.load(ds_dir / "solver_targets" / case / "state_k020.npy")
        starts = {"cold": w0, "teacher_k20": teacher, "neural": neural}
        pre_times = {"cold": 0.0, "teacher_k20": 0.0, "neural": 0.0}
        state = neural.copy()
        elapsed = 0.0
        for step in range(1, 6):
            started = time.perf_counter()
            state = bridge.simple_step(state)
            elapsed += time.perf_counter() - started
            if step == 1:
                starts["neural_t1"] = state.copy()
                pre_times["neural_t1"] = elapsed
        starts["neural_t5"] = state.copy()
        pre_times["neural_t5"] = elapsed
        rotation = frozen_cases.index(case) % len(METHODS)
        order = list(METHODS[rotation:] + METHODS[:rotation])
        for method in order:
            path = case_output / f"{method}.json"
            if args.resume and path.is_file():
                continue
            initial_residual = bridge.residual(starts[method])
            initial_rho = float(np.linalg.norm(initial_residual) / (cold_norm + EPS))
            status, remaining, simple_time, final, history = run_to_convergence(
                bridge, starts[method], cold_norm, threshold, reference,
                mesh.n_cells, args.max_steps, args.save_every)
            pre_steps = {"neural_t1": 1, "neural_t5": 5}.get(method, 0)
            # Teacher generation cost is reported separately from remaining work.
            total_steps = remaining + pre_steps + (20 if method == "teacher_k20" else 0)
            total_time = simple_time + pre_times.get(method, 0.0) + (inference_time if method.startswith("neural") else 0.0)
            atomic_json(path, {
                "case": case, "method": method, "status": status,
                "rho_reference": rho_reference,
                "initial_rho": initial_rho, "threshold": threshold,
                "pre_steps": pre_steps, "remaining_steps": remaining,
                "total_simple_steps": total_steps,
                "inference_time_s": inference_time if method.startswith("neural") else 0.0,
                "precorrection_time_s": pre_times.get(method, 0.0),
                "remaining_simple_time_s": simple_time, "total_time_s": total_time,
                "final_rho": final["rho_residual"], "final_rel_u": final["rel_u"],
                "final_rel_p": final["rel_p"], "execution_order": order, "history": history,
            })
        if all((case_output / f"{method}.json").is_file() for method in METHODS):
            atomic_json(case_output / "DONE.json", {"case": case, "status": "complete", "methods": METHODS})
        del bridge
    aggregate(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
