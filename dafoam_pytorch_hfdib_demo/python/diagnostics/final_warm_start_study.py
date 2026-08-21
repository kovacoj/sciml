"""Run and summarize the paired five-start residual-controlled CFD study."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import PROJECT_ROOT
from diagnostics.final_warm_start_worker import METHODS


TIME_PATTERN = re.compile(r"^Time = (\d+)", re.MULTILINE)


def gauge_corrected_pressure_error(candidate: np.ndarray, reference: np.ndarray,
                                   n_cells: int) -> float:
    n_u = 3 * n_cells
    a = candidate[n_u:n_u + n_cells]
    b = reference[n_u:n_u + n_cells]
    difference = (a - np.mean(a)) - (b - np.mean(b))
    centered_reference = b - np.mean(b)
    return float(np.linalg.norm(difference) / (np.linalg.norm(centered_reference) + 1e-30))


def velocity_error(candidate: np.ndarray, reference: np.ndarray, n_cells: int) -> float:
    a = candidate[:3 * n_cells].reshape(n_cells, 3)[:, :2]
    b = reference[:3 * n_cells].reshape(n_cells, 3)[:, :2]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/four_port_64"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-set", type=Path,
                        default=Path("outputs/final_campaign/test_set.json"))
    parser.add_argument("--teacher-state-dir", type=Path,
                        default=Path("outputs/final_warmstart_study/teacher_states"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/final_warmstart_study/convergence"))
    parser.add_argument("--residual-tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    project = Path(PROJECT_ROOT)
    absolute = lambda path: path if path.is_absolute() else project / path
    dataset = absolute(args.dataset)
    checkpoint = absolute(args.checkpoint)
    test_set = absolute(args.test_set)
    teacher_dir = absolute(args.teacher_state_dir)
    output_dir = absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_ids = json.loads(test_set.read_text())["test_topology_ids"]
    n_cells = int(json.loads((dataset / "shared/mesh_metadata.json").read_text())["n_cells"])

    runs = []
    for topology_id in test_ids:
        for method in METHODS:
            run_dir = output_dir / topology_id / method
            result_path = run_dir / "result.json"
            log_path = run_dir / "solver.log"
            work_dir = run_dir / "case"
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-m", "diagnostics.final_warm_start_worker",
                "--dataset", str(dataset), "--checkpoint", str(checkpoint),
                "--teacher-state", str(teacher_dir / f"{topology_id}_k020.npy"),
                "--topology-id", topology_id, "--method", method,
                "--work-dir", str(work_dir), "--output", str(result_path),
                "--residual-tolerance", str(args.residual_tolerance),
            ]
            with log_path.open("w") as log:
                completed = subprocess.run(command, cwd=project, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if not result_path.exists():
                runs.append({"topology_id": topology_id, "method": method,
                             "status": "FAILED_TO_CONVERGE", "returncode": completed.returncode})
                continue
            result = json.loads(result_path.read_text())
            times = [int(value) for value in TIME_PATTERN.findall(log_path.read_text())]
            result["remaining_simple_iterations"] = max(times) if times else None
            result["returncode"] = completed.returncode
            runs.append(result)

    for topology_id in test_ids:
        topology_runs = [run for run in runs if run["topology_id"] == topology_id]
        cold = next((run for run in topology_runs if run["method"] == "cold"
                     and "final_state" in run), None)
        if cold is None:
            for run in topology_runs:
                run["status"] = "FAILED_TO_CONVERGE"
            continue
        cold_state = np.load(project / cold["final_state"])
        for run in topology_runs:
            if "final_state" not in run or run.get("primal_failed"):
                run["status"] = "FAILED_TO_CONVERGE"
                continue
            final_state = np.load(project / run["final_state"])
            run["final_vs_cold_rel_u"] = velocity_error(final_state, cold_state, n_cells)
            run["final_vs_cold_rel_p_gauge"] = gauge_corrected_pressure_error(
                final_state, cold_state, n_cells
            )
            run["status"] = (
                "CONVERGED_SAME_STATE"
                if run["final_vs_cold_rel_u"] <= 1e-6
                and run["final_vs_cold_rel_p_gauge"] <= 1e-6
                else "DIFFERENT_FINAL_STATE"
            )

    summary = {}
    cold_iterations = {run["topology_id"]: run.get("remaining_simple_iterations")
                       for run in runs if run["method"] == "cold"}
    cold_times = {run["topology_id"]: run.get("total_time_s")
                  for run in runs if run["method"] == "cold"}
    for method in METHODS:
        method_runs = [run for run in runs if run["method"] == method]
        valid = [run for run in method_runs if run.get("status") == "CONVERGED_SAME_STATE"
                 and run.get("remaining_simple_iterations") is not None]
        iteration_savings = [
            1.0 - run["remaining_simple_iterations"] / cold_iterations[run["topology_id"]]
            for run in valid if cold_iterations.get(run["topology_id"])
        ]
        speedups = [
            cold_times[run["topology_id"]] / run["total_time_s"]
            for run in valid if cold_times.get(run["topology_id"])
        ]
        summary[method] = {
            "median_simple_iterations": float(np.median(
                [run["remaining_simple_iterations"] for run in valid])) if valid else None,
            "median_iteration_saving_fraction": float(np.median(iteration_savings))
            if iteration_savings else None,
            "median_total_time_s": float(np.median(
                [run["total_time_s"] for run in valid])) if valid else None,
            "median_paired_speedup": float(np.median(speedups)) if speedups else None,
            "status_counts": {status: sum(run.get("status") == status for run in method_runs)
                              for status in ("CONVERGED_SAME_STATE", "FAILED_TO_CONVERGE",
                                             "DIFFERENT_FINAL_STATE")},
        }
    payload = {
        "criterion": {
            "type": "DAFOAM/OpenFOAM pressure initial residual",
            "p": args.residual_tolerance,
            "note": "U is excluded because the inactive 2D U2 equation has a non-decaying reported residual.",
        },
        "execution_order": "interleaved by topology, then method",
        "summary": summary,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
