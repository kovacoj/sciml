#!/usr/bin/env python3
"""Reproducible partial-primal runs: exactly k SIMPLE iterations per copy.

Never modifies the canonical case dir.  For each k, copies the case into
outputs/work/partial-k{K}/case, caps the pseudo-time loop at k iterations
(controlDict endTime=k, primalMinResTol=1e-30, primalMinIters=k+1), runs the
primal once, and records W_k, R_k, timing, and the ACTUAL iteration count
parsed from the C++ solver log (fd-1 level redirection captures C++ cout).
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from common import channel_baseline_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402


def edit_control_dict(path: str, k: int) -> None:
    with open(path) as f:
        s = f.read()
    s = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {max(k, 1)};", s)
    with open(path, "w") as f:
        f.write(s)


def capture_fd(path: str, fn) -> None:
    """Run fn() with OS-level stdout (fd 1) redirected to `path` (append),
    so C++/OpenFOAM stdout lands in the solver log too."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    saved = os.dup(1)
    os.dup2(fd, 1)
    try:
        fn()
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.dup2(saved, 1)
        os.close(fd)
        os.close(saved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="*", default=[0, 1, 2, 3, 5, 8, 10, 15, 20])
    args = parser.parse_args()

    from common import channel_baseline_options  # noqa: E402
    from dafoam_bridge import DAFoamResidualBridge  # noqa: E402

    case_dir = os.path.join(PROJECT_ROOT, "cases", "channel_baseline")

    for k in args.ks:
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "work", f"partial-k{k}")
        work = os.path.join(out_dir, "case")
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(work, exist_ok=True)
        shutil.copytree(case_dir, work, dirs_exist_ok=True)
        # remove generated time dirs like "112" (but NEVER the "0" fields dir)
        for d in os.listdir(work):
            if re.fullmatch(r"[1-9][0-9]*", d):
                shutil.rmtree(os.path.join(work, d), ignore_errors=True)

        edit_control_dict(os.path.join(work, "system", "controlDict"), k)

        opts = channel_baseline_options(work)
        opts["primalMinResTol"] = 1e-30
        opts["primalMinIters"] = max(k + 1, 2)
        opts["printInterval"] = 1

        os.chdir(work)
        log_path = os.path.join(out_dir, "solver.log")
        with open(log_path, "w") as f:
            f.write(f"=== partial-primal k={k} ===\n")

        t0 = time.perf_counter()
        bridge = DAFoamResidualBridge(work, opts)
        t_init = time.perf_counter()

        if k > 0:
            capture_fd(log_path, bridge.solver)
        t_solve = time.perf_counter()

        # Actual iteration count parsed from the C++ solver log.
        try:
            with open(log_path) as f:
                times = re.findall(r"^Time = (\d+)", f.read(), re.M)
            actual = int(times[-1]) if times else 0
        except OSError:
            actual = -1

        wk = np.ascontiguousarray(
            bridge.solver.getStates().copy(), dtype=np.float64)
        rk = bridge.residual(wk)
        np.save(os.path.join(out_dir, "W_k.npy"), wk)
        np.save(os.path.join(out_dir, "R_k.npy"), rk)

        payload = {
            "requested_iterations": k,
            "actual_iterations": actual,
            "iterations_match": (actual == k),
            "residual_l2": float(np.linalg.norm(rk)),
            "residual_linf": float(np.max(np.abs(rk))),
            "init_seconds": round(t_init - t0, 3),
            "solve_seconds": round(t_solve - t_init, 3),
            "wall_seconds": round(t_solve - t0, 3),
            "full_primal_iteration_fraction": None,
        }
        write_json(os.path.join(out_dir, "partial_primal.json"), payload)

        os.chdir(PROJECT_ROOT)
        print(f"[partial] k={k:2d} actual={actual:2d} "
              f"match={payload['iterations_match']} "
              f"||R||_2={payload['residual_l2']:.3e} "
              f"t={payload['wall_seconds']:.2f}s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
