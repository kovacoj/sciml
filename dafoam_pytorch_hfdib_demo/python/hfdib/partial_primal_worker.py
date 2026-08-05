#!/usr/bin/env python3
"""Exact-k partial-primal worker: one DAFoam process per k.

Reuses the verified mechanism from run_partial_primal.py:
  copy case -> edit controlDict endTime=k -> run solver -> parse log
  -> verify actual_iterations == k -> save W_k.npy R_k.npy
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import (channel_baseline_options, isothermal_channel_options,  # noqa: E402
                     hfdib_options, write_json, PROJECT_ROOT)
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402


def edit_end_time(path: str, k: int) -> None:
    with open(path) as f:
        s = f.read()
    s = re.sub(r"(?m)^endTime\s+\S+;", f"endTime         {max(k, 1)};", s)
    with open(path, "w") as f:
        f.write(s)


def capture_fd(path: str, fn) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    saved = os.dup(1)
    os.dup2(fd, 1)
    try:
        fn()
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(fd)
        os.close(saved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-src", required=True)
    ap.add_argument("--options", required=True, choices=["channel", "isothermal", "hfdib"])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    opts_factory = {
        "channel": channel_baseline_options,
        "isothermal": isothermal_channel_options,
        "hfdib": hfdib_options,
    }[args.options]

    out_dir = args.out_dir
    work = os.path.join(out_dir, "case")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    shutil.copytree(args.case_src, work, dirs_exist_ok=True)
    for d in os.listdir(work):
        if re.fullmatch(r"[1-9][0-9]*", d):
            shutil.rmtree(os.path.join(work, d), ignore_errors=True)

    edit_end_time(os.path.join(work, "system", "controlDict"), args.k)

    opts = opts_factory(work)
    opts["primalMinResTol"] = 1e-30
    opts["primalMinIters"] = max(args.k + 1, 2)
    opts["printInterval"] = 1

    os.chdir(work)
    log_path = os.path.join(out_dir, "solver.log")
    with open(log_path, "w") as f:
        f.write(f"=== partial-primal k={args.k} ===\n")

    t0 = time.perf_counter()
    bridge = DAFoamResidualBridge(work, opts)
    t_init = time.perf_counter()

    if args.k > 0:
        capture_fd(log_path, bridge.solver)
    t_solve = time.perf_counter()

    # parse actual iterations
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
        "requested_iterations": args.k,
        "actual_iterations": actual,
        "iterations_match": actual == args.k,
        "residual_l2": float(np.linalg.norm(rk)),
        "init_seconds": round(t_init - t0, 3),
        "solve_seconds": round(t_solve - t_init, 3),
        "wall_seconds": round(t_solve - t0, 3),
    }
    write_json(os.path.join(out_dir, "partial_primal.json"), payload)
    print(f"[pp-worker] k={args.k} actual={actual} match={payload['iterations_match']} "
          f"||R||={payload['residual_l2']:.3e} t={payload['wall_seconds']:.2f}s")
    return 0 if payload["iterations_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
