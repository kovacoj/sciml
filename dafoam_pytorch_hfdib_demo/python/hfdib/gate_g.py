#!/usr/bin/env python3
"""Gate G: pure orchestrator. No DAFoamResidualBridge in this process.

Phases:
  1. Copy build report from image
  2. Launch partial_primal_worker for each k (exact-k)
  3. Launch physical_worker: baseline solve + HFDIB solve
  4. Launch physical_worker: base residual at HFDIB state (source isolation)
  5. Load geometry manifest from HFDIB solve output
  6. Run physical_checks from arrays
  7. Launch batch_derivative_worker per (case, k)
  8. Read results, compute full + delta JTV comparisons
  9. Select warm k, write summary + verdict
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import write_json, PROJECT_ROOT  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from hfdib.geometry_manifest import load as load_manifest, validate as validate_manifest  # noqa: E402
from hfdib.physical_checks import run_all as physical_checks_all  # noqa: E402

KS_FOR_JTV = [3, 5, 8, 10, 15]
N_INTERNAL_FACES = 1224
N_INLET_FACES = 16
U_IN = 0.2
CASE_ISO = os.path.join(PROJECT_ROOT, "cases", "channel_isothermal")
CASE_OBSTACLE = os.path.join(PROJECT_ROOT, "cases", "single_obstacle")
PYTHON = sys.executable


def _run(cmd, cwd=None):
    """Run a subprocess, raise on failure."""
    print(f"[gate_g] launching: {' '.join(cmd[:4])}...", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def _python_module(module, *args, cwd=None):
    """Launch a python -m module with args."""
    cmd = [PYTHON, "-m", module] + list(args)
    _run(cmd, cwd=cwd)


def main() -> int:
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "hfdib_gate_g")
    os.makedirs(out_dir, exist_ok=True)
    layout = build_state_layout("isothermal")
    failures = []

    # ---- 1. Build report -------------------------------------------------
    build_report_path = os.path.expanduser(
        "~/hfdib_build_report.json")
    if os.path.exists(build_report_path):
        shutil.copy(build_report_path,
                    os.path.join(out_dir, "build_report.json"))
    with open(os.path.join(out_dir, "build_report.json")) as f:
        build_report = json.load(f)
    if not build_report.get("hfdib_in_normal"):
        failures.append("hfdib not in normal library")
    if not build_report.get("hfdib_in_adr"):
        failures.append("hfdib not in ADR library")

    # ---- 2. Partial-primal captures (exact-k) -----------------------------
    pp_dir = os.path.join(out_dir, "partial_primal")
    for k in KS_FOR_JTV:
        k_dir = os.path.join(pp_dir, f"k{k}")
        os.makedirs(k_dir, exist_ok=True)
        _python_module("hfdib.partial_primal_worker",
                       "--case-src", CASE_OBSTACLE,
                       "--options", "hfdib",
                       "--k", str(k),
                       "--out-dir", k_dir,
                       cwd=str(PYTHON_ROOT))
        meta = json.load(open(os.path.join(k_dir, "partial_primal.json")))
        if not meta["iterations_match"]:
            failures.append(f"partial-primal k={k}: iterations mismatch "
                            f"(actual={meta['actual_iterations']})")

    # ---- 3. Baseline + HFDIB primal solves --------------------------------
    solve_base_dir = os.path.join(out_dir, "solve_base")
    solve_hfdib_dir = os.path.join(out_dir, "solve_hfdib")
    os.makedirs(solve_base_dir, exist_ok=True)
    os.makedirs(solve_hfdib_dir, exist_ok=True)

    # copy cases for solving
    work_base = os.path.join(solve_base_dir, "case")
    work_hfdib = os.path.join(solve_hfdib_dir, "case")
    shutil.rmtree(work_base, ignore_errors=True)
    shutil.rmtree(work_hfdib, ignore_errors=True)
    shutil.copytree(CASE_ISO, work_base, dirs_exist_ok=True)
    shutil.copytree(CASE_OBSTACLE, work_hfdib, dirs_exist_ok=True)

    _python_module("hfdib.physical_worker",
                   "--mode", "solve", "--case", work_base,
                   "--options", "isothermal", "--out", solve_base_dir,
                   cwd=str(PYTHON_ROOT))
    _python_module("hfdib.physical_worker",
                   "--mode", "solve", "--case", work_hfdib,
                   "--options", "hfdib", "--out", solve_hfdib_dir,
                   cwd=str(PYTHON_ROOT))

    w_base = np.load(os.path.join(solve_base_dir, "W.npy"))
    r_base = np.load(os.path.join(solve_base_dir, "R.npy"))
    w_hfdib = np.load(os.path.join(solve_hfdib_dir, "W.npy"))
    r_hfdib = np.load(os.path.join(solve_hfdib_dir, "R.npy"))

    # ---- 4. Source isolation: R_0(W_H) at the SAME state ------------------
    r0_at_wh_path = os.path.join(solve_hfdib_dir, "R_base_at_whfdib.npy")
    _python_module("hfdib.physical_worker",
                   "--mode", "residual", "--case", work_base,
                   "--options", "isothermal",
                   "--state", os.path.join(solve_hfdib_dir, "W.npy"),
                   "--out", r0_at_wh_path,
                   cwd=str(PYTHON_ROOT))
    r_base_at_whfdib = np.load(r0_at_wh_path)

    # also need R_H(W_H) — already have r_hfdib
    r_hfdib_at_whfdib = r_hfdib

    # ---- 5. Geometry manifest ---------------------------------------------
    manifest_path = os.path.join(work_hfdib, "postProcessing", "hfdibGeometry")
    if os.path.isdir(manifest_path):
        shutil.copytree(manifest_path,
                        os.path.join(out_dir, "hfdibGeometry"),
                        dirs_exist_ok=True)
    try:
        manifest = load_manifest(work_hfdib)
    except FileNotFoundError as e:
        failures.append(f"manifest not found: {e}")
        manifest = None

    if manifest:
        geo_errors = validate_manifest(manifest)
        failures.extend([f"geometry: {e}" for e in geo_errors])
        print(f"[gate_g] geometry: {manifest.n_fluid} fluid, "
              f"{manifest.n_solid} solid, {manifest.n_interface} interface")

    # ---- 6. Physical checks -----------------------------------------------
    if manifest:
        phys = physical_checks_all(
            w_hfdib, w_base, r_hfdib_at_whfdib, r_base_at_whfdib,
            manifest, layout, U_IN, N_INTERNAL_FACES, N_INLET_FACES)
        write_json(os.path.join(out_dir, "physical_report.json"), phys)
        for name, check in phys.items():
            if isinstance(check, dict) and not check.get("pass", True):
                failures.append(f"{name}: {check}")
            print(f"[gate_g] {name}: {check.get('pass', '?')}")

    # ---- 7. Derivative checks (batch workers) -----------------------------
    full_by_k = {}
    delta_by_k = {}

    for k in KS_FOR_JTV:
        k_dir = os.path.join(pp_dir, f"k{k}")
        w_k = np.load(os.path.join(k_dir, "W_k.npy"))

        # HFDIB batch
        out_h = os.path.join(out_dir, f"jtv_full_hfdib_k{k}.json")
        _python_module("hfdib.batch_derivative_worker",
                       "--case", work_hfdib,
                       "--options", "hfdib",
                       "--state", os.path.join(k_dir, "W_k.npy"),
                       "--k", str(k),
                       "--out", out_h,
                       cwd=str(PYTHON_ROOT))
        with open(out_h) as f:
            full_by_k[k] = json.load(f)

        # baseline batch
        out_0 = os.path.join(out_dir, f"jtv_full_base_k{k}.json")
        _python_module("hfdib.batch_derivative_worker",
                       "--case", work_base,
                       "--options", "isothermal",
                       "--state", os.path.join(k_dir, "W_k.npy"),
                       "--k", str(k),
                       "--out", out_0,
                       cwd=str(PYTHON_ROOT))
        with open(out_0) as f:
            base_results = json.load(f)

        # compute delta JTV in the parent process
        delta_rows = []
        for h_row, b_row in zip(full_by_k[k], base_results):
            delta_ad = h_row["ad_value"] - b_row["ad_value"]
            # FD of delta: need (R_H - R_0)(W+eps*d) - (R_H - R_0)(W-eps*d)
            # We don't have the per-eps FD here (batch worker only saved best).
            # For now, approximate: use the best-eps fd from both
            delta_fd = h_row["fd_value"] - b_row["fd_value"]
            denom = max(abs(delta_fd), abs(delta_ad), 1e-12)
            delta_rel = abs(delta_fd - delta_ad) / denom
            delta_rows.append({
                "k": k,
                "v_block": h_row["v_block"],
                "d_block": h_row["d_block"],
                "direction_id": h_row["direction_id"],
                "best_eps": h_row["best_eps"],
                "relative_error": delta_rel,
                "absolute_error": abs(delta_fd - delta_ad),
                "fd_value": delta_fd,
                "ad_value": delta_ad,
            })
        delta_by_k[k] = delta_rows

    # write CSVs with k column
    import csv
    all_full = [r for k in sorted(full_by_k) for r in full_by_k[k]]
    all_delta = [r for k in sorted(delta_by_k) for r in delta_by_k[k]]
    for rows, name in [(all_full, "jtv_full.csv"), (all_delta, "jtv_delta.csv")]:
        if rows:
            with open(os.path.join(out_dir, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    # ---- 8. Select warm k -------------------------------------------------
    k_star = None
    for k in KS_FOR_JTV:
        fr = full_by_k.get(k, [])
        dr = delta_by_k.get(k, [])
        if not fr or not dr:
            continue
        full_ok = all(r["relative_error"] < 1e-4 for r in fr)
        delta_ok = all(r["relative_error"] < 1e-5 for r in dr)
        if full_ok and delta_ok:
            k_star = k
            break

    if k_star is None:
        failures.append("no warm state passes JTV criteria")

    check_k = k_star if k_star else KS_FOR_JTV[-1]
    full_max = max(r["relative_error"] for r in full_by_k.get(check_k, [{"relative_error": 1}]))
    delta_max = max(r["relative_error"] for r in delta_by_k.get(check_k, [{"relative_error": 1}]))

    # ---- 9. Summary + verdict --------------------------------------------
    summary = {
        "normal_build": "pass" if build_report.get("hfdib_in_normal") else "fail",
        "adr_build": "pass" if build_report.get("hfdib_in_adr") else "fail",
        "source_registered": build_report.get("hfdib_in_normal", False)
                             and build_report.get("hfdib_in_adr", False),
        "fluid_cells": manifest.n_fluid if manifest else 0,
        "solid_cells": manifest.n_solid if manifest else 0,
        "interface_cells": manifest.n_interface if manifest else 0,
        "source_l2": phys.get("source_activation", {}).get("source_l2", 0.0) if manifest else 0,
        "max_solid_speed": phys.get("solid_noslip", {}).get("max_solid_speed", -1) if manifest else -1,
        "interface_velocity_error": phys.get("interface_velocity", {}).get("error", 0) if manifest else 0,
        "relative_solution_difference": phys.get("flow_deflection", {}).get("relative_difference", 0) if manifest else 0,
        "mass_imbalance": phys.get("mass_imbalance", {}).get("mass_imbalance", 1) if manifest else 1,
        "residual_l2": phys.get("residuals", {}).get("residual_l2", 1e10) if manifest else 1e10,
        "selected_warm_k": k_star if k_star else -1,
        "full_jtv_max_error": full_max,
        "delta_jtv_max_error": delta_max,
        "gate_g": "fail" if failures else "pass",
        "failures": failures,
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)

    if failures:
        print("GATE G FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("GATE G PASS:")
        print("Static HFDIB is physically active, convergent, "
              "and included in the reverse-AD tape.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
