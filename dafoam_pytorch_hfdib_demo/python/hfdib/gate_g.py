#!/usr/bin/env python3
"""Gate G: pure orchestrator. No DAFoamResidualBridge in this process.

Phases:
  1. Validate build report from image
  2. Launch partial_primal_worker for each k (exact-k)
  3. Launch physical_worker: baseline solve + HFDIB solve
  4. Launch physical_worker: base residual at HFDIB state (source isolation)
  5. Load geometry manifest
  6. Run physical_checks from arrays (G1)
  6b. Converged-state derivative check (G2)
  7. Warm-state derivative checks (batch workers)
  8. Select warm k with median+max criteria
  9. Write summary + verdict

Modes:
  --smoke: k=8 only, 1 direction, 3 momentum-seeded block pairs, eps=1e-4
  --full: 5 k values, 10 directions, 9 block pairs, full eps sweep
"""
from __future__ import annotations

import csv
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

N_INTERNAL_FACES = 1224
N_INLET_FACES = 16
U_IN = 0.2
CASE_ISO = os.path.join(PROJECT_ROOT, "cases", "channel_isothermal")
CASE_OBSTACLE = os.path.join(PROJECT_ROOT, "cases", "single_obstacle")
PYTHON = sys.executable
EXPECTED_COMMIT = "31433b41d1d59638b459d24f02c7f89a756848c2"

SMOKE_KS = [8]
FULL_KS = [3, 5, 8, 10, 15]
SMOKE_N_DIRS = 1
FULL_N_DIRS = 10


def _run_module(module, *args, cwd=None):
    cmd = [PYTHON, "-m", module] + list(args)
    print(f"[gate_g] launching: {module} {' '.join(args[:4])}...", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def _evaluate_jtv_rows(raw_rows, k):
    """Select best epsilon per direction from batch-worker output."""
    result = []
    for row in raw_rows:
        best = None
        for eps, fd in zip(row["epsilons"], row["fd_values"]):
            ad = row["ad_value"]
            denom = max(abs(fd), abs(ad), 1e-12)
            rel = abs(fd - ad) / denom
            if best is None or rel < best[1]:
                best = (eps, rel, abs(fd - ad), fd)
        result.append({
            "k": k, "v_block": row["v_block"], "d_block": row["d_block"],
            "direction_id": row["direction_id"],
            "best_eps": best[0], "relative_error": best[1],
            "absolute_error": best[2], "fd_value": best[3],
            "ad_value": ad,
        })
    return result


def _compute_delta_jtv(h_results, b_results):
    """Compute same-epsilon delta-JTV from batch-worker results."""
    if len(h_results) != len(b_results):
        raise RuntimeError(
            f"Derivative row count mismatch: "
            f"HFDIB={len(h_results)}, base={len(b_results)}")
    delta_rows = []
    for row_idx, (h_row, b_row) in enumerate(zip(h_results, b_results)):
        if h_row["direction_hash"] != b_row["direction_hash"]:
            raise RuntimeError(
                f"Direction mismatch at row {row_idx}: "
                f"{h_row['direction_hash']} != {b_row['direction_hash']}")
        if h_row["v_block"] != b_row["v_block"]:
            raise RuntimeError(f"v_block mismatch at row {row_idx}")
        if h_row["d_block"] != b_row["d_block"]:
            raise RuntimeError(f"d_block mismatch at row {row_idx}")
        if h_row["direction_id"] != b_row["direction_id"]:
            raise RuntimeError(f"direction_id mismatch at row {row_idx}")
        if h_row["epsilons"] != b_row["epsilons"]:
            raise RuntimeError(f"epsilon grid mismatch at row {row_idx}")
        if len(h_row["fd_values"]) != len(b_row["fd_values"]):
            raise RuntimeError(f"fd_values length mismatch at row {row_idx}")

        delta_ad = h_row["ad_value"] - b_row["ad_value"]
        best_eps = best_rel = best_abs = best_fd = None
        for eps, fd_h, fd_b in zip(
                h_row["epsilons"], h_row["fd_values"], b_row["fd_values"]):
            delta_fd = fd_h - fd_b
            denom = max(abs(delta_fd), abs(delta_ad), 1e-12)
            rel = abs(delta_fd - delta_ad) / denom
            if best_rel is None or rel < best_rel:
                best_eps = eps
                best_rel = rel
                best_abs = abs(delta_fd - delta_ad)
                best_fd = delta_fd

        delta_rows.append({
            "k": h_row["k"],
            "v_block": h_row["v_block"],
            "d_block": h_row["d_block"],
            "direction_id": h_row["direction_id"],
            "best_eps": best_eps,
            "relative_error": best_rel,
            "absolute_error": best_abs,
            "fd_value": best_fd,
            "ad_value": delta_ad,
        })
    return delta_rows


def _select_warm_k(full_by_k, delta_by_k, ks):
    for k in sorted(ks):
        fr = full_by_k.get(k, [])
        dr = delta_by_k.get(k, [])
        if not fr or not dr:
            continue
        full_errs = np.asarray([r["relative_error"] for r in fr])
        delta_errs = np.asarray([r["relative_error"] for r in dr])
        full_ok = (np.median(full_errs) < 1e-5
                   and np.max(full_errs) < 1e-4)
        delta_ok = (np.median(delta_errs) < 1e-6
                     and np.max(delta_errs) < 1e-5)
        if full_ok and delta_ok:
            return k, float(np.median(full_errs)), float(np.max(full_errs)), \
                   float(np.median(delta_errs)), float(np.max(delta_errs))
    return None, None, None, None, None


def _write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    smoke = args.smoke or not args.full
    ks = SMOKE_KS if smoke else FULL_KS
    n_dirs = SMOKE_N_DIRS if smoke else FULL_N_DIRS
    mode_label = "smoke" if smoke else "full"
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "hfdib_gate_g", mode_label)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    layout = build_state_layout("isothermal")
    failures = []

    # ---- 1. Build report --------------------------------------------------
    build_report_path = os.path.expanduser("~/hfdib_build_report.json")
    if not os.path.isfile(build_report_path):
        print(f"GATE G FAIL: build report missing in image: {build_report_path}")
        return 1
    shutil.copy(build_report_path, os.path.join(out_dir, "build_report.json"))
    with open(os.path.join(out_dir, "build_report.json")) as f:
        build_report = json.load(f)

    if build_report.get("dafoam_source_commit") != EXPECTED_COMMIT:
        failures.append(f"source commit mismatch")
    if build_report.get("normal_lib_hash", "MISSING") == "MISSING":
        failures.append("normal lib hash missing")
    if build_report.get("adr_lib_hash", "MISSING") == "MISSING":
        failures.append("ADR lib hash missing")
    if not build_report.get("hfdib_in_normal"):
        failures.append("hfdib not in normal library")
    if not build_report.get("hfdib_in_adr"):
        failures.append("hfdib not in ADR library")

    # ---- 2. Partial-primal captures (exact-k) -----------------------------
    pp_dir = os.path.join(out_dir, "partial_primal")
    for k in ks:
        k_dir = os.path.join(pp_dir, f"k{k}")
        _run_module("hfdib.partial_primal_worker",
                    "--case-src", CASE_OBSTACLE,
                    "--options", "hfdib", "--k", str(k),
                    "--out-dir", k_dir, cwd=str(PYTHON_ROOT))
        meta = json.load(open(os.path.join(k_dir, "partial_primal.json")))
        if not meta["iterations_match"]:
            failures.append(f"partial-primal k={k}: iterations mismatch "
                            f"(actual={meta['actual_iterations']})")

    # ---- 3. Baseline + HFDIB primal solves --------------------------------
    work_base = os.path.join(out_dir, "solve_base", "case")
    work_hfdib = os.path.join(out_dir, "solve_hfdib", "case")
    shutil.copytree(CASE_ISO, work_base, dirs_exist_ok=True)
    shutil.copytree(CASE_OBSTACLE, work_hfdib, dirs_exist_ok=True)

    solve_base_dir = os.path.join(out_dir, "solve_base")
    solve_hfdib_dir = os.path.join(out_dir, "solve_hfdib")

    _run_module("hfdib.physical_worker", "--mode", "solve",
                "--case", work_base, "--options", "isothermal",
                "--out", solve_base_dir, cwd=str(PYTHON_ROOT))
    _run_module("hfdib.physical_worker", "--mode", "solve",
                "--case", work_hfdib, "--options", "hfdib",
                "--out", solve_hfdib_dir, cwd=str(PYTHON_ROOT))

    w_base = np.load(os.path.join(solve_base_dir, "W.npy"))
    r_base = np.load(os.path.join(solve_base_dir, "R.npy"))
    w_hfdib = np.load(os.path.join(solve_hfdib_dir, "W.npy"))
    r_hfdib = np.load(os.path.join(solve_hfdib_dir, "R.npy"))

    # ---- 4. Source isolation: R_0(W_H) at SAME state ---------------------
    r0_path = os.path.join(solve_hfdib_dir, "R_base_at_whfdib.npy")
    _run_module("hfdib.physical_worker", "--mode", "residual",
                "--case", work_base, "--options", "isothermal",
                "--state", os.path.join(solve_hfdib_dir, "W.npy"),
                "--out", r0_path, cwd=str(PYTHON_ROOT))
    r_base_at_whfdib = np.load(r0_path)
    r_hfdib_at_whfdib = r_hfdib

    # ---- 5. Geometry manifest ---------------------------------------------
    manifest_src = os.path.join(work_hfdib, "postProcessing", "hfdibGeometry")
    if os.path.isdir(manifest_src):
        shutil.copytree(manifest_src,
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

    # ---- 6. Physical checks (G1) ------------------------------------------
    phys = {}
    g1_pass = False
    if manifest:
        phys = physical_checks_all(
            w_hfdib, w_base, r_hfdib_at_whfdib, r_base_at_whfdib,
            manifest, layout, U_IN, N_INTERNAL_FACES, N_INLET_FACES)
        write_json(os.path.join(out_dir, "physical_report.json"), phys)
        g1_pass = all(
            check.get("pass", True) for check in phys.values()
            if isinstance(check, dict))
        for name, check in phys.items():
            if isinstance(check, dict) and not check.get("pass", True):
                failures.append(f"{name}: {check}")
            print(f"[gate_g] {name}: {check.get('pass', '?')}")

    # ---- 6b. Converged-state derivative check (G2) ------------------------
    print("[gate_g] running converged-state derivative check (G2)...", flush=True)
    conv_w_path = os.path.join(solve_hfdib_dir, "W.npy")

    conv_out_h = os.path.join(out_dir, "jtv_conv_hfdib.json")
    conv_out_0 = os.path.join(out_dir, "jtv_conv_base.json")
    conv_args_h = ["--case", work_hfdib, "--options", "hfdib",
                   "--state", conv_w_path, "--k", str(-1),
                   "--n-dirs", str(n_dirs), "--out", conv_out_h]
    if smoke:
        conv_args_h.append("--smoke")
    _run_module("hfdib.batch_derivative_worker", *conv_args_h, cwd=str(PYTHON_ROOT))
    with open(conv_out_h) as f:
        conv_h = json.load(f)

    conv_args_0 = ["--case", work_base, "--options", "isothermal",
                   "--state", conv_w_path, "--k", str(-1),
                   "--n-dirs", str(n_dirs), "--out", conv_out_0]
    if smoke:
        conv_args_0.append("--smoke")
    _run_module("hfdib.batch_derivative_worker", *conv_args_0, cwd=str(PYTHON_ROOT))
    with open(conv_out_0) as f:
        conv_b = json.load(f)

    conv_full_rows = _evaluate_jtv_rows(conv_h, -1)
    conv_delta_rows = _compute_delta_jtv(conv_h, conv_b)

    conv_full_errs = np.asarray([r["relative_error"] for r in conv_full_rows]) if conv_full_rows else np.array([1.0])
    conv_delta_errs = np.asarray([r["relative_error"] for r in conv_delta_rows]) if conv_delta_rows else np.array([1.0])
    conv_full_med = float(np.median(conv_full_errs))
    conv_full_max = float(np.max(conv_full_errs))
    conv_delta_med = float(np.median(conv_delta_errs))
    conv_delta_max = float(np.max(conv_delta_errs))

    _write_csv(conv_full_rows, os.path.join(out_dir, "jtv_conv_full.csv"))
    _write_csv(conv_delta_rows, os.path.join(out_dir, "jtv_conv_delta.csv"))

    print(f"[gate_g] converged full:  median={conv_full_med:.2e} max={conv_full_max:.2e}")
    print(f"[gate_g] converged delta: median={conv_delta_med:.2e} max={conv_delta_max:.2e}")

    g2_pass = (conv_full_med < 1e-5 and conv_full_max < 1e-4
               and conv_delta_med < 1e-6 and conv_delta_max < 1e-5)
    g2_delta_only = (conv_delta_med < 1e-6 and conv_delta_max < 1e-5)
    if g2_pass:
        g2_status = "pass"
    elif g2_delta_only:
        g2_status = "partial"
        print("[gate_g] G2: delta-JTV PASS, full-JTV FAIL (baseline Jacobian)")
    else:
        g2_status = "fail"
        failures.append(f"G2 delta-JTV FAIL: median={conv_delta_med:.2e} max={conv_delta_max:.2e}")

    # ---- 7. Warm-state derivative checks (batch workers) ------------------
    full_by_k = {}
    delta_by_k = {}

    for k in ks:
        k_dir = os.path.join(pp_dir, f"k{k}")
        w_k_path = os.path.join(k_dir, "W_k.npy")

        out_h = os.path.join(out_dir, f"jtv_full_hfdib_k{k}.json")
        batch_args = ["--case", work_hfdib, "--options", "hfdib",
                      "--state", w_k_path, "--k", str(k),
                      "--n-dirs", str(n_dirs), "--out", out_h]
        if smoke:
            batch_args.append("--smoke")
        _run_module("hfdib.batch_derivative_worker", *batch_args, cwd=str(PYTHON_ROOT))
        with open(out_h) as f:
            h_results = json.load(f)

        out_0 = os.path.join(out_dir, f"jtv_full_base_k{k}.json")
        batch_args_0 = ["--case", work_base, "--options", "isothermal",
                       "--state", w_k_path, "--k", str(k),
                       "--n-dirs", str(n_dirs), "--out", out_0]
        if smoke:
            batch_args_0.append("--smoke")
        _run_module("hfdib.batch_derivative_worker", *batch_args_0, cwd=str(PYTHON_ROOT))
        with open(out_0) as f:
            b_results = json.load(f)

        full_by_k[k] = _evaluate_jtv_rows(h_results, k)
        delta_by_k[k] = _compute_delta_jtv(h_results, b_results)

    all_full = [r for k in sorted(full_by_k) for r in full_by_k[k]]
    all_delta = [r for k in sorted(delta_by_k) for r in delta_by_k[k]]
    _write_csv(all_full, os.path.join(out_dir, "jtv_full.csv"))
    _write_csv(all_delta, os.path.join(out_dir, "jtv_delta.csv"))

    # ---- 8. Select warm k with median+max ---------------------------------
    k_star, full_med, full_max, delta_med, delta_max = _select_warm_k(
        full_by_k, delta_by_k, ks)

    if k_star is None:
        failures.append("no warm state passes JTV criteria "
                         f"(full median<1e-5 + max<1e-4, "
                         f"delta median<1e-6 + max<1e-5)")

    # ---- 9. Summary + verdict --------------------------------------------
    summary = {
        "mode": mode_label,
        "G1_physical_hfdib": "pass" if g1_pass else "fail",
        "G2_converged_derivative": g2_status,
        "G3_trainable_parameter_gradient": "pending",
        "normal_build": "pass" if build_report.get("hfdib_in_normal") else "fail",
        "adr_build": "pass" if build_report.get("hfdib_in_adr") else "fail",
        "source_registered": bool(build_report.get("hfdib_in_normal")
                                 and build_report.get("hfdib_in_adr")),
        "fluid_cells": manifest.n_fluid if manifest else 0,
        "solid_cells": manifest.n_solid if manifest else 0,
        "interface_cells": manifest.n_interface if manifest else 0,
        "source_l2": phys.get("source_activation", {}).get("source_l2", 0),
        "max_source_outside_chi": phys.get("source_activation", {}).get(
            "max_source_outside_chi", 0),
        "max_solid_speed": phys.get("solid_noslip", {}).get("max_solid_speed", -1),
        "interface_velocity_error": phys.get("interface_velocity", {}).get("error", 0),
        "relative_solution_difference": phys.get("flow_deflection", {}).get(
            "relative_difference", 0),
        "mass_imbalance": phys.get("mass_imbalance", {}).get("mass_imbalance", 1),
        "residual_l2": phys.get("residuals", {}).get("residual_l2", 1e10),
        "selected_warm_k": k_star if k_star else -1,
        "full_jtv_median": full_med if full_med is not None else 1.0,
        "full_jtv_max": full_max if full_max is not None else 1.0,
        "delta_jtv_median": delta_med if delta_med is not None else 1.0,
        "delta_jtv_max": delta_max if delta_max is not None else 1.0,
        "conv_full_jtv_median": conv_full_med,
        "conv_full_jtv_max": conv_full_max,
        "conv_delta_jtv_median": conv_delta_med,
        "conv_delta_jtv_max": conv_delta_max,
        "gate_g": "fail" if failures else "pass",
        "failures": failures,
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)

    if failures:
        print("GATE G FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    elif smoke:
        print("GATE G SMOKE COMPLETE:")
        print(f"  G1 physical:      {summary['G1_physical_hfdib']}")
        print(f"  G2 converged der: {summary['G2_converged_derivative']}")
        print(f"  G3 param gradient: {summary['G3_trainable_parameter_gradient']}")
        print("Run --full for certification.")
        return 0
    else:
        print("GATE G PASS:")
        print("Static HFDIB is physically active, convergent, "
              "and included in the reverse-AD tape.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
