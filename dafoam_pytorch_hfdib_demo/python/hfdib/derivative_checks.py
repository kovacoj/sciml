"""Derivative checks for Gate G using isolated subprocess workers.

Each base/HFDIB evaluation runs in its own process to avoid PYDAFOAM
registry conflicts. Results are exchanged via npz files.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
N_DIRS = 10


def _scaled_dir(rng, n, ids):
    d = np.zeros(n)
    d[ids] = rng.standard_normal(ids.size)
    norm = np.linalg.norm(d)
    return d / norm if norm else d


def _run_worker(case_dir, options, state, seed, out_path):
    """Run derivative_worker as a subprocess."""
    cmd = [
        sys.executable, "-m", "hfdib.derivative_worker",
        "--case", case_dir,
        "--options", options,
        "--state", state,
        "--seed", seed,
        "--out", out_path,
    ]
    subprocess.run(cmd, check=True, cwd=str(PYTHON_ROOT.parent))
    return np.load(out_path)


def cross_block_jtv_worker(case_dir, options, w, layout,
                            out_dir, n_dirs=N_DIRS):
    """Cross-block JTV via a single bridge (one process per case)."""
    n = w.size
    blocks = {"U": layout.indices("U"),
              "p": layout.indices("p"),
              "phi": layout.indices("phi")}
    rng = np.random.default_rng(42)
    results = []

    # save state once
    state_path = os.path.join(out_dir, "W_check.npy")
    np.save(state_path, w)

    for v_name, v_ids in blocks.items():
        for d_name, d_ids in blocks.items():
            for dir_i in range(n_dirs):
                d = _scaled_dir(rng, n, d_ids)
                v = _scaled_dir(rng, n, v_ids)
                seed_path = os.path.join(out_dir, f"seed_{v_name}_{d_name}_{dir_i}.npy")
                np.save(seed_path, v)

                out_path = os.path.join(out_dir, f"jtv_{options}_{v_name}_{d_name}_{dir_i}.npz")
                result = _run_worker(case_dir, options, state_path, seed_path, out_path)
                jtv = result["jtv"]
                ad = float(d @ jtv)

                fd_vals = []
                for eps in EPS_GRID:
                    # FD: perturb state, eval residual in a fresh worker
                    w_p = w + eps * d
                    w_m = w - eps * d
                    np.save(state_path, w_p)
                    r_p = _run_worker(case_dir, options, state_path, seed_path,
                                       out_path + ".p")
                    np.save(state_path, w_m)
                    r_m = _run_worker(case_dir, options, state_path, seed_path,
                                       out_path + ".m")
                    fd = float(v @ (r_p["residual"] - r_m["residual"]) / (2 * eps))
                    fd_vals.append((eps, fd))

                # restore state
                np.save(state_path, w)

                best = None
                for eps, fd in fd_vals:
                    denom = max(abs(fd), abs(ad), 1e-12)
                    rel = abs(fd - ad) / denom
                    if best is None or rel < best[1]:
                        best = (eps, rel, abs(fd - ad), fd, ad)

                results.append({
                    "v_block": v_name, "d_block": d_name,
                    "direction_id": dir_i,
                    "best_eps": best[0], "relative_error": best[1],
                    "absolute_error": best[2], "fd_value": best[3],
                    "ad_value": best[4],
                })
    return results


def delta_jtv_worker(case_hfdib, case_base, w, layout,
                      out_dir, n_dirs=N_DIRS):
    """HFDIB-isolated delta JTV using two subprocess workers."""
    n = w.size
    blocks = {"U": layout.indices("U"),
              "p": layout.indices("p"),
              "phi": layout.indices("phi")}
    rng = np.random.default_rng(99)
    results = []

    state_path = os.path.join(out_dir, "W_delta.npy")
    np.save(state_path, w)

    for v_name, v_ids in blocks.items():
        for d_name, d_ids in blocks.items():
            for dir_i in range(n_dirs):
                d = _scaled_dir(rng, n, d_ids)
                v = _scaled_dir(rng, n, v_ids)
                seed_path = os.path.join(out_dir, f"dseed_{v_name}_{d_name}_{dir_i}.npy")
                np.save(seed_path, v)

                # JTV from hfdib and base workers
                out_h = os.path.join(out_dir, f"djtv_h_{v_name}_{d_name}_{dir_i}.npz")
                out_0 = os.path.join(out_dir, f"djtv_0_{v_name}_{d_name}_{dir_i}.npz")
                rh = _run_worker(case_hfdib, "hfdib", state_path, seed_path, out_h)
                r0 = _run_worker(case_base, "isothermal", state_path, seed_path, out_0)
                delta_ad = float(d @ (rh["jtv"] - r0["jtv"]))

                # FD of delta R
                fd_vals = []
                for eps in EPS_GRID:
                    np.save(state_path, w + eps * d)
                    rp_h = _run_worker(case_hfdib, "hfdib", state_path, seed_path,
                                        out_h + ".p")
                    rp_0 = _run_worker(case_base, "isothermal", state_path, seed_path,
                                        out_0 + ".p")
                    np.save(state_path, w - eps * d)
                    rm_h = _run_worker(case_hfdib, "hfdib", state_path, seed_path,
                                        out_h + ".m")
                    rm_0 = _run_worker(case_base, "isothermal", state_path, seed_path,
                                        out_0 + ".m")
                    fd = float(v @ ((rp_h["residual"] - rp_0["residual"])
                                    - (rm_h["residual"] - rm_0["residual"]))
                               / (2 * eps))
                    fd_vals.append((eps, fd))

                np.save(state_path, w)

                best = None
                for eps, fd in fd_vals:
                    denom = max(abs(fd), abs(delta_ad), 1e-12)
                    rel = abs(fd - delta_ad) / denom
                    if best is None or rel < best[1]:
                        best = (eps, rel, abs(fd - delta_ad), fd, delta_ad)

                results.append({
                    "v_block": v_name, "d_block": d_name,
                    "direction_id": dir_i,
                    "best_eps": best[0], "relative_error": best[1],
                    "absolute_error": best[2], "fd_value": best[3],
                    "ad_value": best[4],
                })
    return results


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def select_warm_k(full_by_k, delta_by_k, ks):
    for k in sorted(ks):
        fr = full_by_k.get(k, [])
        dr = delta_by_k.get(k, [])
        if not fr or not dr:
            continue
        full_ok = all(r["relative_error"] < 1e-4 for r in fr)
        delta_ok = all(r["relative_error"] < 1e-5 for r in dr)
        if full_ok and delta_ok:
            return k
    return None
