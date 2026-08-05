"""Derivative checks for Gate G: cross-block JTV + HFDIB-isolated delta-JTV.

All use denom = max(abs(fd), abs(ad), 1e-12) — never max(1.0, ...).
"""
from __future__ import annotations

import csv
import numpy as np

EPS_GRID = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
N_DIRS = 10


def _best_err(fd_vals, ad_val):
    best = None
    for eps, fd in fd_vals:
        denom = max(abs(fd), abs(ad_val), 1e-12)
        rel = abs(fd - ad_val) / denom
        if best is None or rel < best[1]:
            best = (eps, rel, abs(fd - ad_val), fd, ad_val)
    return best  # (best_eps, rel_err, abs_err, fd_val, ad_val)


def _scaled_dir(rng, n, ids):
    d = np.zeros(n)
    d[ids] = rng.standard_normal(ids.size)
    norm = np.linalg.norm(d)
    return d / norm if norm else d


def cross_block_jtv(bridge, w, layout, n_dirs=N_DIRS):
    """Test v_block^T J d_block for all block pairs."""
    n = w.size
    blocks = {"U": layout.indices("U"),
              "p": layout.indices("p"),
              "phi": layout.indices("phi")}
    rng = np.random.default_rng(42)
    results = []

    for v_name, v_ids in blocks.items():
        for d_name, d_ids in blocks.items():
            rels = []
            for _ in range(n_dirs):
                d = _scaled_dir(rng, n, d_ids)
                v = _scaled_dir(rng, n, v_ids)
                jtv = bridge.residual_jacobian_transpose_vector(w, v)
                ad = float(d @ jtv)
                fd_vals = []
                for eps in EPS_GRID:
                    fd = float(v @ (bridge.residual(w + eps * d)
                                    - bridge.residual(w - eps * d))
                               / (2 * eps))
                    fd_vals.append((eps, fd))
                best = _best_err(fd_vals, ad)
                rels.append(best[1])
            results.append({
                "v_block": v_name, "d_block": d_name,
                "median_rel": float(np.median(rels)),
                "max_rel": float(max(rels)),
                "n_dirs": n_dirs,
            })
    return results


def delta_jtv(bridge_hfdib, bridge_base, w, layout, n_dirs=N_DIRS):
    """HFDIB-isolated: d^T (J_H^T v - J_0^T v) vs FD of (R_H - R_0)."""
    n = w.size
    blocks = {"U": layout.indices("U"),
              "p": layout.indices("p"),
              "phi": layout.indices("phi")}
    rng = np.random.default_rng(99)
    results = []

    def delta_R(x):
        return bridge_hfdib.residual(x) - bridge_base.residual(x)

    for v_name, v_ids in blocks.items():
        for d_name, d_ids in blocks.items():
            rels = []
            for _ in range(n_dirs):
                d = _scaled_dir(rng, n, d_ids)
                v = _scaled_dir(rng, n, v_ids)
                jtv_h = bridge_hfdib.residual_jacobian_transpose_vector(w, v)
                jtv_0 = bridge_base.residual_jacobian_transpose_vector(w, v)
                delta_ad = float(d @ (jtv_h - jtv_0))
                fd_vals = []
                for eps in EPS_GRID:
                    fd = float(v @ (delta_R(w + eps * d)
                                    - delta_R(w - eps * d))
                               / (2 * eps))
                    fd_vals.append((eps, fd))
                best = _best_err(fd_vals, delta_ad)
                rels.append(best[1])
            results.append({
                "v_block": v_name, "d_block": d_name,
                "median_rel": float(np.median(rels)),
                "max_rel": float(max(rels)),
                "n_dirs": n_dirs,
            })
    return results


def write_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def select_warm_k(full_results, delta_results, ks):
    """Return smallest k where full < 1e-5/1e-4 and delta < 1e-6/1e-5."""
    for k in sorted(ks):
        fr = full_results.get(k, [])
        dr = delta_results.get(k, [])
        if not fr or not dr:
            continue
        full_ok = all(r["median_rel"] < 1e-5 and r["max_rel"] < 1e-4
                      for r in fr)
        delta_ok = all(r["median_rel"] < 1e-6 and r["max_rel"] < 1e-5
                       for r in dr)
        if full_ok and delta_ok:
            return k
    return None
