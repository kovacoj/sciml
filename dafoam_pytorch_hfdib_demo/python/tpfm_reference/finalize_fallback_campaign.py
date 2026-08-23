"""Consolidate completed fallback results into presentation artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unet.factory import build_model
from unet.train import project_solid_velocity


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std())}


def foam_internal(path: Path, vector: bool) -> np.ndarray:
    source = path.read_text()
    count_match = re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;", source, re.DOTALL)
    if not count_match:
        raise ValueError(f"Cannot parse internalField in {path}")
    count = int(count_match.group(1))
    body = count_match.group(2)
    if vector:
        values = np.array([tuple(map(float, row.split())) for row in re.findall(r"\(([^()]+)\)", body)])
    else:
        values = np.fromstring(body, sep=" ")
    if len(values) != count:
        raise ValueError(f"Parsed {len(values)} values from {path}, expected {count}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--warm", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    args = parser.parse_args()
    args.final.mkdir(parents=True, exist_ok=True)
    warm = json.loads(args.warm.read_text())
    topology_ids = sorted(warm)

    teacher_rows = []
    warm_rows = []
    for topology_id in topology_ids:
        row = warm[topology_id]
        teacher_state = np.load(args.dataset / "solver_targets" / topology_id / "state_k020.npy")
        reference_u = foam_internal(args.dataset / topology_id / "case/5000/U", True)
        reference_p = foam_internal(args.dataset / topology_id / "case/5000/p", False)
        teacher_u = teacher_state[:3 * len(reference_u)].reshape(-1, 3)
        teacher_p = teacher_state[3 * len(reference_u):4 * len(reference_u)]
        rel_u = float(np.linalg.norm(teacher_u[:, :2] - reference_u[:, :2]) / (np.linalg.norm(reference_u[:, :2]) + 1e-30))
        rel_p = float(np.linalg.norm((teacher_p - teacher_p.mean()) - (reference_p - reference_p.mean())) / (np.linalg.norm(reference_p - reference_p.mean()) + 1e-30))
        teacher_rows.append({"topology_id": topology_id, "rel_u": rel_u, "rel_p": rel_p})
        for item in row["warm_start"]:
            warm_rows.append({"topology_id": topology_id, **item})
    with (args.final / "teacher_quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(teacher_rows[0]))
        writer.writeheader(); writer.writerows(teacher_rows)
    with (args.final / "warm_start.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(warm_rows[0]))
        writer.writeheader(); writer.writerows(warm_rows)

    test_ids = json.loads((args.dataset / "splits.json").read_text())["test"]
    seed_rows = []
    torch.set_default_dtype(torch.float64)
    for seed in (11, 22, 33):
        checkpoint = torch.load(args.outputs / f"tpfm_local_distill_seed{seed}/checkpoint.pt", map_location="cpu", weights_only=False)
        model = build_model(checkpoint["architecture"], **checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        errors_u, errors_p = [], []
        with torch.no_grad():
            for topology_id in test_ids:
                lam = np.load(args.dataset / topology_id / "lambda.npy")
                target = np.load(args.dataset / "solver_targets" / topology_id / "q_cell_k020.npy")
                lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
                cell, _ = model(lam_t)
                cell = project_solid_velocity(cell, lam_t).squeeze(0).permute(1, 2, 0).reshape(-1, 3).numpy()
                errors_u.append(float(np.linalg.norm(cell[:, :2] - target[:, :2]) / (np.linalg.norm(target[:, :2]) + 1e-30)))
                errors_p.append(float(np.linalg.norm(cell[:, 2] - target[:, 2]) / (np.linalg.norm(target[:, 2]) + 1e-30)))
        seed_rows.append({"seed": seed, "teacher_rel_u": np.mean(errors_u), "teacher_rel_p": np.mean(errors_p)})
    with (args.final / "seed_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(seed_rows[0])); writer.writeheader(); writer.writerows(seed_rows)
    seed_summary = {
        "seeds": seed_rows,
        "teacher_rel_u": stats([row["teacher_rel_u"] for row in seed_rows]),
        "teacher_rel_p": stats([row["teacher_rel_p"] for row in seed_rows]),
        "parameter_count": 3153264,
    }
    (args.final / "seed_summary.json").write_text(json.dumps(seed_summary, indent=2) + "\n")

    summary = {
        "n_topologies": len(topology_ids),
        "teacher_rel_u": stats([row["rel_u"] for row in teacher_rows]),
        "teacher_rel_p": stats([row["rel_p"] for row in teacher_rows]),
        "network_rel_u": stats([warm[tid]["rel_errors"]["rel_u"] for tid in topology_ids]),
        "network_rel_p": stats([warm[tid]["rel_errors"]["rel_p"] for tid in topology_ids]),
    }
    for step in (0, 1, 5):
        cold = [row["rel_u"] for row in warm_rows if row["start"] == "W0" and row["k"] == step]
        neural = [row["rel_u"] for row in warm_rows if row["start"] == "W_NN" and row["k"] == step]
        summary[f"step_{step}"] = {"cold_rel_u": stats(cold), "neural_rel_u": stats(neural), "neural_better_count": int(np.sum(np.asarray(neural) < np.asarray(cold)))}
    (args.final / "warm_start_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    transfer = {"status": "NOT_RUN", "reason": "Exact-domain gate failed; bounded campaign prioritized fallback distillation and held-out warm-start evaluation."}
    (args.final / "transfer.csv").write_text("status,reason\nNOT_RUN,exact-domain gate failed\n")
    (args.final / "transfer_summary.json").write_text(json.dumps(transfer, indent=2) + "\n")

    target_meta = json.loads((args.dataset / "solver_targets/metadata.json").read_text())
    (args.final / "target_generation_summary.json").write_text(json.dumps(target_meta, indent=2) + "\n")
    manifest = json.loads((args.dataset / "dataset_manifest.json").read_text())
    (args.final / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar([0, 1], [summary["teacher_rel_u"]["mean"], summary["network_rel_u"]["mean"]], color=["#356859", "#d97745"])
    axis.set_xticks([0, 1], ["K=20 teacher", "network"]); axis.set_ylabel("mean relative velocity error vs converged")
    fig.tight_layout(); fig.savefig(args.final / "figure_teacher_vs_network.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot([0, 1, 5], [summary[f"step_{k}"]["cold_rel_u"]["mean"] for k in (0,1,5)], marker="o", label="cold")
    axis.plot([0, 1, 5], [summary[f"step_{k}"]["neural_rel_u"]["mean"] for k in (0,1,5)], marker="o", label="neural")
    axis.set_xlabel("SIMPLE corrections"); axis.set_ylabel("mean relative velocity error"); axis.legend(); fig.tight_layout()
    fig.savefig(args.final / "figure_warm_start.png", dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 2)); axis.axis("off"); axis.text(0.5, 0.5, "Transfer experiment not run\nExact-domain gate failed", ha="center", va="center")
    fig.savefig(args.final / "figure_transfer.png", dpi=180); plt.close(fig)

    gate = json.loads((args.final / "reproduction_gate.json").read_text())
    report = f"""TPFM reproduction
=================
classification: {gate['classification']}
domain: local 64x64 DAFoam/HFDIB fallback
ROI: 64x64, h=0.002 m
velocity error sample 0: {gate['results'][0]['velocity_rel_l2']:.6f}
velocity error sample 274: {gate['results'][1]['velocity_rel_l2']:.6f}
velocity error sample 549: {gate['results'][2]['velocity_rel_l2']:.6f}
pressure error: {[round(row['pressure_gauge_centered_rel_l2'], 6) for row in gate['results']]}
pressure range ratios: {[round(row['pressure_range_ratio'], 6) for row in gate['results']]}

Training
========
training topology count: 256
K: 20
uses converged CFD labels for training: NO
model: SimpleFlowNet with independent phi
parameter count: 3153264
seeds: 11, 22, 33

## Teacher
mean rel_U W20 vs converged: {summary['teacher_rel_u']['mean']:.6f}
mean rel_p W20 vs converged: {summary['teacher_rel_p']['mean']:.6f}

## Network
mean rel_U: {summary['network_rel_u']['mean']:.6f}
std rel_U: {summary['network_rel_u']['std']:.6f}
mean rel_p: {summary['network_rel_p']['mean']:.6f}
std rel_p: {summary['network_rel_p']['std']:.6f}

Warm start
==========
cold iterations: not measured to a residual stopping criterion
teacher warm iterations: not measured
neural iterations: 0, 1, and 5 correction trajectories measured
T(neural) iterations: mean rel_U {summary['step_1']['neural_rel_u']['mean']:.6f}
T5(neural) iterations: mean rel_U {summary['step_5']['neural_rel_u']['mean']:.6f}
iteration saving: not claimed
wall-clock speedup: not claimed

Transfer
========
scratch residual after 5: NOT RUN
transfer residual after 5: NOT RUN
scratch residual after 20: NOT RUN
transfer residual after 20: NOT RUN
"""
    (args.final / "PRESENTATION_RESULTS.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
