"""Build the supervisor package from persisted warm-start campaign outputs."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/warmstart_convergence"
FINAL = ROOT / "outputs/final_dafoam"
PACKAGE = FINAL / "supervisor_2026_08_24"
CASES = [f"topology_{index:04d}" for index in range(290, 298)]
METHODS = ("cold", "teacher_k20", "neural", "neural_t1", "neural_t5")


def read(case, method):
    return json.loads((SOURCE / case / f"{method}.json").read_text())


def bootstrap(values):
    rng = np.random.default_rng(20260824)
    samples = values[rng.integers(0, len(values), size=(10000, len(values)))]
    return {
        "mean_95_ci": np.quantile(np.mean(samples, axis=1), [0.025, 0.975]).tolist(),
        "median_95_ci": np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist(),
    }


def main():
    figures = PACKAGE / "figures"
    convergence_out = PACKAGE / "warmstart_convergence"
    figures.mkdir(parents=True, exist_ok=True)
    if convergence_out.exists():
        shutil.rmtree(convergence_out)
    shutil.copytree(SOURCE, convergence_out)

    rows = []
    long_rows = []
    for case in CASES:
        data = {method: read(case, method) for method in METHODS}
        cold, neural = data["cold"], data["neural"]
        row = {
            "case_id": case,
            "status_cold": cold["status"], "status_neural": neural["status"],
            "N_cold": cold["total_simple_steps"],
            "N_neural": neural["total_simple_steps"],
            "N_neural_T1": data["neural_t1"]["total_simple_steps"],
            "N_neural_T5": data["neural_t5"]["total_simple_steps"],
            "N_teacher_remaining": data["teacher_k20"]["remaining_steps"],
            "N_teacher_total": data["teacher_k20"]["total_simple_steps"],
            "iteration_saving_neural": 1 - neural["total_simple_steps"] / cold["total_simple_steps"],
            "iteration_saving_T1": 1 - data["neural_t1"]["total_simple_steps"] / cold["total_simple_steps"],
            "iteration_saving_T5": 1 - data["neural_t5"]["total_simple_steps"] / cold["total_simple_steps"],
            "t_cold": cold["total_time_s"], "t_neural": neural["total_time_s"],
            "t_neural_T1": data["neural_t1"]["total_time_s"],
            "t_neural_T5": data["neural_t5"]["total_time_s"],
            "wall_speedup_neural": cold["total_time_s"] / neural["total_time_s"],
            "wall_speedup_T1": cold["total_time_s"] / data["neural_t1"]["total_time_s"],
            "wall_speedup_T5": cold["total_time_s"] / data["neural_t5"]["total_time_s"],
            "initial_rel_U_cold": cold["history"][0]["rel_u"],
            "initial_rel_U_neural": neural["history"][0]["rel_u"],
            "final_rel_U": neural["final_rel_u"], "final_rel_p": neural["final_rel_p"],
            "final_rho": neural["final_rho"],
        }
        for step in (1, 5):
            old = json.loads((FINAL / "warm_start_raw.json").read_text())[case]["warm_start"]
            for start, key in (("W0", "cold"), ("W_NN", "neural")):
                match = next(item for item in old if item["start"] == start and item["k"] == step)
                row[f"rel_U_after_{step}_{key}"] = match["rel_u"]
        rows.append(row)
        for method in METHODS:
            for point in data[method]["history"]:
                long_rows.append({"case_id": case, "method": method, **point})

    def write_csv(path, records):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)

    write_csv(PACKAGE / "test_case_metrics.csv", rows)
    write_csv(PACKAGE / "equal_budget_metrics.csv", long_rows)
    savings = np.array([row["iteration_saving_neural"] for row in rows])
    speedups = np.array([row["wall_speedup_neural"] for row in rows])
    cold5 = np.array([row["rel_U_after_5_cold"] for row in rows])
    neural5 = np.array([row["rel_U_after_5_neural"] for row in rows])
    summary = {
        "test_cases": len(rows),
        "iteration_saving": {"mean": float(savings.mean()), "median": float(np.median(savings)),
                             "iqr": np.quantile(savings, [0.25, 0.75]).tolist(),
                             "min": float(savings.min()), "max": float(savings.max()),
                             "wins": int(np.sum(savings > 0)), **bootstrap(savings)},
        "wall_speedup": {"mean": float(speedups.mean()), "median": float(np.median(speedups)),
                         "min": float(speedups.min()), "max": float(speedups.max()),
                         "wins": int(np.sum(speedups > 1))},
        "equal_budget_k5": {"cold_rel_u_mean": float(cold5.mean()),
                            "neural_rel_u_mean": float(neural5.mean()),
                            "neural_wins": int(np.sum(neural5 < cold5))},
    }
    (PACKAGE / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.axis("off")
    boxes = [(0.08, .62, "TPFM topology\n$\\lambda$"), (.32, .62, "Seed-22 network"),
             (.56, .62, "$W_\\theta$"), (.80, .62, "DAFoam SIMPLE\n$\\rightarrow W^*$"),
             (.32, .18, "$W_0$ -- 20 SIMPLE --> $W_{20}$\ntraining target")]
    for x0, y0, text in boxes:
        ax.text(x0, y0, text, ha="center", va="center", fontsize=11,
                bbox={"boxstyle":"round,pad=.5", "facecolor":"#edf6f9", "edgecolor":"#457b9d"})
    for a, b in ((.14,.26),(.39,.50),(.62,.73)):
        ax.annotate("", xy=(b,.62), xytext=(a,.62), arrowprops={"arrowstyle":"->","lw":1.8})
    ax.annotate("training only", xy=(.32,.49), xytext=(.32,.30), ha="center", arrowprops={"arrowstyle":"->","lw":1.4})
    ax.text(.55,.08,"Training uses $W_{20}$; evaluation uses converged $W^*$ (never a training label).",ha="center",fontsize=10)
    fig.tight_layout(); fig.savefig(figures / "01_solver_distillation_pipeline.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([r["N_cold"] for r in rows], [r["N_neural"] for r in rows], color="#bb3e03")
    bounds = [min(r["N_cold"] for r in rows) - 10, max(r["N_neural"] for r in rows) + 10]
    ax.plot(bounds, bounds, "--", color="0.35"); ax.set(xlabel="Cold SIMPLE calls", ylabel="Neural SIMPLE calls")
    fig.tight_layout(); fig.savefig(figures / "05_iterations_to_convergence.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["case_id"].split("_")[-1] for r in rows], 100 * savings, color=np.where(savings > 0, "#2a9d8f", "#bb3e03"))
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Held-out topology", ylabel="Iteration saving (%)")
    fig.tight_layout(); fig.savefig(figures / "05b_iteration_savings_per_case.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["case_id"].split("_")[-1] for r in rows], speedups, color="#457b9d"); ax.axhline(1, ls="--", color="black")
    ax.set(xlabel="Held-out topology", ylabel="Cold / neural wall time")
    fig.tight_layout(); fig.savefig(figures / "06_walltime_speedup.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(rows)); ax.plot(x, cold5, "o-", label="Cold + 5 SIMPLE"); ax.plot(x, neural5, "o-", label="Neural + 5 SIMPLE")
    ax.set(xlabel="Held-out case", ylabel="Relative velocity error"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "03_equal_budget_velocity_error.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, color in (("cold", "#264653"), ("neural", "#e76f51"), ("neural_t5", "#2a9d8f")):
        grid = np.arange(0, 801, 5); curves = []
        for case in CASES:
            history = read(case, method)["history"]
            xp = np.array([p["simple_steps"] + (5 if method == "neural_t5" else 0) for p in history])
            yp = np.array([p["rho_residual"] for p in history]); curves.append(np.interp(grid, xp, yp, left=np.nan, right=np.nan))
        curves = np.array(curves); median = np.nanmedian(curves, axis=0); q1, q3 = np.nanquantile(curves, [0.25, 0.75], axis=0)
        ax.plot(grid, median, label=method, color=color); ax.fill_between(grid, q1, q3, alpha=.18, color=color)
    ax.set_yscale("log"); ax.set(xlabel="Total SIMPLE calls", ylabel="Normalized DAFoam residual"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "04_residual_convergence.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    baseline = [18.0, 19.6, 20.3]; reconstructed = [15.89, 18.51, 18.52]
    x = np.arange(3); ax.bar(x-.18, baseline, .36, label="Local 64x64"); ax.bar(x+.18, reconstructed, .36, label="32-cell reconstruction")
    ax.set(xticks=x, xticklabels=["0", "274", "549"], ylabel="Velocity discrepancy (%)", title="TPFM_TOPOLOGIES_ONLY: reproduction gate failed"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "08_tpfm_compatibility_audit.png", dpi=220); plt.close(fig)
    raw = json.loads((FINAL / "warm_start_raw.json").read_text())
    teacher_u = [raw[c]["teacher_rel_errors"]["rel_u"] for c in CASES]
    network_u = [raw[c]["rel_errors"]["rel_u"] for c in CASES]
    network_t5_u = [next(v["rel_u"] for v in raw[c]["warm_start"] if v["start"]=="W_NN" and v["k"]==5) for c in CASES]
    teacher_quality = list(csv.DictReader((FINAL / "teacher_quality.csv").open()))
    teacher_p = [float(r["rel_p"]) for r in teacher_quality]
    network_p = [raw[c]["rel_errors"]["rel_p"] for c in CASES]
    fig, axes = plt.subplots(1,2,figsize=(9,4))
    axes[0].boxplot([teacher_u,network_u,network_t5_u],tick_labels=["$W_{20}$","$W_\\theta$","$T^5(W_\\theta)$"]); axes[0].set_ylabel("Relative velocity error")
    axes[1].boxplot([teacher_p,network_p],tick_labels=["$W_{20}$","$W_\\theta$"]); axes[1].set_ylabel("Relative pressure error")
    fig.tight_layout(); fig.savefig(figures / "07_teacher_network_accuracy.png", dpi=220); plt.close(fig)

    positive = summary["iteration_saving"]["median"] > 0
    conclusion = ("Neural initialization reduces iterative CFD work." if positive else
                  "Neural initialization improves equal-budget flow approximation, but does not reduce total convergence iterations.")
    report = f"""# DAFoam Final Campaign\n\n## Compatibility\nExact TPFM CFD reproduction did not satisfy the gate. Results use the TPFM topology distribution under our DAFoam/HFDIB formulation.\n\n## Training\nSeed 22 was selected on validation performance. Training targets are states after 20 SIMPLE iterations; converged CFD states were not training labels.\n\n## Result\n{conclusion}\n\n- Test cases: {len(rows)}/16 available campaign cases (frozen first eight)\n- Median iteration saving: {100*summary['iteration_saving']['median']:.2f}%\n- Mean iteration saving: {100*summary['iteration_saving']['mean']:.2f}%\n- Iteration wins: {summary['iteration_saving']['wins']}/{len(rows)}\n- Median wall speedup: {summary['wall_speedup']['median']:.3f}x\n- Equal-budget k=5 mean velocity error: cold {summary['equal_budget_k5']['cold_rel_u_mean']:.4f}, neural {summary['equal_budget_k5']['neural_rel_u_mean']:.4f}\n- Equal-budget k=5 wins: {summary['equal_budget_k5']['neural_wins']}/{len(rows)}\n\n## Limitation\nThe one-shot network is not a converged CFD replacement, and pressure accuracy remains weak.\n"""
    (PACKAGE / "PRESENTATION_RESULTS.md").write_text(report)
    (PACKAGE / "README.md").write_text("# Supervisor Package\n\nGenerated from persisted held-out DAFoam trajectories. See `PRESENTATION_RESULTS.md`.\n")
    (PACKAGE / "CLAIMS.md").write_text("# Supported\n\n- Training used K=20 truncated SIMPLE states, not converged CFD labels.\n- Evaluation uses held-out TPFM topologies under our DAFoam/HFDIB formulation.\n- Equal-budget neural starts improve velocity error on the evaluated cases.\n\n# Not Supported\n\n- Exact reproduction of the published CFD benchmark.\n- One-shot CFD replacement or accurate pressure prediction.\n- Reduced convergence iterations; the measured eight-case result is negative.\n")
    manifest = {}
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest[str(path.relative_to(PACKAGE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
