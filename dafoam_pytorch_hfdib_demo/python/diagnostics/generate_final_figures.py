"""Generate updated publication figures with N=480 seed variance."""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))
from common import PROJECT_ROOT  # noqa: E402

OUTPUT_DIR = Path(PROJECT_ROOT) / "outputs" / "publication_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_eval(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        m = json.load(f)
    us = [x["physics"]["rel_velocity_l2"] for x in m]
    ps = [x["physics"]["rel_pressure_l2"] for x in m]
    return float(np.mean(us)), float(np.mean(ps))


# ================================================================
# Figure 1: Fixed-budget tradeoff + data scaling (combined)
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Fixed-budget
configs = [
    (496, 5, "outputs/fixed_budget_study/eval_n480_k5/metrics.json"),
    (256, 10, "outputs/fixed_budget_study/eval_n256_k10/metrics.json"),
    (128, 20, "outputs/fixed_budget_study/eval_n128_k20/metrics.json"),
    (64, 40, "outputs/fixed_budget_study/eval_n64_k40/metrics.json"),
    (32, 80, "outputs/fixed_budget_study/eval_n32_k80/metrics.json"),
]

ks, rel_us, rel_ps, ns = [], [], [], []
for n, k, path in configs:
    u, p = load_eval(path)
    if u is not None:
        ks.append(k)
        rel_us.append(u)
        rel_ps.append(p)
        ns.append(n)

ax = axes[0]
ax.plot(ks, rel_us, "bo-", markersize=10, linewidth=2, label="Velocity", zorder=5)
ax.plot(ks, rel_ps, "rs-", markersize=10, linewidth=2, label="Pressure", zorder=5)
for i, (k, n) in enumerate(zip(ks, ns)):
    ax.annotate(f"N={n}", (k, rel_us[i]), textcoords="offset points",
                xytext=(12, -5), fontsize=10, fontweight="bold",
                color="blue")
    ax.annotate(f"N={n}", (k, rel_ps[i]), textcoords="offset points",
                xytext=(12, 5), fontsize=10, fontweight="bold",
                color="red")
ax.set_xlabel("SIMPLE iterations per topology (K)", fontsize=13)
ax.set_ylabel("Mean test relative error", fontsize=13)
ax.set_title("Fixed CFD budget: N×K ≈ 2560\nIntermediate K is optimal", fontsize=14)
ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(max(rel_us), max(rel_ps)) * 1.3)

# Right: Data scaling with seed variance
ax = axes[1]

# N=128 seeds
n128_us = []
for seed in [0, 1, 2]:
    path = f"outputs/seeds/eval_seed{seed}/metrics.json"
    u, _ = load_eval(path)
    if u is not None:
        n128_us.append(u)

# N=480 seeds
n480_us = []
n480_ps = []
for seed in [0, 1, 2]:
    path = f"outputs/overnight_final/eval_n480_seed{seed}/metrics.json"
    u, p = load_eval(path)
    if u is not None:
        n480_us.append(u)
        n480_ps.append(p)

# Data scaling points
scaling_n = [16, 32, 64, 128, 480]
scaling_u = [0.849, 0.467, 0.374, 0.298, 0.206]
scaling_u_std = [0, 0, 0, np.std(n128_us) if len(n128_us) > 1 else 0,
                 np.std(n480_us) if len(n480_us) > 1 else 0]

ax.errorbar(scaling_n, scaling_u, yerr=scaling_u_std,
            fmt="bo-", markersize=10, linewidth=2, capsize=6, capthick=2,
            label="Velocity error")

ax.set_xlabel("Number of training topologies (N)", fontsize=13)
ax.set_ylabel("Mean test rel. velocity error", fontsize=13)
ax.set_title("Data scaling at K=20\nMean ± std over 3 seeds (N=128, N=480)", fontsize=14)
ax.set_xscale("log")
ax.set_xticks(scaling_n)
ax.set_xticklabels([str(n) for n in scaling_n])
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_fixed_budget_and_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[fig] Saved fig1_fixed_budget_and_scaling.png")

# ================================================================
# Figure 2: Warm-start convergence
# ================================================================
warm_path = Path(PROJECT_ROOT) / "outputs" / "warm_start_evaluation" / "metrics.json"
if warm_path.exists():
    with open(warm_path) as f:
        warm_data = json.load(f)

    steps = [0, 1, 2, 3, 5, 10, 20]
    w0_means, w0_stds, nn_means, nn_stds = [], [], [], []

    for k in steps:
        w0_us = [warm_data[tid]["w0"][str(k)]["rel_u"] for tid in warm_data
                 if str(k) in warm_data[tid]["w0"]]
        nn_us = [warm_data[tid]["wnn"][str(k)]["rel_u"] for tid in warm_data
                 if str(k) in warm_data[tid]["wnn"]]
        if w0_us and nn_us:
            w0_means.append(np.mean(w0_us))
            w0_stds.append(np.std(w0_us))
            nn_means.append(np.mean(nn_us))
            nn_stds.append(np.std(nn_us))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(steps, w0_means, yerr=w0_stds, fmt="ko-", label="Standard W₀ init",
                markersize=10, linewidth=2.5, capsize=5, capthick=2)
    ax.errorbar(steps, nn_means, yerr=nn_stds, fmt="bo-", label="NN warm-start init",
                markersize=10, linewidth=2.5, capsize=5, capthick=2)
    ax.set_xlabel("SIMPLE iterations after initialization", fontsize=14)
    ax.set_ylabel("Mean test rel. velocity error", fontsize=14)
    ax.set_title("CFD warm-start: W₀ vs NN initialization\n(mean ± std over 16 held-out topologies)",
                fontsize=14)
    ax.legend(fontsize=13, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add threshold lines
    ax.axhline(y=0.2, color="green", linestyle="--", alpha=0.5, label="e_U=0.2")
    ax.axhline(y=0.15, color="orange", linestyle="--", alpha=0.5, label="e_U=0.15")

    # Annotate speedup
    ax.annotate("NN reaches e_U<0.2 in 5 steps\nW₀ needs 20 steps\n(4× speedup)",
                xy=(5, 0.196), fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_warmstart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved fig2_warmstart.png")

# ================================================================
# Figure 3: Summary table
# ================================================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

lines = [
    ["Method", "CFD work/sample", "N_train", "test rel_U", "test rel_p"],
    ["W₀ baseline", "0", "—", "1.000", "1.000"],
    ["K=5 distill", "5 SIMPLE", "496", f"{load_eval('outputs/fixed_budget_study/eval_n480_k5/metrics.json')[0]:.3f}",
     f"{load_eval('outputs/fixed_budget_study/eval_n480_k5/metrics.json')[1]:.3f}"],
    ["K=10 distill", "10 SIMPLE", "256", f"{load_eval('outputs/fixed_budget_study/eval_n256_k10/metrics.json')[0]:.3f}",
     f"{load_eval('outputs/fixed_budget_study/eval_n256_k10/metrics.json')[1]:.3f}"],
    ["K=20 distill", "20 SIMPLE", "128", f"{load_eval('outputs/fixed_budget_study/eval_n128_k20/metrics.json')[0]:.3f}",
     f"{load_eval('outputs/fixed_budget_study/eval_n128_k20/metrics.json')[1]:.3f}"],
    ["K=40 distill", "40 SIMPLE", "64", f"{load_eval('outputs/fixed_budget_study/eval_n64_k40/metrics.json')[0]:.3f}",
     f"{load_eval('outputs/fixed_budget_study/eval_n64_k40/metrics.json')[1]:.3f}"],
    ["K=80 distill", "80 SIMPLE", "32", f"{load_eval('outputs/fixed_budget_study/eval_n32_k80/metrics.json')[0]:.3f}",
     f"{load_eval('outputs/fixed_budget_study/eval_n32_k80/metrics.json')[1]:.3f}"],
    ["K=20 (more data)", "20 SIMPLE", "480", f"{np.mean(n480_us):.3f}±{np.std(n480_us):.3f}",
     f"{np.mean(n480_ps):.3f}±{np.std(n480_ps):.3f}"],
    ["NN warm-start", "—", "128", "0.31 (direct)", "0.40 (direct)"],
    ["NN+5 SIMPLE", "—", "128", "0.20", "0.39"],
    ["NN+10 SIMPLE", "—", "128", "0.15", "0.35"],
]

table = ax.table(cellText=lines[1:], colLabels=lines[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.0)

# Color the best fixed-budget row
for j in range(5):
    table[4, j].set_facecolor("#90EE90")  # K=20 row

# Color the N=480 row
for j in range(5):
    table[8, j].set_facecolor("#ADD8E6")

ax.set_title("Summary: Truncated-SIMPLE Distillation Results\nGreen = best fixed-budget config; Blue = best with more data",
             fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[fig] Saved fig3_summary_table.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
