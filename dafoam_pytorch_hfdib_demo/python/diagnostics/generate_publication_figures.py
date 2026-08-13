"""Generate publication figures from all existing experimental data."""
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


def load_eval_metrics(path):
    """Load evaluation metrics.json and return mean rel_U, rel_p."""
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        m = json.load(f)
    us = [x["physics"]["rel_velocity_l2"] for x in m]
    ps = [x["physics"]["rel_pressure_l2"] for x in m]
    return float(np.mean(us)), float(np.mean(ps))


# ================================================================
# Figure 1: Fixed-budget tradeoff (N×K=2560)
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

configs = [
    (496, 5, "outputs/fixed_budget_study/eval_n480_k5/metrics.json"),
    (256, 10, "outputs/fixed_budget_study/eval_n256_k10/metrics.json"),
    (128, 20, "outputs/fixed_budget_study/eval_n128_k20/metrics.json"),
    (64, 40, "outputs/fixed_budget_study/eval_n64_k40/metrics.json"),
    (32, 80, "outputs/fixed_budget_study/eval_n32_k80/metrics.json"),
]

ks, rel_us, rel_ps, ns = [], [], [], []
for n, k, path in configs:
    u, p = load_eval_metrics(path)
    if u is not None:
        ks.append(k)
        rel_us.append(u)
        rel_ps.append(p)
        ns.append(n)

ax1.plot(ks, rel_us, "bo-", markersize=8, linewidth=2)
for i, (k, n, u) in enumerate(zip(ks, ns, rel_us)):
    ax1.annotate(f"N={n}", (k, u), textcoords="offset points",
                xytext=(10, 5), fontsize=9)
ax1.set_xlabel("SIMPLE iterations per topology (K)", fontsize=12)
ax1.set_ylabel("Mean test rel. velocity error", fontsize=12)
ax1.set_title("Fixed CFD budget: N×K = 2560\nVelocity", fontsize=13)
ax1.set_xscale("log")
ax1.set_xticks(ks)
ax1.set_xticklabels([str(k) for k in ks])
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(rel_us) * 1.2)

ax2.plot(ks, rel_ps, "rs-", markersize=8, linewidth=2)
for i, (k, n, p) in enumerate(zip(ks, ns, rel_ps)):
    ax2.annotate(f"N={n}", (k, p), textcoords="offset points",
                xytext=(10, 5), fontsize=9)
ax2.set_xlabel("SIMPLE iterations per topology (K)", fontsize=12)
ax2.set_ylabel("Mean test rel. pressure error", fontsize=12)
ax2.set_title("Fixed CFD budget: N×K = 2560\nPressure", fontsize=13)
ax2.set_xscale("log")
ax2.set_xticks(ks)
ax2.set_xticklabels([str(k) for k in ks])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_fixed_budget.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[fig] Saved fig1_fixed_budget.png")

# ================================================================
# Figure 2: Data scaling at K=20
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Data scaling results
scaling = [
    (16, 0.849, 0.401),
    (32, 0.467, 0.391),
    (64, 0.374, 0.394),
    (128, 0.298, 0.381),
    (480, 0.206, 0.369),
]

ns_sc = [s[0] for s in scaling]
us_sc = [s[1] for s in scaling]
ps_sc = [s[2] for s in scaling]

ax1.plot(ns_sc, us_sc, "bo-", markersize=8, linewidth=2)
ax1.set_xlabel("Number of training topologies (N)", fontsize=12)
ax1.set_ylabel("Mean test rel. velocity error", fontsize=12)
ax1.set_title("Data scaling at K=20\nVelocity", fontsize=13)
ax1.set_xscale("log")
ax1.grid(True, alpha=0.3)

ax2.plot(ns_sc, ps_sc, "rs-", markersize=8, linewidth=2)
ax2.set_xlabel("Number of training topologies (N)", fontsize=12)
ax2.set_ylabel("Mean test rel. pressure error", fontsize=12)
ax2.set_title("Data scaling at K=20\nPressure", fontsize=13)
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_data_scaling.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[fig] Saved fig2_data_scaling.png")

# ================================================================
# Figure 3: Warm-start convergence (16 test topologies)
# ================================================================
warm_path = Path(PROJECT_ROOT) / "outputs" / "warm_start_evaluation" / "metrics.json"
if warm_path.exists():
    with open(warm_path) as f:
        warm_data = json.load(f)

    steps = [0, 1, 2, 3, 5, 10, 20]
    w0_means, w0_stds = [], []
    nn_means, nn_stds = [], []

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
        else:
            w0_means.append(np.nan)
            w0_stds.append(0)
            nn_means.append(np.nan)
            nn_stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(steps, w0_means, yerr=w0_stds, fmt="ko-", label="W₀ initialization",
                markersize=8, linewidth=2, capsize=5)
    ax.errorbar(steps, nn_means, yerr=nn_stds, fmt="bo-", label="NN initialization",
                markersize=8, linewidth=2, capsize=5)
    ax.set_xlabel("SIMPLE iterations after initialization", fontsize=12)
    ax.set_ylabel("Mean test rel. velocity error", fontsize=12)
    ax.set_title("CFD warm-start: W₀ vs NN initialization\n(mean ± std over 16 held-out topologies)",
                fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_warmstart.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[fig] Saved fig3_warmstart.png")

# ================================================================
# Figure 4: Summary table
# ================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")

# Build summary text
lines = [
    ["Method", "CFD work/sample", "N_train", "test rel_U", "test rel_p"],
    ["W₀ baseline", "0", "—", "1.000", "1.000"],
    ["K=5 distill", "5 SIMPLE", "496", f"{load_eval_metrics('outputs/fixed_budget_study/eval_n480_k5/metrics.json')[0]:.3f}",
     f"{load_eval_metrics('outputs/fixed_budget_study/eval_n480_k5/metrics.json')[1]:.3f}"],
    ["K=10 distill", "10 SIMPLE", "256", f"{load_eval_metrics('outputs/fixed_budget_study/eval_n256_k10/metrics.json')[0]:.3f}",
     f"{load_eval_metrics('outputs/fixed_budget_study/eval_n256_k10/metrics.json')[1]:.3f}"],
    ["K=20 distill", "20 SIMPLE", "128", f"{load_eval_metrics('outputs/fixed_budget_study/eval_n128_k20/metrics.json')[0]:.3f}",
     f"{load_eval_metrics('outputs/fixed_budget_study/eval_n128_k20/metrics.json')[1]:.3f}"],
    ["K=40 distill", "40 SIMPLE", "64", f"{load_eval_metrics('outputs/fixed_budget_study/eval_n64_k40/metrics.json')[0]:.3f}",
     f"{load_eval_metrics('outputs/fixed_budget_study/eval_n64_k40/metrics.json')[1]:.3f}"],
    ["K=80 distill", "80 SIMPLE", "32", f"{load_eval_metrics('outputs/fixed_budget_study/eval_n32_k80/metrics.json')[0]:.3f}",
     f"{load_eval_metrics('outputs/fixed_budget_study/eval_n32_k80/metrics.json')[1]:.3f}"],
    ["K=20 (N=480)", "20 SIMPLE", "480", "0.206", "0.369"],
]

table = ax.table(cellText=lines[1:], colLabels=lines[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# Color the best row
for j in range(5):
    table[4, j].set_facecolor("#90EE90")  # K=20 row (best velocity)

ax.set_title("Summary: Fixed-budget distillation (N×K≈2560)\nGreen = best velocity configuration",
             fontsize=14, pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig4_summary_table.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"[fig] Saved fig4_summary_table.png")

print(f"\nAll figures saved to {OUTPUT_DIR}")
