"""Plot the frozen direct Brinkman alpha gate."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    outputs = Path("outputs")
    destination = outputs / "brinkman_topologies"
    alpha_runs = (50, 100, 2500, 10000, 100000, 1000000)
    rows = []
    for alpha in alpha_runs:
        path = outputs / f"brinkman_direct_A_alpha{alpha}/metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    resolution64 = []
    for alpha in (625, 1250, 2500):
        path = outputs / f"brinkman_direct_A_r64_alpha{alpha}/metrics.json"
        if path.exists():
            resolution64.append(json.loads(path.read_text()))
    sharp_path = outputs / "brinkman_direct_A_r64_alpha2500_sharp/metrics.json"
    sharp = json.loads(sharp_path.read_text()) if sharp_path.exists() else None
    for row in rows:
        h = 0.128 / row["resolution"]
        row["brinkman_length"] = math.sqrt(0.01 / row["alpha"])
        row["brinkman_length_over_h"] = row["brinkman_length"] / h
    for row in resolution64:
        h = 0.128 / row["resolution"]
        row["brinkman_length"] = math.sqrt(0.01 / row["alpha"])
        row["brinkman_length_over_h"] = row["brinkman_length"] / h
    (destination / "brinkman_alpha_gate.json").write_text(json.dumps({
        "claim": "variational Brinkman approximation, not article HFDIB",
        "neural_training_launched": False,
        "reason": "solid leakage remained high after bounded mesh and topology-field checks",
        "resolution32_diffuse": rows,
        "resolution64_diffuse": resolution64,
        "resolution64_sharp_alpha2500": sharp,
    }, indent=2) + "\n")
    figure, left = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    alpha = [row["alpha"] for row in rows]
    leakage = [row["solid_leakage"] for row in rows]
    divergence = [row["divergence_l2"] for row in rows]
    left.semilogx(alpha, leakage, "o-", label="solid leakage")
    left.set_xlabel(r"Brinkman penalty $\alpha$")
    left.set_ylabel("solid leakage ratio")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.semilogx(alpha, divergence, "s--", color="tab:red", label="divergence L2")
    right.set_ylabel("divergence L2", color="tab:red")
    if resolution64:
        left.semilogx(
            [row["alpha"] for row in resolution64],
            [row["solid_leakage"] for row in resolution64],
            "D-", label="solid leakage (64x64)",
        )
    if sharp:
        left.scatter(
            [sharp["alpha"]], [sharp["solid_leakage"]], marker="*", s=100,
            color="tab:green", label="sharp chi (64x64)", zorder=5,
        )
    left.axvline(625, color="0.45", linestyle=":", linewidth=1.2,
                 label=r"$\delta_B=h$ (32x32)")
    left.axvline(2500, color="0.65", linestyle="-.", linewidth=1.2,
                 label=r"$\delta_B=h$ (64x64)")
    handles, labels = [], []
    for axis in (left, right):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles); labels.extend(axis_labels)
    left.legend(handles, labels, loc="center right", fontsize=8)
    left.set_title("Topology A direct Brinkman gate (32x32)")
    figure.savefig(destination / "figure_brinkman_alpha_gate.png", dpi=190)
    plt.close(figure)


if __name__ == "__main__":
    main()
