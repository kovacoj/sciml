"""Trajectory audit, residual decomposition, and field-error-vs-residual figures."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "outputs/final_dafoam/supervisor_2026_08_24"
RAW = ROOT / "outputs/final_dafoam/warm_start_raw.json"
BENCH = ROOT / "outputs/final_dafoam/unet_parity/baseline_benchmark"
OUT = ROOT / "outputs/final_dafoam/final_followup"
OUT.mkdir(parents=True, exist_ok=True)
CASES = [f"topology_{i:04d}" for i in range(290, 298)]

def load_conv(case, method):
    return json.loads((PACKAGE / "warmstart_convergence" / case / f"{method}.json").read_text())

# ===== P1: Trajectory integrity audit =====
raw = json.loads(RAW.read_text())
audit_rows = []
audit_pass = True
for case in CASES:
    if case not in raw:
        continue
    for start_label, start_key in (("cold", "W0"), ("neural", "W_NN")):
        entries = [e for e in raw[case]["warm_start"] if e["start"] == start_key]
        prev_u = None
        for entry in sorted(entries, key=lambda e: e["k"]):
            delta = abs(entry["rel_u"] - prev_u) if prev_u is not None else float("inf")
            if entry["k"] > 0 and delta < 1e-14:
                audit_pass = False
            audit_rows.append({
                "case_id": case, "start_type": start_label, "k": entry["k"],
                "rel_u": entry["rel_u"], "rel_p": entry["rel_p"],
                "state_delta_proxy": delta if prev_u is not None else 0.0,
                "continuity": entry.get("continuity", ""),
                "dp_ratio": entry.get("dp_ratio", ""),
                "time_s": entry.get("time_s", 0.0),
            })
            prev_u = entry["rel_u"]

with (OUT / "trajectory_integrity.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
    w.writeheader(); w.writerows(audit_rows)

# Also verify from convergence histories that k=0 != k=5
conv_audit = []
for case in CASES:
    for method in ("cold", "neural", "teacher_k20"):
        try:
            data = load_conv(case, method)
            h = data["history"]
            if len(h) >= 2:
                delta = abs(h[1]["rho_residual"] - h[0]["rho_residual"])
                conv_audit.append({"case_id": case, "method": method,
                    "k0_rho": h[0]["rho_residual"], "k5_rho": h[1]["rho_residual"],
                    "delta": delta, "passes": delta > 1e-14})
        except FileNotFoundError:
            pass

(OUT / "trajectory_integrity.json").write_text(json.dumps({
    "status": "PASS" if audit_pass else "FAIL",
    "source": "warm_start_raw.json sequential k=0,1,2,5 entries plus convergence history k=0,5",
    "cases_audited": len(set(r["case_id"] for r in audit_rows)),
    "total_trajectory_points": len(audit_rows),
    "convergence_history_check": conv_audit,
    "assertion": "all sequential state changes have |delta_rel_u| > 1e-14",
}, indent=2) + "\n")

print(f"P1 TRAJECTORY_AUDIT = {'PASS' if audit_pass else 'FAIL'}")

# ===== P3: Residual block decomposition =====
decomp_rows = []
for case in CASES:
    for method, label in [("cold", "cold"), ("neural", "neural"), ("teacher_k20", "teacher_W20")]:
        try:
            data = load_conv(case, method)
            h = data["history"][0]  # k=0 initial state
            vel_err = float(h["rel_u"])
            pres_err = float(h["rel_p"])
            decomp_rows.append({
                "case_id": case, "state": label,
                "residual_total": h["rho_residual"],
                "residual_momentum_u": h["rho_u"],
                "residual_pressure_p": h["rho_p"],
                "residual_flux_phi": h["rho_phi"],
                "velocity_error": vel_err,
                "pressure_error": pres_err,
            })
        except FileNotFoundError:
            pass

with (OUT / "solver_residual_decomposition.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(decomp_rows[0]))
    w.writeheader(); w.writerows(decomp_rows)

# Aggregate medians by state
agg = {}
for state in ("cold", "neural", "teacher_W20"):
    rows = [r for r in decomp_rows if r["state"] == state]
    if rows:
        agg[state] = {key: float(np.median([r[key] for r in rows])) for key in
                       ("residual_total", "residual_momentum_u", "residual_pressure_p",
                        "residual_flux_phi", "velocity_error", "pressure_error")}

(OUT / "residual_decomposition_summary.json").write_text(json.dumps(agg, indent=2) + "\n")
print(f"P3 RESIDUAL_DECOMPOSITION: {json.dumps({s: {k: round(v, 4) for k, v in d.items()} for s, d in agg.items()})}")

# ===== P4: Field-error-vs-solver-residual figure =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
markers = {"cold": ("o", "#277da1"), "neural": ("s", "#d1495b"), "teacher_W20": ("^", "#2a9d8f")}
for ax, ykey, ylabel in [(axes[0], "residual_total", "Total DAFoam residual (normalized)"),
                          (axes[1], "residual_momentum_u", "Momentum residual (normalized)")]:
    for state in ("cold", "neural", "teacher_W20"):
        rows = [r for r in decomp_rows if r["state"] == state]
        ax.scatter([r["velocity_error"] for r in rows], [r[ykey] for r in rows],
                   marker=markers[state][0], color=markers[state][1], s=60, label=state, alpha=0.8)
    ax.set_xlabel("Velocity error $e_u$")
    ax.set_ylabel(ylabel)
    ax.axhline(1.0, color="black", ls=":", lw=0.8)
    ax.axvline(1.0, color="black", ls=":", lw=0.8)
    ax.legend()
    ax.set_yscale("log")
fig.suptitle("Field error does not imply solver-residual quality", fontsize=13)
fig.savefig(OUT / "field_error_vs_solver_residual.png", dpi=320)
fig.savefig(OUT / "field_error_vs_solver_residual.pdf")
plt.close(fig)
print("P4 field_error_vs_solver_residual.png saved")

# ===== P3 figure: residual components =====
fig, ax = plt.subplots(figsize=(8, 5))
states = ("cold", "neural", "teacher_W20")
components = ("residual_momentum_u", "residual_pressure_p", "residual_flux_phi")
labels = ("Momentum (U)", "Pressure (p)", "Flux (phi)")
x = np.arange(len(components))
width = 0.25
colors = {"cold": "#277da1", "neural": "#d1495b", "teacher_W20": "#2a9d8f"}
for i, state in enumerate(states):
    vals = [agg[state][c] for c in components]
    bars = ax.bar(x + i * width, vals, width, label=state, color=colors[state])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylabel("Normalized residual (median over 8 cases)")
ax.set_title("Solver residual decomposition at initial state")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
fig.savefig(OUT / "solver_residual_components.png", dpi=320)
fig.savefig(OUT / "solver_residual_components.pdf")
plt.close(fig)
print("P3 solver_residual_components.png saved")

# ===== P2: Finite-budget comparison figure (from existing field data) =====
fields = np.load(BENCH / "benchmark_fields.npz")
selected = ["topology_0291", "topology_0292", "topology_0290"]
labels_cases = ["P1: tortuous", "P2: merge/split", "P3: parallel"]
bench_rows = list(csv.DictReader((BENCH / "per_case_metrics.csv").open()))

fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
for i, (case, label) in enumerate(zip(selected, labels_cases)):
    lam = fields[f"{case}_lambda"]
    ref_u = np.linalg.norm(fields[f"{case}_reference_u"][..., :2], axis=2)
    nn_u = np.linalg.norm(fields[f"{case}_solver_distilled_u"][..., :2], axis=2)
    ref_p = fields[f"{case}_reference_p"] - fields[f"{case}_reference_p"].mean()
    nn_p = fields[f"{case}_solver_distilled_p"] - fields[f"{case}_solver_distilled_p"].mean()
    vmax = max(ref_u.max(), nn_u.max())
    pmax = max(np.abs(ref_p).max(), np.abs(nn_p).max())
    for j, (data, cmap, vmin, vmax_) in enumerate([
        (lam, "Greys", 0, 1),
        (ref_u, "viridis", 0, vmax),
        (nn_u, "viridis", 0, vmax),
        (nn_u - ref_u, "magma", None, None),
    ]):
        axes[i, j].imshow(data, origin="lower", cmap=cmap,
                          vmin=vmin, vmax=vmax_)
        axes[i, j].axis("off")
    m = next(r for r in bench_rows if r["case_id"] == case and r["method"] == "solver_distilled")
    axes[i, 0].set_ylabel(f"{label}\n{case[-3:]}\n$e_u$={float(m['relative_velocity_error']):.3f}", fontsize=9)
for ax, title in zip(axes[0], ["λ", "Converged |u|", "NN |u|", "Velocity error"]):
    ax.set_title(title, fontsize=11)
fig.suptitle("Finite-budget velocity: solver-distilled NN vs converged DAFoam", fontsize=13)
fig.savefig(OUT / "dafoam_finite_budget_comparison.png", dpi=320)
fig.savefig(OUT / "dafoam_finite_budget_comparison.pdf")
plt.close(fig)
print("P2 dafoam_finite_budget_comparison.png saved")

# Pressure version
fig, axes = plt.subplots(3, 4, figsize=(13, 10), constrained_layout=True)
for i, (case, label) in enumerate(zip(selected, labels_cases)):
    lam = fields[f"{case}_lambda"]
    ref_p = fields[f"{case}_reference_p"] - fields[f"{case}_reference_p"].mean()
    nn_p = fields[f"{case}_solver_distilled_p"] - fields[f"{case}_solver_distilled_p"].mean()
    teacher_p = fields[f"{case}_teacher_W20_p"] - fields[f"{case}_teacher_W20_p"].mean()
    pmax = max(np.abs(ref_p).max(), np.abs(nn_p).max(), np.abs(teacher_p).max())
    for j, (data, cmap) in enumerate([
        (lam, "Greys"), (ref_p, "coolwarm"), (teacher_p, "coolwarm"), (nn_p, "coolwarm"),
    ]):
        axes[i, j].imshow(data, origin="lower", cmap=cmap, vmin=-pmax, vmax=pmax)
        axes[i, j].axis("off")
    m_nn = next(r for r in bench_rows if r["case_id"] == case and r["method"] == "solver_distilled")
    m_t = next(r for r in bench_rows if r["case_id"] == case and r["method"] == "teacher_W20")
    axes[i, 0].set_ylabel(f"{label}\n{case[-3:]}\nNN $e_p$={float(m_nn['relative_pressure_error']):.3f}\nW$_{{20}}$ $e_p$={float(m_t['relative_pressure_error']):.3f}", fontsize=9)
for ax, title in zip(axes[0], ["λ", "Converged p", "W$_{20}$ teacher p", "NN p"]):
    ax.set_title(title, fontsize=11)
fig.suptitle("Pressure: teacher truncation vs network imitation", fontsize=13)
fig.savefig(OUT / "dafoam_finite_budget_pressure.png", dpi=320)
fig.savefig(OUT / "dafoam_finite_budget_pressure.pdf")
plt.close(fig)
print("P2 dafoam_finite_budget_pressure.png saved")

# ===== P8: Synthesis figure (SVG) =====
svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="620" viewBox="0 0 1100 620">
<rect width="100%" height="100%" fill="white"/>
<text x="550" y="28" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">DAFoam solver-distillation: cost and quality summary</text>
<text x="180" y="60" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#277da1">OFFLINE</text>
<rect x="60" y="75" width="140" height="50" rx="10" fill="#edf6f9" stroke="#277da1"/>
<text x="130" y="105" text-anchor="middle" font-family="sans-serif" font-size="11">λ → 20 SIMPLE → W₂₀</text>
<text x="130" y="145" text-anchor="middle" font-family="sans-serif" font-size="10">~1.9 s/case × 256 = ~486 s</text>
<rect x="280" y="75" width="120" height="50" rx="10" fill="#edf6f9" stroke="#277da1"/>
<text x="340" y="105" text-anchor="middle" font-family="sans-serif" font-size="11">Train NN</text>
<text x="340" y="145" text-anchor="middle" font-family="sans-serif" font-size="10">~200 s</text>
<path d="M200 100 L280 100" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<text x="550" y="60" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#2a9d8f">ONLINE</text>
<rect x="470" y="75" width="100" height="50" rx="10" fill="#d8f3dc" stroke="#2a9d8f"/>
<text x="520" y="100" text-anchor="middle" font-family="sans-serif" font-size="11">NN inference</text>
<text x="520" y="118" text-anchor="middle" font-family="sans-serif" font-size="10">~13.6 ms</text>
<rect x="620" y="75" width="100" height="50" rx="10" fill="#d8f3dc" stroke="#2a9d8f"/>
<text x="670" y="100" text-anchor="middle" font-family="sans-serif" font-size="11">+ 5 SIMPLE</text>
<text x="670" y="118" text-anchor="middle" font-family="sans-serif" font-size="10">~0.49 s</text>
<rect x="770" y="75" width="120" height="50" rx="10" fill="#ffe5d9" stroke="#e76f51"/>
<text x="830" y="100" text-anchor="middle" font-family="sans-serif" font-size="11">→ convergence</text>
<text x="830" y="118" text-anchor="middle" font-family="sans-serif" font-size="10">~58.9 s</text>
<path d="M570 100 L620 100" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<path d="M720 100 L770 100" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<text x="960" y="75" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="#e76f51">Cold CFD</text>
<rect x="910" y="85" width="100" height="40" rx="10" fill="#ffe5d9" stroke="#e76f51"/>
<text x="960" y="110" text-anchor="middle" font-family="sans-serif" font-size="11">~60.0 s</text>
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>
<text x="180" y="200" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">Finite-budget result</text>
<rect x="60" y="215" width="450" height="140" rx="12" fill="#f7f7f7" stroke="#999"/>
<text x="80" y="245" font-family="sans-serif" font-size="12">✓ NN + 5 SIMPLE improves velocity on 8/8 cases</text>
<text x="80" y="268" font-family="sans-serif" font-size="12">  cold k=5: median e_u = 0.7386</text>
<text x="80" y="288" font-family="sans-serif" font-size="12">  NN   k=5: median e_u = 0.5881</text>
<text x="80" y="311" font-family="sans-serif" font-size="12">  improvement: 20.4%</text>
<text x="80" y="334" font-family="sans-serif" font-size="12" fill="#888">  cost: 0.49 s vs 60.0 s cold</text>
<text x="800" y="200" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">Full convergence</text>
<rect x="560" y="215" width="450" height="140" rx="12" fill="#f7f7f7" stroke="#999"/>
<text x="580" y="245" font-family="sans-serif" font-size="12">✗ No iteration saving (0/8 wins)</text>
<text x="580" y="268" font-family="sans-serif" font-size="12">  cold: 616 iterations, ~60.0 s</text>
<text x="580" y="288" font-family="sans-serif" font-size="12">  NN:   629 iterations, ~58.9 s</text>
<text x="580" y="311" font-family="sans-serif" font-size="12">  residual ratio NN/cold: 3.86×</text>
<text x="580" y="334" font-family="sans-serif" font-size="12" fill="#888">  field error ratio: 0.646×</text>
<text x="550" y="400" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="bold">Diagnosis</text>
<rect x="60" y="415" width="950" height="120" rx="12" fill="#fff8e1" stroke="#f4a261"/>
<text x="80" y="445" font-family="sans-serif" font-size="12">• Velocity field is closer to converged, but momentum residual is 3.9× worse</text>
<text x="80" y="468" font-family="sans-serif" font-size="12">• Pressure is weak: W₂₀ teacher e_p=0.379, NN e_p=0.995 (imitation adds error)</text>
<text x="80" y="491" font-family="sans-serif" font-size="12">• Break-even: ~11 evaluations (W₂₀) vs ~259 (converged labels)</text>
<text x="80" y="514" font-family="sans-serif" font-size="12">• Next step: physics-informed residual training to improve solver-state quality</text>
<text x="550" y="580" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#888">All times measured on host CPU; paper timings are reported values on unknown hardware</text>
</svg>"""
(OUT / "dafoam_method_summary.svg").write_text(svg)
print("P8 dafoam_method_summary.svg saved")

# ===== Manifest =====
import hashlib
manifest = {}
for path in sorted(OUT.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        manifest[str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Manifest saved")
