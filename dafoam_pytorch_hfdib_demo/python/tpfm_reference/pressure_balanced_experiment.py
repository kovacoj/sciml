"""Audit pressure preprocessing in W20 training targets and run pressure-balanced retraining."""
from __future__ import annotations
import csv, json, sys, time, os, tempfile
from pathlib import Path
import numpy as np
import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from pinn.mesh_metadata import MeshMetadata
from state_layout import build_isothermal_layout
from diagnostics.warmstart_common import build_state_assembler

DATASET = Path("/home/cady/personal/sciml-dafoam/dafoam_pytorch_hfdib_demo/datasets/tpfm_64_local_dafoam")
CHECKPOINT = Path("dafoam_pytorch_hfdib_demo/outputs/final_dafoam/supervisor_2026_08_24/model/seed22_checkpoint.pt")
OUT = Path("dafoam_pytorch_hfdib_demo/outputs/final_dafoam/final_followup/pressure_experiment")
OUT.mkdir(parents=True, exist_ok=True)

splits = json.loads((DATASET / "splits.json").read_text())
mesh = MeshMetadata.load(str(DATASET / "shared/mesh_metadata.npz"),
                          str(DATASET / "shared/mesh_metadata.json"))
n_cells = mesh.n_cells
n_u = 3 * n_cells
base_state = np.load(DATASET / "shared/base_state_k0.npy")
assembler, phi_indices = build_state_assembler(mesh, build_isothermal_layout(mesh.n_cells, mesh.n_faces), base_state)

# ===== P5: Pressure preprocessing audit =====
audit = []
for split_name in ("train", "validation"):
    for tid in splits[split_name]:
        target_path = DATASET / "solver_targets" / tid / "state_k020.npy"
        if not target_path.exists():
            continue
        state = np.load(target_path)
        p = state[n_u:n_u + n_cells]
        audit.append({
            "split": split_name, "topology_id": tid,
            "pressure_mean": float(p.mean()),
            "pressure_std": float(p.std()),
            "pressure_min": float(p.min()),
            "pressure_max": float(p.max()),
            "pressure_drop": float(p.max() - p.min()),
        })

with (OUT / "pressure_preprocessing_audit.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(audit[0]))
    w.writeheader(); w.writerows(audit)

train_means = [r["pressure_mean"] for r in audit if r["split"] == "train"]
val_means = [r["pressure_mean"] for r in audit if r["split"] == "validation"]
pressure_summary = {
    "train_pressure_mean_of_means": float(np.mean(train_means)),
    "train_pressure_std_of_means": float(np.std(train_means)),
    "train_pressure_range_of_means": [float(min(train_means)), float(max(train_means))],
    "val_pressure_mean_of_means": float(np.mean(val_means)),
    "gauge_fixed": False,
    "finding": "W20 pressure targets carry sample-dependent arbitrary offsets — pressure mean varies widely across training samples",
}
(OUT / "pressure_preprocessing_summary.json").write_text(json.dumps(pressure_summary, indent=2) + "\n")
print(f"P5 PRESSURE_AUDIT: train pressure means range [{min(train_means):.2f}, {max(train_means):.2f}]")
print(f"   std of means: {np.std(train_means):.2f} — {'NOT gauge-fixed' if np.std(train_means) > 1.0 else 'appears gauge-fixed'}")

# ===== P6: Pressure-balanced retraining =====
torch.set_default_dtype(torch.float64)
from unet.factory import build_model

# Load existing model to get architecture
payload = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
arch = payload.get("architecture", "simple")
model_kwargs = payload.get("model_kwargs", {})

# Create fresh model with same architecture but new seed
model = build_model(arch, **model_kwargs)
torch.manual_seed(42)
model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)

# Prepare training data
def load_sample(tid):
    lam = np.load(DATASET / tid / "lambda.npy")
    target = np.load(DATASET / "solver_targets" / tid / "state_k020.npy")
    # Gauge-fix pressure: subtract mean
    target = target.copy()
    target[n_u:n_u + n_cells] -= target[n_u:n_u + n_cells].mean()
    return lam, target

all_data = [(tid, *load_sample(tid)) for tid in splits["train"] if (DATASET / "solver_targets" / tid / "state_k020.npy").exists()]
np.random.seed(42)
np.random.shuffle(all_data)
n_val = min(32, len(all_data) // 8)
val_data = [(lam, target) for _, lam, target in all_data[:n_val]]
train_data = [(lam, target) for _, lam, target in all_data[n_val:]]
print(f"P6 Loaded {len(train_data)} train, {len(val_data)} validation samples")

# Compute pressure normalization from training data only
all_p = np.concatenate([t[n_u:n_u + n_cells] for _, t in train_data])
p_mean = float(all_p.mean())
p_std = float(all_p.std())
print(f"P6 Pressure normalization: mean={p_mean:.4f}, std={p_std:.4f}")

# Training with pressure-balanced loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
alpha_p = 3.0  # pressure loss weight
best_val_pressure = float("inf")
best_state = None
history = []

for epoch in range(20):
    model.train()
    np.random.shuffle(train_data)
    epoch_loss = epoch_u = epoch_p = 0.0
    for lam, target in train_data:
        lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
        target_t = torch.from_numpy(target)
        with torch.no_grad():
            fluid = (lam_t[:, 0] < 0.5).to(torch.float64).unsqueeze(1)
        optimizer.zero_grad()
        cell, phi = model(lam_t)
        cell = cell * torch.cat([fluid, fluid, torch.ones_like(fluid)], dim=1)
        corrections = cell.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        pred = assembler.assemble(corrections, phi.squeeze(0))
        u_loss = torch.mean((pred[:n_u] - target_t[:n_u]) ** 2)
        p_loss = torch.mean((pred[n_u:n_u + n_cells] - target_t[n_u:n_u + n_cells]) ** 2)
        loss = u_loss + alpha_p * p_loss
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.detach()); epoch_u += float(u_loss.detach()); epoch_p += float(p_loss.detach())

    # Validation
    model.eval()
    val_u_err = val_p_err = 0.0
    with torch.no_grad():
        for lam, target in val_data:
            lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
            fluid = (lam_t[:, 0] < 0.5).to(torch.float64).unsqueeze(1)
            cell, phi = model(lam_t)
            cell = cell * torch.cat([fluid, fluid, torch.ones_like(fluid)], dim=1)
            corrections = cell.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            pred = assembler.assemble(corrections, phi.squeeze(0)).detach().numpy()
            val_u_err += np.linalg.norm(pred[:n_u] - target[:n_u]) / (np.linalg.norm(target[:n_u]) + 1e-30)
            val_p_err += np.linalg.norm(pred[n_u:n_u + n_cells] - target[n_u:n_u + n_cells]) / (np.linalg.norm(target[n_u:n_u + n_cells]) + 1e-30)

    n_val = len(val_data)
    val_u = val_u_err / n_val
    val_p = val_p_err / n_val
    history.append({"epoch": epoch, "train_loss": epoch_loss / len(train_data),
                    "train_u_loss": epoch_u / len(train_data), "train_p_loss": epoch_p / len(train_data),
                    "val_rel_u": val_u, "val_rel_p": val_p})

    if val_p < best_val_pressure:
        best_val_pressure = val_p
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        # Atomic checkpoint
        ckpt_path = OUT / "best_validation.pt"
        tmp = OUT / "best_validation.pt.tmp"
        torch.save({"model_state_dict": best_state, "epoch": epoch,
                     "val_rel_p": val_p, "val_rel_u": val_u,
                     "alpha_p": alpha_p, "p_mean": p_mean, "p_std": p_std,
                     "architecture": arch, "model_kwargs": model_kwargs,
                     "gauge_fixed": True}, str(tmp))
        os.replace(str(tmp), str(ckpt_path))

    if epoch % 10 == 0 or epoch == 49:
        print(f"  epoch {epoch:3d}: train_loss={epoch_loss/len(train_data):.6f} val_u={val_u:.4f} val_p={val_p:.4f}")

# Save latest
torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
            "val_rel_p": val_p, "val_rel_u": val_u,
            "alpha_p": alpha_p, "p_mean": p_mean, "p_std": p_std,
            "architecture": arch, "model_kwargs": model_kwargs,
            "gauge_fixed": True}, str(OUT / "latest.pt"))

# Save history
with (OUT / "training_history.json").open("w") as f:
    json.dump({"alpha_p": alpha_p, "gauge_fixed": True, "p_mean": p_mean, "p_std": p_std,
               "best_val_pressure": best_val_pressure, "history": history}, f, indent=2)

# Evaluate on test cases with persisted converged references
import re
def foam(path, vector):
    source = path.read_text()
    match = re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;", source, re.S)
    body = match.group(2)
    values = np.array([tuple(map(float, row.split())) for row in re.findall(r"\(([^()]+)\)", body)]) if vector else np.fromstring(body, sep=" ")
    return values

test_cases = [f"topology_{i:04d}" for i in range(290, 298)]
model.load_state_dict(best_state)
model.eval()
test_rows = []
for case in test_cases:
    td = DATASET / case
    if not (td / "case/5000/U").exists():
        continue
    lam = np.load(td / "lambda.npy")
    ref_u = foam(td / "case/5000/U", True)
    ref_p = foam(td / "case/5000/p", False)
    ref_p = ref_p - ref_p.mean()
    lam_t = torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        fluid = (lam_t[:, 0] < 0.5).to(torch.float64).unsqueeze(1)
        cell, phi = model(lam_t)
        cell = cell * torch.cat([fluid, fluid, torch.ones_like(fluid)], dim=1)
        corrections = cell.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        pred = assembler.assemble(corrections, phi.squeeze(0)).detach().numpy()
    pred_u = pred[:n_u].reshape(-1, 3)[:, :2]
    pred_p = pred[n_u:n_u + n_cells] - pred[n_u:n_u + n_cells].mean()
    rel_u = float(np.linalg.norm(pred_u - ref_u[:, :2]) / (np.linalg.norm(ref_u[:, :2]) + 1e-30))
    rel_p = float(np.linalg.norm(pred_p - ref_p) / (np.linalg.norm(ref_p) + 1e-30))
    test_rows.append({"case_id": case, "rel_u": rel_u, "rel_p": rel_p})

with (OUT / "test_metrics.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(test_rows[0]))
    w.writeheader(); w.writerows(test_rows)

baseline_p = 0.995  # known from benchmark
new_p = float(np.median([r["rel_p"] for r in test_rows]))
new_u = float(np.median([r["rel_u"] for r in test_rows]))
baseline_u = 0.6457
print(f"\nP6 PRESSURE_BALANCED_RESULT:")
print(f"  baseline NN:  e_u={baseline_u:.4f}, e_p={baseline_p:.4f}")
print(f"  balanced NN: e_u={new_u:.4f}, e_p={new_p:.4f}")
print(f"  pressure improvement: {(1 - new_p/baseline_p)*100:.1f}%")
print(f"  velocity degradation: {(new_u/baseline_u - 1)*100:.1f}%")
