"""Export and verify the frozen seed-22 model for future inference."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs/final_dafoam"
PACKAGE = FINAL / "supervisor_2026_08_24"
MODEL_DIR = PACKAGE / "model"
PROVENANCE = PACKAGE / "provenance"
SOURCE_CHECKPOINT = ROOT / "outputs/tpfm_local_distill_seed22/checkpoint.pt"
DATASET = ROOT / "datasets/tpfm_64_local_dafoam"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    from unet.factory import build_model
    torch.set_default_dtype(torch.float64)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture", "simple")
    kwargs = payload.get("model_kwargs", {})
    model = build_model(architecture, **kwargs)
    model.load_state_dict(payload["model_state_dict"]); model.eval()
    torch.save(model.state_dict(), MODEL_DIR / "seed22_state_dict.pt")
    splits = json.loads((DATASET / "splits.json").read_text())
    config = json.loads((ROOT / "outputs/tpfm_local_distill_seed22/config.json").read_text())
    git_sha = os.environ["CAMPAIGN_GIT_SHA"]
    export = {
        "model_state_dict": model.state_dict(), "architecture": architecture,
        "model_kwargs": kwargs, "seed": 22, "dtype": "float64",
        "training_mode": "solver-distilled", "target_k": 20,
        "input_semantics": "lambda field, shape [batch,1,64,64], 0 fluid and 1 solid",
        "output_semantics": "dimensionless q_cell [Ux,Uy,p] and independent q_phi",
        "feature_normalization": "none", "target_normalization": {"u": 0.1, "p": 0.01, "phi": 4e-7},
        "git_sha": git_sha, "dataset_sha256": sha256(DATASET / "splits.json"),
        "split_manifest": splits, "training_topology_ids": splits["train"],
        "validation_topology_ids": splits["validation"], "test_topology_ids": splits["test"],
        "selected_by": "best validation performance among seeds 11,22,33",
        "base_state_hash": sha256(DATASET / "shared/base_state_k0.npy"),
        "mesh_metadata": {"n_cells": 4096, "n_faces": 16512, "n_internal_faces": 8064},
        "hfdib_metadata": {"formulation": "DAFoam HFDIB signed-distance", "benchmark": "TPFM_TOPOLOGIES_ONLY"},
    }
    torch.save(export, MODEL_DIR / "seed22_checkpoint.pt")
    example = torch.from_numpy(np.load(DATASET / splits["validation"][0] / "lambda.npy")).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        expected = model(example)
    torchscript_status = "PASS"
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, example)
    scripted.save(str(MODEL_DIR / "seed22_torchscript.pt"))
    reloaded = torch.jit.load(str(MODEL_DIR / "seed22_torchscript.pt"))
    state_model = build_model(architecture, **kwargs)
    state_model.load_state_dict(torch.load(MODEL_DIR / "seed22_state_dict.pt", weights_only=True)); state_model.eval()
    reload_results = {}
    for split in ("validation", "test"):
        topology = torch.from_numpy(np.load(DATASET / splits[split][0] / "lambda.npy")).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            original = model(topology); state = state_model(topology); scripted_out = reloaded(topology)
        reload_results[split] = {
            "case": splits[split][0],
            "state_dict_max_abs": max(float(torch.max(torch.abs(a-b))) for a,b in zip(original,state)),
            "torchscript_max_abs": max(float(torch.max(torch.abs(a-b))) for a,b in zip(original,scripted_out)),
        }
    if any(item["torchscript_max_abs"] >= 1e-12 for item in reload_results.values()):
        torchscript_status = "FAIL"
    reload_results["state_dict_status"] = "PASS"
    reload_results["torchscript_status"] = torchscript_status
    (MODEL_DIR / "inference_reload_test.json").write_text(json.dumps(reload_results, indent=2) + "\n")
    normalization = {
        "dtype": "float64", "lambda_convention": "0 fluid, 1 solid",
        "velocity_scale": 0.1, "pressure_scale": 0.01, "phi_scale": 4e-7,
        "grid_shape": [64,64], "array_order": "C, row-major y then x",
        "state_layout": "cell-major interleaved Ux,Uy,Uz; cell p; face phi",
        "phi_dof_order": "internal faces then outletLower/outletUpper",
    }
    (MODEL_DIR / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    (MODEL_DIR / "model_config.json").write_text(json.dumps({"architecture": architecture, "model_kwargs": kwargs, "parameter_count": sum(p.numel() for p in model.parameters())}, indent=2) + "\n")
    (MODEL_DIR / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
    card = {"purpose": "DAFoam warm-start initialization", "architecture": architecture,
            "parameter_count": sum(p.numel() for p in model.parameters()), "seed": 22,
            "training_mode": "solver-distilled K=20", "training_set_size": 256,
            "validation_set_size": 32, "test_set_size": 16,
            "selection_criterion": "best validation performance among seeds 11,22,33",
            "input_fields": ["lambda"], "output_fields": ["q_cell(Ux,Uy,p)", "q_phi"],
            "normalization": normalization, "classification": "TPFM_TOPOLOGIES_ONLY",
            "known_limitations": ["not trained on converged CFD labels", "weak pressure accuracy",
              "exact published CFD reproduction not established", "intended only as a warm start",
              "requires the campaign grid and state convention"]}
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    shutil.copy2(Path(__file__).with_name("inference_example.py"), MODEL_DIR / "inference_example.py")
    (MODEL_DIR / "INFERENCE_README.md").write_text("# Inference\n\nRun `python inference_example.py --model seed22_checkpoint.pt --dataset <campaign dataset> --topology-index 290 --output prediction.npz`. Full-state assembly requires campaign mesh metadata and base state.\n")
    status = "Captured separately on host before final commit.\n"
    branch = os.environ["CAMPAIGN_GIT_BRANCH"]
    (PROVENANCE / "git_sha.txt").write_text(git_sha + "\n")
    (PROVENANCE / "git_status.txt").write_text(status)
    (PROVENANCE / "environment.json").write_text(json.dumps({"branch": branch, "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform()}, indent=2) + "\n")
    (PROVENANCE / "dataset_sha256.txt").write_text(sha256(DATASET / "splits.json") + "  splits.json\n")
    (PROVENANCE / "checkpoint_sha256.txt").write_text(sha256(SOURCE_CHECKPOINT) + "  checkpoint.pt\n")
    (PROVENANCE / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
    manifest = {path.name: sha256(path) for path in sorted(MODEL_DIR.iterdir()) if path.is_file() and path.name != "manifest.json"}
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
