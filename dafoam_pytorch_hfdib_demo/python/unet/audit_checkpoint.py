"""Audit solver-distilled checkpoints without modifying datasets or checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def git_head(project: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()


def audit(checkpoint: Path, dataset: Path, project: Path) -> dict:
    import torch
    from pinn.mesh_metadata import MeshMetadata
    from unet.factory import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = ckpt.get("architecture")
    model_kwargs = ckpt.get("model_kwargs", {})
    reasons = []
    if architecture != "simple":
        reasons.append(f"architecture is {architecture!r}, expected 'simple'")
    model = build_model(architecture, **model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    mesh_npz = dataset / "shared" / "mesh_metadata.npz"
    mesh_json = dataset / "shared" / "mesh_metadata.json"
    base_state = dataset / "shared" / "base_state_k0.npy"
    normalization = dataset / "solver_targets" / "target_normalization_k020.json"
    split_path = dataset / "splits.json"
    split = json.loads(split_path.read_text())
    mesh = MeshMetadata.load(str(mesh_npz), str(mesh_json))

    required_metadata = {
        "seed": ckpt.get("seed"),
        "target_k": ckpt.get("target_k"),
        "training_topology_ids": ckpt.get("training_topology_ids"),
        "base_state_sha256": ckpt.get("base_state_sha256"),
        "mesh_metadata_sha256": ckpt.get("mesh_metadata_sha256"),
        "target_normalization_sha256": ckpt.get("target_normalization_sha256"),
        "state_scales": ckpt.get("state_scales"),
        "phi_trainable_indices": ckpt.get("phi_trainable_indices"),
    }
    missing = [key for key, value in required_metadata.items() if value is None]
    if missing:
        reasons.append("checkpoint lacks: " + ", ".join(missing))

    status = "TRUSTED" if not reasons else "INCOMPLETE_PROVENANCE"
    return {
        "status": status,
        "reasons": reasons,
        "checkpoint": str(checkpoint.relative_to(project)),
        "checkpoint_sha256": sha256(checkpoint),
        "git_head_at_audit": git_head(project),
        "architecture": architecture,
        "model_kwargs": model_kwargs,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "mode": ckpt.get("mode"),
        "step": ckpt.get("step"),
        "seed": ckpt.get("seed"),
        "target_k": ckpt.get("target_k"),
        "n_training_samples": (
            len(ckpt["training_topology_ids"])
            if ckpt.get("training_topology_ids") is not None else None
        ),
        "training_topology_ids_hash": (
            json_hash(ckpt["training_topology_ids"])
            if ckpt.get("training_topology_ids") is not None else None
        ),
        "current_dataset": {
            "split_sha256": sha256(split_path),
            "test_topology_ids": split["test"],
            "test_topology_ids_hash": json_hash(split["test"]),
            "base_state_sha256": sha256(base_state),
            "mesh_metadata_npz_sha256": sha256(mesh_npz),
            "mesh_metadata_json_sha256": sha256(mesh_json),
            "target_normalization_k020_sha256": sha256(normalization),
            "n_cells": mesh.n_cells,
            "n_faces": mesh.n_faces,
            "n_internal_faces": mesh.n_internal_faces,
            "n_phi_trainable_expected": mesh.n_internal_faces + 16,
            "state_scales_from_code": {"u": 0.1, "p": 0.01, "phi": 4e-7},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    dataset = args.dataset if args.dataset.is_absolute() else project / args.dataset
    checkpoints = [p if p.is_absolute() else project / p for p in args.checkpoints]
    report = {
        "policy": "Only TRUSTED checkpoints may be used for final warm-start claims.",
        "checkpoints": [audit(p, dataset, project) for p in checkpoints],
    }
    output = args.output if args.output.is_absolute() else project / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "statuses": [entry["status"] for entry in report["checkpoints"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
