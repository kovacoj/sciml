"""Create an immutable manifest for the external TPFM reference archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(array: np.ndarray) -> list[dict]:
    return [
        {
            "channel": index,
            "min": float(channel.min()),
            "max": float(channel.max()),
            "mean": float(channel.mean()),
            "std": float(channel.std()),
        }
        for index, channel in enumerate(array.transpose(1, 0, 2, 3))
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    archive = args.reference_root / "data" / "mixer_64.npz"
    test_archive = args.reference_root / "data" / "mixer_64_test.npz"
    with np.load(archive) as data:
        inputs = data["inputs"]
        outputs = data["outputs"]
        sampled_unique = np.unique(inputs[::max(1, len(inputs) // 25)])
        manifest = {
            "source": "techMathGroup/tpfm_unet",
            "source_git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.reference_root, text=True
            ).strip(),
            "source_file": str(archive),
            "sha256": sha256(archive),
            "n_samples": int(inputs.shape[0]),
            "input_shape": list(inputs.shape),
            "output_shape": list(outputs.shape),
            "input_dtype": str(inputs.dtype),
            "output_dtype": str(outputs.dtype),
            "input_channels": stats(inputs),
            "output_channels": stats(outputs),
            "output_channel_order": ["Ux", "Uy", "Uz", "p"],
            "sampled_lambda_unique_count": int(sampled_unique.size),
            "sampled_lambda_unique": sampled_unique[:100].tolist(),
            "published_test_archive_present": test_archive.exists(),
            "split_policy": (
                "Preserve mixer_64_test.npz exactly."
                if test_archive.exists()
                else "No published test archive is present; no published test split is claimed."
            ),
            "training_policy": "Only inputs may be used to generate truncated-SIMPLE training targets.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
