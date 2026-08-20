"""Freeze the evaluation split and its content hash as a small JSON artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.split.read_bytes()
    split = json.loads(raw)
    test = split["test"]
    test_hash = hashlib.sha256(
        json.dumps(test, separators=(",", ":")).encode()
    ).hexdigest()
    result = {
        "split_file": str(args.split),
        "split_file_sha256": hashlib.sha256(raw).hexdigest(),
        "n_test": len(test),
        "test_topology_ids": test,
        "test_topology_ids_sha256": test_hash,
        "policy": "Converged states for these IDs are evaluation-only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
