"""Archive Stokes residual-preconditioning pilot data and checkpoints."""

import hashlib
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    root = Path("outputs/stokes_preconditioning_research")
    root.mkdir(parents=True, exist_ok=True)
    files = []
    patterns = (
        Path("outputs/stokes_loss_pilot/final"),
        Path("outputs/stokes_mlp_loss_pilot/final"),
    )
    for source_dir in patterns:
        target_dir = root / source_dir.parent.name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
    checkpoints = root / "networks"
    checkpoints.mkdir(exist_ok=True)
    for source in sorted(Path("outputs/stokes_mlp_loss_pilot").glob("model_*.pt")):
        shutil.copy2(source, checkpoints / source.name)
    runs = root / "runs"
    runs.mkdir(exist_ok=True)
    for source in sorted(Path("outputs/stokes_mlp_loss_pilot").glob("mlp_*.json")):
        shutil.copy2(source, runs / source.name)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append({
                "path": str(path.relative_to(root)), "sha256": digest(path),
                "bytes": path.stat().st_size,
            })
    manifest = {
        "claim": "matched-initialization Stokes residual metric pilot",
        "meshes": ["16x8", "32x16"],
        "seeds": [11, 22, 33, 44, 55],
        "losses": ["raw", "dual", "correction"],
        "optimizer_budget": "300 L-BFGS iterations",
        "network_checkpoints": 30,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
