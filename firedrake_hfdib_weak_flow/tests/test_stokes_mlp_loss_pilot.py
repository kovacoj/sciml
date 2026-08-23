from pathlib import Path

import torch


def test_archived_mlp_pilot_checkpoints_load():
    root = Path(__file__).parents[1] / "outputs/stokes_preconditioning_research/networks"
    if not root.exists():
        return
    paths = sorted(root.glob("model_*.pt"))
    assert len(paths) == 30
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert payload["loss"] in {"raw", "dual", "correction"}
    assert payload["seed"] in {11, 22, 33, 44, 55}
    assert payload["model_state"]
