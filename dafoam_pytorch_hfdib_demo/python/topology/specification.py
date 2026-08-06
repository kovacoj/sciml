"""Topology specification and dataset discovery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass(frozen=True)
class TopologySpec:
    topology_id: str
    mask: np.ndarray
    design_x_min: float
    design_x_max: float
    design_y_min: float
    design_y_max: float
    source_path: Path | None = None
    solid_value: int = 1

    def __post_init__(self):
        if self.mask.ndim != 2:
            raise ValueError("mask must be 2D")
        if not np.all(np.isin(self.mask, [0, 1])):
            raise ValueError("mask must contain only 0 and 1")
        if self.design_x_max <= self.design_x_min:
            raise ValueError("design bounds have non-positive x area")
        if self.design_y_max <= self.design_y_min:
            raise ValueError("design bounds have non-positive y area")


def load_dataset(dataset_dir: str | Path) -> tuple[list[TopologySpec], dict]:
    """Discover topology directories and load masks."""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "dataset.json") as f:
        meta = json.load(f)

    bounds = meta["design_bounds"]
    specs = []
    for topo_dir in sorted(dataset_dir.iterdir()):
        if not topo_dir.is_dir() or not topo_dir.name.startswith("topology_"):
            continue
        mask_path = topo_dir / "mask.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path)
        topo_json_path = topo_dir / "topology.json"
        if topo_json_path.exists():
            with open(topo_json_path) as f:
                tj = json.load(f)
            tid = tj.get("topology_id", topo_dir.name)
        else:
            tid = topo_dir.name
        specs.append(TopologySpec(
            topology_id=tid,
            mask=mask,
            design_x_min=bounds[0], design_x_max=bounds[1],
            design_y_min=bounds[2], design_y_max=bounds[3],
        ))
    return specs, meta
