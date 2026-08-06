"""Prepared topology context: holds all data needed for training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pinn.losses import ResidualLossConfig
from pinn.state_assembly import StateAssembler


@dataclass
class PreparedTopology:
    topology_id: str
    case_dir: str
    features: torch.Tensor  # [C, H, W]
    warm_state: torch.Tensor  # [N_state]
    loss_config: ResidualLossConfig
    state_assembler: StateAssembler
    u_ids: np.ndarray
    p_ids: np.ndarray
    phi_ids: np.ndarray
    inlet_patches: list = None
    outlet_patches: list = None


def load_prepared_topology(topo_dir: str | Path,
                            layout,
                            mesh_meta,
                            flux_assembler) -> PreparedTopology:
    """Load a prepared topology from disk."""
    topo_dir = Path(topo_dir)
    tid = topo_dir.name

    features = np.load(topo_dir / "features.npy")
    warm_state = np.load(next(topo_dir.glob("warm_state_k*.npy")))

    import json
    with open(topo_dir / "loss_config.json") as f:
        lc = json.load(f)
    config = ResidualLossConfig(
        gamma_u=lc["gamma_u"], gamma_p=lc["gamma_p"], gamma_phi=lc["gamma_phi"])

    base_state = torch.from_numpy(warm_state)
    state_asm = StateAssembler(base_state, layout, flux_assembler,
                                mesh_meta.n_cells, mesh_meta.n_faces)

    return PreparedTopology(
        topology_id=tid,
        case_dir=str(topo_dir / "case"),
        features=torch.from_numpy(features),
        warm_state=base_state,
        loss_config=config,
        state_assembler=state_asm,
        u_ids=layout.indices("U"),
        p_ids=layout.indices("p"),
        phi_ids=layout.indices("phi"),
        inlet_patches=lc.get("inlet_patches", ["inlet"]),
        outlet_patches=lc.get("outlet_patches", ["outlet"]),
    )
