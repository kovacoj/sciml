"""Create the exact seed-22 neural DAFoam initialization for one topology."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from diagnostics.warmstart_common import build_state_assembler, predict_full_state
from pinn.mesh_metadata import MeshMetadata
from state_layout import build_isothermal_layout
from unet.factory import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True); parser.add_argument("--dataset", required=True)
parser.add_argument("--topology-index", type=int, required=True); parser.add_argument("--output", required=True)
args = parser.parse_args(); dataset = Path(args.dataset)
payload = torch.load(args.model, map_location="cpu", weights_only=False); torch.set_default_dtype(torch.float64)
model = build_model(payload["architecture"], **payload.get("model_kwargs", {})); model.load_state_dict(payload["model_state_dict"]); model.eval()
mesh = MeshMetadata.load(str(dataset/"shared/mesh_metadata.npz"), str(dataset/"shared/mesh_metadata.json"))
base = np.load(dataset/"shared/base_state_k0.npy"); assembler, _ = build_state_assembler(mesh, build_isothermal_layout(mesh.n_cells, mesh.n_faces), base)
case = dataset/f"topology_{args.topology_index:04d}"; lam = np.load(case/"lambda.npy"); state, _ = predict_full_state(model, lam, assembler)
n_u = 3*mesh.n_cells; cells = state[:n_u].reshape(mesh.n_cells,3)
np.savez_compressed(args.output, **{"lambda": lam, "Ux": cells[:,0], "Uy": cells[:,1], "p": state[n_u:n_u+mesh.n_cells], "W_theta": state})
