"""Evaluate teacher, imitation, and end-to-end errors on the frozen test set."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import PROJECT_ROOT, hfdib_signed_distance_options
from dafoam_bridge import DAFoamResidualBridge
from pinn.flux_assembly import FluxAssembler
from pinn.mesh_metadata import MeshMetadata
from pinn.state_assembly_independent_phi import IndependentPhiStateAssembler
from state_layout import build_isothermal_layout
from unet.factory import build_model
from unet.train import project_solid_velocity


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / (np.linalg.norm(reference) + 1e-30))


def field_blocks(state: np.ndarray, n_cells: int, phi_indices: np.ndarray):
    n_u = 3 * n_cells
    velocity = state[:n_u].reshape(n_cells, 3)[:, :2]
    pressure = state[n_u:n_u + n_cells]
    flux = state[n_u + n_cells:][phi_indices]
    return velocity, pressure, flux


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/four_port_64"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-set", type=Path,
                        default=Path("outputs/final_campaign/test_set.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/final_warmstart_study/error_decomposition.json"))
    parser.add_argument("--teacher-state-dir", type=Path,
                        default=Path("outputs/final_warmstart_study/teacher_states"))
    args = parser.parse_args()

    project = Path(PROJECT_ROOT)
    dataset = args.dataset if args.dataset.is_absolute() else project / args.dataset
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else project / args.checkpoint
    test_set_path = args.test_set if args.test_set.is_absolute() else project / args.test_set
    output = args.output if args.output.is_absolute() else project / args.output
    teacher_dir = (args.teacher_state_dir if args.teacher_state_dir.is_absolute()
                   else project / args.teacher_state_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float64)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint_data["architecture"], **checkpoint_data.get("model_kwargs", {}))
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    model.eval()

    mesh = MeshMetadata.load(str(dataset / "shared/mesh_metadata.npz"),
                             str(dataset / "shared/mesh_metadata.json"))
    layout = build_isothermal_layout(mesh.n_cells, mesh.n_faces)
    w0 = np.load(dataset / "shared/base_state_k0.npy")
    phi_indices = np.asarray(checkpoint_data["phi_trainable_indices"], dtype=np.int64)
    flux_assembler = FluxAssembler(
        owners=mesh.owners, neighbours=mesh.neighbours,
        sf_vec=mesh.face_area_vectors, owner_weights=mesh.owner_weights,
        n_cells=mesh.n_cells, n_faces=mesh.n_faces,
    )
    assembler = IndependentPhiStateAssembler(
        torch.from_numpy(w0), layout, flux_assembler, mesh.n_cells,
        mesh.n_faces, phi_indices,
    )
    test_ids = json.loads(test_set_path.read_text())["test_topology_ids"]

    cases = []
    for topology_id in test_ids:
        topology = dataset / topology_id
        case_dir = topology / "case"
        lambda_array = np.load(topology / "lambda.npy")
        lambda_tensor = torch.from_numpy(lambda_array).unsqueeze(0).unsqueeze(0)
        inference_start = time.perf_counter()
        with torch.no_grad():
            cell_prediction, phi_prediction = model(lambda_tensor)
            cell_prediction = project_solid_velocity(cell_prediction, lambda_tensor)
            correction = cell_prediction.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
            w_network = assembler.assemble(correction, phi_prediction.squeeze(0)).numpy()
        inference_time = time.perf_counter() - inference_start

        os.chdir(case_dir)
        from mpi4py import MPI
        bridge = DAFoamResidualBridge(
            str(case_dir),
            hfdib_signed_distance_options(
                str(case_dir), inlet_patches=["inletLower", "inletUpper"],
                outlet_patches=["outletLower", "outletUpper"],
            ), comm=MPI.COMM_SELF,
        )
        teacher_start = time.perf_counter()
        w_teacher = w0.copy()
        for _ in range(20):
            w_teacher = bridge.simple_step(w_teacher)
        teacher_time = time.perf_counter() - teacher_start
        np.save(teacher_dir / f"{topology_id}_k020.npy", w_teacher)
        del bridge

        reference_velocity = np.column_stack([
            np.load(topology / "ux_hfdib.npy").reshape(-1),
            np.load(topology / "uy_hfdib.npy").reshape(-1),
        ])
        reference_pressure = np.load(topology / "pressure_hfdib.npy").reshape(-1)
        u0, p0, _ = field_blocks(w0, mesh.n_cells, phi_indices)
        u_teacher, p_teacher, phi_teacher = field_blocks(w_teacher, mesh.n_cells, phi_indices)
        u_network, p_network, phi_network = field_blocks(w_network, mesh.n_cells, phi_indices)

        cases.append({
            "topology_id": topology_id,
            "teacher_time_s": teacher_time,
            "inference_time_s": inference_time,
            "teacher_to_cfd": {
                "rel_u": relative_error(u_teacher, reference_velocity),
                "rel_p": relative_error(p_teacher, reference_pressure),
            },
            "network_to_teacher": {
                "rel_u": relative_error(u_network, u_teacher),
                "rel_p": relative_error(p_network, p_teacher),
                "rel_phi": relative_error(phi_network, phi_teacher),
                "rel_delta_u": relative_error(u_network - u0, u_teacher - u0),
                "rel_delta_p": relative_error(p_network - p0, p_teacher - p0),
            },
            "network_to_cfd": {
                "rel_u": relative_error(u_network, reference_velocity),
                "rel_p": relative_error(p_network, reference_pressure),
            },
        })

    metric_paths = {
        "teacher_to_cfd": ("rel_u", "rel_p"),
        "network_to_teacher": ("rel_u", "rel_p", "rel_phi", "rel_delta_u", "rel_delta_p"),
        "network_to_cfd": ("rel_u", "rel_p"),
    }
    means = {
        group: {metric: float(np.mean([case[group][metric] for case in cases]))
                for metric in metrics}
        for group, metrics in metric_paths.items()
    }
    payload = {
        "checkpoint": str(checkpoint.relative_to(project)),
        "checkpoint_git_sha": checkpoint_data["git_sha"],
        "target_k": 20,
        "n_test": len(cases),
        "test_topology_ids": test_ids,
        "means": means,
        "cases": cases,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(means, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
