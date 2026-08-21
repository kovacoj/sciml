"""Run one residual-controlled CFD continuation for the final warm-start study."""
from __future__ import annotations

import argparse
import json
import os
import shutil
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


METHODS = ("cold", "neural", "neural_1", "neural_5", "teacher_20")


def add_residual_control(path: Path, tolerance: float) -> None:
    text = path.read_text()
    marker = "    nNonOrthogonalCorrectors 0;"
    if marker not in text:
        raise RuntimeError(f"Cannot locate SIMPLE block in {path}")
    block = (
        f"{marker}\n"
        "    residualControl\n"
        "    {\n"
        f"        p {tolerance:.16g};\n"
        "    }"
    )
    path.write_text(text.replace(marker, block, 1))


def network_state(checkpoint: dict, topology: Path, dataset: Path,
                  mesh: MeshMetadata, w0: np.ndarray) -> tuple[np.ndarray, float]:
    model = build_model(checkpoint["architecture"], **checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    lambda_tensor = torch.from_numpy(np.load(topology / "lambda.npy")).unsqueeze(0).unsqueeze(0)
    phi_indices = np.asarray(checkpoint["phi_trainable_indices"], dtype=np.int64)
    flux_assembler = FluxAssembler(
        owners=mesh.owners, neighbours=mesh.neighbours,
        sf_vec=mesh.face_area_vectors, owner_weights=mesh.owner_weights,
        n_cells=mesh.n_cells, n_faces=mesh.n_faces,
    )
    assembler = IndependentPhiStateAssembler(
        torch.from_numpy(w0), build_isothermal_layout(mesh.n_cells, mesh.n_faces),
        flux_assembler, mesh.n_cells, mesh.n_faces, phi_indices,
    )
    started = time.perf_counter()
    with torch.no_grad():
        cell, phi = model(lambda_tensor)
        cell = project_solid_velocity(cell, lambda_tensor)
        correction = cell.squeeze(0).permute(1, 2, 0).reshape(-1, 3)
        state = assembler.assemble(correction, phi.squeeze(0)).numpy()
    return state, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-state", type=Path, required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residual-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    project = Path(PROJECT_ROOT)
    absolute = lambda path: path if path.is_absolute() else project / path
    dataset = absolute(args.dataset)
    checkpoint_path = absolute(args.checkpoint)
    teacher_state_path = absolute(args.teacher_state)
    work_dir = absolute(args.work_dir)
    output = absolute(args.output)
    source_case = dataset / args.topology_id / "case"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    for name in ("0", "constant", "system"):
        shutil.copytree(source_case / name, work_dir / name)
    add_residual_control(work_dir / "system/fvSolution", args.residual_tolerance)

    torch.set_default_dtype(torch.float64)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mesh = MeshMetadata.load(str(dataset / "shared/mesh_metadata.npz"),
                             str(dataset / "shared/mesh_metadata.json"))
    w0 = np.load(dataset / "shared/base_state_k0.npy")
    inference_time = 0.0
    pre_simple_time = 0.0
    if args.method == "cold":
        state = w0.copy()
    elif args.method == "teacher_20":
        state = np.load(teacher_state_path)
    else:
        state, inference_time = network_state(
            checkpoint, dataset / args.topology_id, dataset, mesh, w0
        )

    os.chdir(work_dir)
    from mpi4py import MPI
    options = hfdib_signed_distance_options(
        str(work_dir), inlet_patches=["inletLower", "inletUpper"],
        outlet_patches=["outletLower", "outletUpper"],
    )
    options["primalMinResTol"] = args.residual_tolerance
    options["primalMinIters"] = 2
    options["printInterval"] = 1
    bridge = DAFoamResidualBridge(str(work_dir), options, comm=MPI.COMM_SELF)
    n_pre_steps = {"neural_1": 1, "neural_5": 5}.get(args.method, 0)
    if n_pre_steps:
        started = time.perf_counter()
        for _ in range(n_pre_steps):
            state = bridge.simple_step(state)
        pre_simple_time = time.perf_counter() - started

    initial_residual = float(np.linalg.norm(bridge.residual(state)))
    bridge.set_state(state)
    started = time.perf_counter()
    bridge.solver()
    solver_time = time.perf_counter() - started
    final_state = np.ascontiguousarray(bridge.solver.getStates().copy(), dtype=np.float64)
    final_residual = float(np.linalg.norm(bridge.residual(final_state)))
    primal_failed = bool(bridge.solver.primalFail)
    final_state_path = output.with_suffix(".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(final_state_path, final_state)
    result = {
        "topology_id": args.topology_id,
        "method": args.method,
        "residual_control": {"p": args.residual_tolerance},
        "initial_residual_l2": initial_residual,
        "final_residual_l2": final_residual,
        "inference_time_s": inference_time,
        "pre_simple_steps": n_pre_steps,
        "pre_simple_time_s": pre_simple_time,
        "remaining_simple_time_s": solver_time,
        "total_time_s": inference_time + pre_simple_time + solver_time,
        "primal_failed": primal_failed,
        "final_state": str(final_state_path.relative_to(project)),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    return 1 if primal_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
