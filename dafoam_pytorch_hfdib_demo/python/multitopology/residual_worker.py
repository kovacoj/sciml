"""Isolated DAFoam residual worker process.

Each worker owns one topology, one case directory, one PYDAFOAM instance.
Communicates via Pipe with the parent trainer process.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np


def worker_main(case_dir: str, topology_id: str,
                u_ids, p_ids, phi_ids,
                gamma_u: float, gamma_p: float, gamma_phi: float,
                parent_pipe,
                inlet_patches=None, outlet_patches=None,
                result_queue=None):
    """Main loop for a DAFoam residual worker."""
    # Set thread limits before importing MPI/DAFoam
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    PYTHON_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PYTHON_ROOT))

    from mpi4py import MPI
    from dafoam import PYDAFOAM
    from common import hfdib_signed_distance_options
    from dafoam_bridge import DAFoamResidualBridge

    # Initialize bridge with MPI.COMM_SELF — wrap in try/except to prevent
    # the parent from blocking forever on child init failure
    try:
        from mpi4py import MPI
        from common import hfdib_signed_distance_options
        from dafoam_bridge import DAFoamResidualBridge

        kw = {}
        if inlet_patches is not None:
            kw["inlet_patches"] = inlet_patches
        if outlet_patches is not None:
            kw["outlet_patches"] = outlet_patches

        os.chdir(case_dir)
        bridge = DAFoamResidualBridge(
            case_dir,
            hfdib_signed_distance_options(case_dir, **kw),
            comm=MPI.COMM_SELF,
        )

        parent_pipe.send({"status": "ready", "topology_id": topology_id})
    except Exception as exc:
        parent_pipe.send({
            "status": "error",
            "topology_id": topology_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        return

    while True:
        try:
            msg = parent_pipe.recv()
        except (EOFError, KeyboardInterrupt):
            break

        if msg["command"] == "close":
            break
        elif msg["command"] == "ping":
            parent_pipe.send({"status": "pong", "topology_id": topology_id})
            continue
        elif msg["command"] == "simple_step":
            try:
                state_path = msg["state_path"]
                request_id = msg["request_id"]

                state = np.load(state_path)
                next_state = bridge.simple_step(state)

                next_path = state_path.replace(".npy", "_simple.npy")
                np.save(next_path, next_state)

                print(f"[worker {topology_id}] simple_step done", flush=True)
                result_queue.put({
                    "status": "ok",
                    "request_id": request_id,
                    "topology_id": topology_id,
                    "next_path": next_path,
                })
            except Exception as e:
                result_queue.put({
                    "status": "error",
                    "request_id": msg.get("request_id", -1),
                    "topology_id": topology_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
        elif msg["command"] == "evaluate":
            try:
                state_path = msg["state_path"]
                request_id = msg["request_id"]

                state = np.load(state_path)

                residual = bridge.residual(state)

                # Build seed: dL/dR = gamma * R per block
                seed = np.zeros_like(residual)
                seed[u_ids] = gamma_u * residual[u_ids]
                seed[p_ids] = gamma_p * residual[p_ids]
                seed[phi_ids] = gamma_phi * residual[phi_ids]

                grad_state = bridge.residual_jacobian_transpose_vector(
                    state, seed)

                # Compute losses
                loss_u = 0.5 * gamma_u * float(np.dot(residual[u_ids], residual[u_ids]))
                loss_p = 0.5 * gamma_p * float(np.dot(residual[p_ids], residual[p_ids]))
                loss_phi = 0.5 * gamma_phi * float(np.dot(residual[phi_ids], residual[phi_ids]))
                loss = loss_u + loss_p + loss_phi

                # Write grad_state to temp file to avoid pipe buffer deadlock
                import tempfile
                grad_path = state_path.replace(".npy", "_grad.npy")
                np.save(grad_path, grad_state)

                print(f"[worker {topology_id}] sending result "
                      f"(loss={loss:.4e})", flush=True)
                result_queue.put({
                    "status": "ok",
                    "request_id": request_id,
                    "topology_id": topology_id,
                    "loss": loss,
                    "loss_u": loss_u,
                    "loss_p": loss_p,
                    "loss_phi": loss_phi,
                    "residual_norm": float(np.linalg.norm(residual)),
                    "grad_path": grad_path,
                })
            except Exception as e:
                result_queue.put({
                    "status": "error",
                    "request_id": msg.get("request_id", -1),
                    "topology_id": topology_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
