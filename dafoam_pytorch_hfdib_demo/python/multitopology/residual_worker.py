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
                parent_pipe):
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

        os.chdir(case_dir)
        bridge = DAFoamResidualBridge(
            case_dir,
            hfdib_signed_distance_options(case_dir),
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
        elif msg["command"] == "evaluate":
            try:
                state = msg["state"]
                request_id = msg["request_id"]

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

                parent_pipe.send({
                    "status": "ok",
                    "request_id": request_id,
                    "topology_id": topology_id,
                    "loss": loss,
                    "loss_u": loss_u,
                    "loss_p": loss_p,
                    "loss_phi": loss_phi,
                    "residual_norm": float(np.linalg.norm(residual)),
                    "grad_state": grad_state,
                })
            except Exception as e:
                parent_pipe.send({
                    "status": "error",
                    "request_id": msg.get("request_id", -1),
                    "topology_id": topology_id,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
