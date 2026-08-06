"""Evaluate DAFoam HFDIB residual at a given state (subprocess worker)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_signed_distance_options  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from mpi4py import MPI  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--topology-id", default="unknown")
    args = ap.parse_args()

    os.chdir(args.case)
    bridge = DAFoamResidualBridge(args.case,
                                  hfdib_signed_distance_options(args.case),
                                  comm=MPI.COMM_SELF)

    state = np.load(args.state)
    residual = bridge.residual(state)

    layout = build_state_layout("isothermal")
    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")

    # Use unit weights for evaluation
    lu = 0.5 * float(np.dot(residual[u_ids], residual[u_ids]))
    lp = 0.5 * float(np.dot(residual[p_ids], residual[p_ids]))
    lphi = 0.5 * float(np.dot(residual[phi_ids], residual[phi_ids]))
    loss = lu + lp + lphi

    # Compute gradient (seed = R per block)
    seed = np.zeros_like(residual)
    seed[u_ids] = residual[u_ids]
    seed[p_ids] = residual[p_ids]
    seed[phi_ids] = residual[phi_ids]
    grad_state = bridge.residual_jacobian_transpose_vector(state, seed)

    # Save grad
    grad_path = f"/tmp/grad_{args.topology_id}.npy"
    np.save(grad_path, grad_state)

    # Output result as JSON to stdout
    print(json.dumps({
        "topology_id": args.topology_id,
        "loss": loss,
        "loss_u": lu,
        "loss_p": lp,
        "loss_phi": lphi,
        "residual_norm": float(np.linalg.norm(residual)),
    }))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
