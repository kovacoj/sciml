#!/usr/bin/env python3
"""Train the HFDIB PINN on single_obstacle.

Usage:
  python -m pinn.train_hfdib_pinn --architecture mlp --k 8 --steps 1000
  python -m pinn.train_hfdib_pinn --architecture cnn --k 8 --steps 1000

The network predicts corrections (dUx, dUy, dp) per cell.
phi is assembled differentiably from delta_U.
Loss is the DAFoam HFDIB residual (no labels).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from dafoam_residual_function import dafoam_residual  # noqa: E402
from pinn.models import CoordinateMLP, CompactDilatedCNN  # noqa: E402
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.state_assembly import StateAssembler  # noqa: E402
from pinn.losses import physics_loss, compute_initial_weights  # noqa: E402
from pinn.gradient_check import gradient_check  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "hfdib_pinn"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.out, exist_ok=True)

    case_dir = os.path.join(PROJECT_ROOT, "cases", "single_obstacle")
    layout = build_state_layout("isothermal")

    # load warm state
    warm_path = os.path.join(PROJECT_ROOT, "outputs", "hfdib_gate_g",
                             "partial_primal", f"k{args.k}", "W_k.npy")
    if not os.path.exists(warm_path):
        # fallback: use converged solve
        warm_path = os.path.join(PROJECT_ROOT, "outputs", "hfdib_gate_g",
                                 "solve_hfdib", "W.npy")
    w_k = np.load(warm_path)
    print(f"[pinn] loaded warm state: {warm_path} ||W||={np.linalg.norm(w_k):.3e}")

    # create bridge
    os.chdir(case_dir)
    bridge = DAFoamResidualBridge(case_dir, hfdib_options(case_dir))

    # load geometry manifest for features
    from hfdib.geometry_manifest import load as load_manifest
    manifest = load_manifest(case_dir)

    # build features
    u_idx = layout.indices("U")
    p_idx = layout.indices("p")
    phi_idx = layout.indices("phi")
    u_ids_t = torch.from_numpy(u_idx)
    p_ids_t = torch.from_numpy(p_idx)
    phi_ids_t = torch.from_numpy(phi_idx)

    n_cells = 640  # 40x16
    if args.architecture == "mlp":
        # features: (x_hat, y_hat, lambda, sigma/h) per cell
        feat = np.zeros((n_cells, 4))
        for c in manifest.cells:
            cid = c["cell_id"]
            feat[cid, 0] = 2 * c["cx"] / 1.0 - 1  # x in [-1, 1]
            feat[cid, 1] = 2 * c["cy"] / 0.1 - 1    # y in [-1, 1]
            feat[cid, 2] = c["lambda"]
            feat[cid, 3] = c["sigma"] / 0.00625     # sigma/h
        features = torch.from_numpy(feat)
        model = CoordinateMLP(input_size=4, width=32, hidden_layers=2)
    else:
        # CNN: [C, H, W] = [6, 16, 40]
        # need cell_id -> (i, j) mapping for 40x16 blockMesh grid
        # blockMesh cell ordering: i + j*40 (x fastest)
        feat = np.zeros((6, 16, 40))
        for c in manifest.cells:
            cid = c["cell_id"]
            j = cid // 40
            i = cid % 40
            feat[0, j, i] = 2 * c["cx"] / 1.0 - 1
            feat[1, j, i] = 2 * c["cy"] / 0.1 - 1
            feat[2, j, i] = c["lambda"]
            feat[3, j, i] = c["sigma"] / 0.00625
            feat[4, j, i] = 1.0 if c["chi"] == 0.0 else 0.0  # fluid mask
            feat[5, j, i] = 1.0 if c["chi"] > 0.0 else 0.0   # solid mask
        features = torch.from_numpy(feat)
        model = CompactDilatedCNN(in_channels=6, width=24, dilations=(1, 2, 4, 8))

    # build flux assembler from mesh data
    from FvMeshDataBuilder import FvMeshDataBuilder
    fv_data = FvMeshDataBuilder.build(mesh._mesh) if hasattr(mesh, '_mesh') else None
    # simpler: extract from bridge
    owners = np.array(bridge.solver.solver.mesh().owner(), dtype=np.int64)
    neighbours = np.array(bridge.solver.solver.mesh().neighbour(), dtype=np.int64)
    sf = np.array([[s[0], s[1]] for s in bridge.solver.solver.mesh().Sf()])
    flux_asm = FluxAssembler(owners, neighbours, sf, n_cells)

    # state assembler
    base_state = torch.from_numpy(w_k)
    state_asm = StateAssembler(base_state, flux_asm,
                               n_u=1920, n_p=640, n_phi=2616)

    # initial loss + weights
    with torch.no_grad():
        cell_corr = model(features)
        state = state_asm.assemble(cell_corr)
    gu, gp, gphi = compute_initial_weights(state, bridge, u_ids_t, p_ids_t, phi_ids_t)
    print(f"[pinn] initial weights: gu={gu:.2e} gp={gp:.2e} gphi={gphi:.2e}")

    # gradient check before training
    print("[pinn] running parameter gradient check...", flush=True)
    gc = gradient_check(model, features, bridge, state_asm,
                        u_ids_t, p_ids_t, phi_ids_t, gu, gp, gphi,
                        n_params=20)
    print(f"[pinn] gradient check: median={gc['median_rel']:.2e} "
          f"max={gc['max_rel']:.2e} pass={gc['pass']}")

    if not gc["pass"]:
        print("[pinn] WARNING: gradient check failed — training may be unreliable")

    # train
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad()
        cell_corr = model(features)
        state = state_asm.assemble(cell_corr)
        loss, loss_info = physics_loss(state, bridge, u_ids_t, p_ids_t,
                                       phi_ids_t, gu, gp, gphi)

        if not torch.isfinite(loss):
            print(f"[pinn] NaN at step {step}, stopping")
            break

        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            gnorm = sum(p.grad.norm().item() for p in model.parameters()
                        if p.grad is not None)
            entry = {"step": step, **loss_info, "grad_norm": gnorm,
                    "elapsed": round(time.perf_counter() - t0, 1)}
            history.append(entry)
            print(f"[pinn] step {step:4d} loss={loss.item():.6e} "
                  f"U={loss_info['momentum']:.2e} p={loss_info['pressure']:.2e} "
                  f"phi={loss_info['flux']:.2e} g={gnorm:.2e}", flush=True)

    # save outputs
    write_json(os.path.join(args.out, "gradient_check.json"), gc)
    write_json(os.path.join(args.out, "history.json"), history)
    write_json(os.path.join(args.out, "config.json"), {
        "architecture": args.architecture,
        "k": args.k, "steps": args.steps, "lr": args.lr,
        "seed": args.seed,
        "gu": gu, "gp": gp, "gphi": gphi,
        "n_params": sum(p.numel() for p in model.parameters()),
        "gradient_check_pass": gc["pass"],
        "uses_flow_labels": False,
    })

    # save final fields
    with torch.no_grad():
        final_state = state_asm.assemble(model(features)).numpy()
    np.save(os.path.join(args.out, "final_state.npy"), final_state)

    initial_loss = history[0]["total"] if history else 0
    final_loss = history[-1]["total"] if history else 0
    ratio = initial_loss / max(final_loss, 1e-30)
    print(f"[pinn] done: {initial_loss:.4e} -> {final_loss:.4e} ({ratio:.1f}x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
