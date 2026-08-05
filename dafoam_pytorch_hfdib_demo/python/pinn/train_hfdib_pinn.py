#!/usr/bin/env python3
"""Train the HFDIB PINN on single_obstacle.

Phases:
  resolve configuration
  create isolated work case
  load exact warm state (no converged fallback)
  load matching geometry manifest + mesh metadata
  build features + model + flux + state assembler
  verify W(theta_0) == W_k
  compute fixed initial loss weights
  run parameter-gradient gate (hard block)
  train only after gate passes
  evaluate final residual
  write checkpoint and provenance

Usage:
  python -m pinn.train_hfdib_pinn --architecture mlp --k 8 --gradient-check-only
  python -m pinn.train_hfdib_pinn --architecture cnn --k 8 --steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from common import hfdib_options, write_json, PROJECT_ROOT  # noqa: E402
from dafoam_bridge import DAFoamResidualBridge  # noqa: E402
from dafoam_residual_function import dafoam_residual  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from pinn.models import CoordinateMLP, CompactDilatedCNN  # noqa: E402
from pinn.mesh_metadata import build_from_polymesh  # noqa: E402
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.state_assembly import StateAssembler  # noqa: E402
from pinn.losses import (  # noqa: E402
    ResidualLossConfig, weighted_residual_loss_torch,
    weighted_residual_loss_numpy, compute_initial_weights,
)
from pinn.gradient_check import gradient_check  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture", default="mlp", choices=["mlp", "cnn"])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gradient-check-only", action="store_true")
    ap.add_argument("--allow-unverified-gradient", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)
    run_id = f"{args.architecture}_k{args.k}_{int(time.time())}"
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "hfdib_pinn",
                           args.architecture, f"k{args.k}", run_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- create isolated work case ---------------------------------------
    case_src = os.path.join(PROJECT_ROOT, "cases", "single_obstacle")
    work_case = os.path.join(out_dir, "case")
    shutil.copytree(case_src, work_case, dirs_exist_ok=True)
    # clean generated dirs (keep 0/ fields!)
    for d in os.listdir(work_case):
        if d[0].isdigit() and d != "0":
            shutil.rmtree(os.path.join(work_case, d), ignore_errors=True)
    shutil.rmtree(os.path.join(work_case, "constant", "polyMesh"), ignore_errors=True)
    shutil.rmtree(os.path.join(work_case, "postProcessing"), ignore_errors=True)

    # regenerate mesh
    import subprocess
    subprocess.run(["blockMesh", "-case", work_case],
                   check=True, capture_output=True,
                   env={**os.environ, "FOAM_SETTINGS": ""} if "FOAM_SETTINGS" not in os.environ else os.environ)

    # ---- load exact warm state (no fallback) -----------------------------
    warm_path = os.path.join(PROJECT_ROOT, "outputs", "hfdib_gate_g",
                             "smoke" if args.k in [8] else "full",
                             "partial_primal", f"k{args.k}", "W_k.npy")
    if not os.path.isfile(warm_path):
        # try the other mode dir
        for mode in ["smoke", "full"]:
            p = os.path.join(PROJECT_ROOT, "outputs", "hfdib_gate_g", mode,
                            "partial_primal", f"k{args.k}", "W_k.npy")
            if os.path.isfile(p):
                warm_path = p
                break
    if not os.path.isfile(warm_path):
        raise FileNotFoundError(
            f"Requested exact warm state k={args.k} is missing: {warm_path}")
    w_k = np.load(warm_path)
    print(f"[pinn] loaded warm state k={args.k}: {warm_path}")

    # ---- load mesh metadata + manifest + layout ---------------------------
    # trigger manifest generation by evaluating one residual
    os.chdir(work_case)
    bridge = DAFoamResidualBridge(work_case, hfdib_options(work_case))
    bridge.residual(w_k)  # triggers calcFvSource -> manifest export

    mesh_meta = build_from_polymesh(work_case)
    mesh_meta.save(os.path.join(out_dir, "mesh_metadata.npz"),
                   os.path.join(out_dir, "mesh_metadata.json"))

    layout = build_state_layout("isothermal")
    u_ids = layout.indices("U")
    p_ids = layout.indices("p")
    phi_ids = layout.indices("phi")
    u_ids_t = torch.from_numpy(u_ids)
    p_ids_t = torch.from_numpy(p_ids)
    phi_ids_t = torch.from_numpy(phi_ids)

    # load geometry manifest from the work case (now generated by bridge.residual)
    from hfdib.geometry_manifest import load as load_manifest
    manifest = load_manifest(work_case)
    print(f"[pinn] manifest: {manifest.n_fluid} fluid, "
          f"{manifest.n_solid} solid, {manifest.n_interface} interface")

    # ---- build features --------------------------------------------------
    n_cells = mesh_meta.n_cells
    nx, ny = 40, 16

    if args.architecture == "mlp":
        feat = np.zeros((n_cells, 4))
        for c in manifest.cells:
            cid = c["cell_id"]
            feat[cid, 0] = 2 * c["cx"] / 1.0 - 1
            feat[cid, 1] = 2 * c["cy"] / 0.1 - 1
            feat[cid, 2] = c["lambda"]
            h_eff = mesh_meta.cell_volumes[cid] ** (1/3)
            feat[cid, 3] = c["sigma"] / h_eff
        features = torch.from_numpy(feat)
        model = CoordinateMLP(input_size=4, width=32, hidden_layers=2)
    else:
        feat = np.zeros((6, ny, nx))
        for c in manifest.cells:
            cid = c["cell_id"]
            j, i = mesh_meta.cell_to_grid[cid]
            feat[0, j, i] = 2 * c["cx"] / 1.0 - 1
            feat[1, j, i] = 2 * c["cy"] / 0.1 - 1
            feat[2, j, i] = c["lambda"]
            h_eff = mesh_meta.cell_volumes[cid] ** (1/3)
            feat[3, j, i] = c["sigma"] / h_eff
            feat[4, j, i] = 1.0 if c["chi"] == 0.0 else 0.0
            feat[5, j, i] = 1.0 if c["chi"] > 0.0 else 0.0
        features = torch.from_numpy(feat)
        model = CompactDilatedCNN(in_channels=6, width=24, dilations=(1, 2, 4, 8))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pinn] model: {args.architecture}, {n_params} parameters")

    # ---- build flux assembler + state assembler ---------------------------
    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=n_cells,
        n_faces=mesh_meta.n_faces,
    )

    base_state = torch.from_numpy(w_k)
    state_asm = StateAssembler(base_state, layout, flux_asm,
                              n_cells, mesh_meta.n_faces)

    # ---- verify W(theta_0) == W_k ----------------------------------------
    with torch.no_grad():
        cell_corr = model(features)
        state_0 = state_asm.assemble(cell_corr)
    w_diff = (state_0 - base_state).abs().max().item()
    print(f"[pinn] W(theta_0) - W_k max diff: {w_diff:.2e}")
    assert w_diff < 1e-12, "Zero-init failed: W(theta_0) != W_k"

    # ---- create bridge (already created above for manifest) --------------

    # ---- compute initial loss + weights -----------------------------------
    config = compute_initial_weights(state_0, bridge, u_ids, p_ids, phi_ids)
    print(f"[pinn] initial weights: gu={config.gamma_u:.2e} "
          f"gp={config.gamma_p:.2e} gphi={config.gamma_phi:.2e}")

    # evaluate initial objective
    with torch.no_grad():
        r_init = bridge.residual(state_0.detach().numpy())
        loss_init = weighted_residual_loss_numpy(
            r_init, u_ids, p_ids, phi_ids, config)
    print(f"[pinn] initial objective: {loss_init:.6e}")

    # ---- gradient gate ---------------------------------------------------
    print("[pinn] running parameter gradient check (Check A: output layer)...",
          flush=True)
    gc_a = gradient_check(
        model, features, bridge, state_asm,
        u_ids_t, p_ids_t, phi_ids_t, config,
        n_directions=10, output_layer_only=True)

    print("[pinn] running parameter gradient check (Check B: probe model)...",
          flush=True)
    # clone model and perturb final layer
    import copy
    probe_model = copy.deepcopy(model)
    last_param = list(probe_model.parameters())[-1]
    with torch.no_grad():
        last_param.add_(torch.randn_like(last_param) * 1e-4)
    gc_b = gradient_check(
        probe_model, features, bridge, state_asm,
        u_ids_t, p_ids_t, phi_ids_t, config,
        n_directions=10, output_layer_only=False)

    write_json(os.path.join(out_dir, "gradient_check.json"), {
        "check_a_output_layer": gc_a,
        "check_b_probe_model": gc_b,
    })

    gc_pass = gc_a["pass"] and gc_b["pass"]
    print(f"[pinn] gradient check A: median={gc_a['median_rel']:.2e} "
          f"max={gc_a['max_rel']:.2e} pass={gc_a['pass']}")
    print(f"[pinn] gradient check B: median={gc_b['median_rel']:.2e} "
          f"max={gc_b['max_rel']:.2e} pass={gc_b['pass']}")

    if not gc_pass and not args.allow_unverified_gradient:
        raise RuntimeError(
            "Parameter gradient verification failed; training is blocked. "
            "Use --allow-unverified-gradient to override (dangerous).")

    if args.gradient_check_only:
        print("[pinn] gradient-check-only mode: done")
        return 0 if gc_pass else 1

    # ---- train -----------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    t0 = time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad()
        cell_corr = model(features)
        state = state_asm.assemble(cell_corr)
        residual = dafoam_residual(state, bridge)
        loss, loss_info = weighted_residual_loss_torch(
            residual, u_ids_t, p_ids_t, phi_ids_t, config)

        if not torch.isfinite(loss):
            print(f"[pinn] NaN at step {step}, stopping")
            break

        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            gnorm = sum(p.grad.norm().item() for p in model.parameters()
                        if p.grad is not None)
            entry = {
                "step": step,
                "total": loss_info["total"].item(),
                "momentum": loss_info["momentum"].item(),
                "pressure": loss_info["pressure"].item(),
                "flux": loss_info["flux"].item(),
                "grad_norm": gnorm,
                "elapsed": round(time.perf_counter() - t0, 1),
            }
            history.append(entry)
            print(f"[pinn] step {step:4d} loss={entry['total']:.6e} "
                  f"U={entry['momentum']:.2e} p={entry['pressure']:.2e} "
                  f"phi={entry['flux']:.2e} g={gnorm:.2e}", flush=True)

    # ---- evaluate final objective ----------------------------------------
    with torch.no_grad():
        final_cell_corr = model(features)
        final_state = state_asm.assemble(final_cell_corr)
        r_final = bridge.residual(final_state.detach().numpy())
        loss_final = weighted_residual_loss_numpy(
            r_final, u_ids, p_ids, phi_ids, config)

    ratio = loss_init / max(loss_final, 1e-30)
    print(f"[pinn] objective: {loss_init:.4e} -> {loss_final:.4e} ({ratio:.1f}x)")

    # ---- save outputs ----------------------------------------------------
    write_json(os.path.join(out_dir, "history.json"), history)
    write_json(os.path.join(out_dir, "config.json"), {
        "architecture": args.architecture, "k": args.k,
        "steps": args.steps, "lr": args.lr, "seed": args.seed,
        "n_params": n_params,
        "gradient_check_pass": gc_pass,
        "uses_flow_labels": False,
    })
    write_json(os.path.join(out_dir, "final_metrics.json"), {
        "initial_objective": loss_init,
        "final_objective": loss_final,
        "reduction_factor": ratio,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    })
    np.save(os.path.join(out_dir, "initial_state.npy"), w_k)
    np.save(os.path.join(out_dir, "final_state.npy"),
            final_state.detach().numpy())
    np.save(os.path.join(out_dir, "initial_residual.npy"), r_init)
    np.save(os.path.join(out_dir, "final_residual.npy"), r_final)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(out_dir, "checkpoint.pt"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
