#!/usr/bin/env python3
"""Train one shared CompactDilatedCNN across multiple HFDIB topologies.

Usage:
  python -m multitopology.train_shared_network \
    --dataset prepared_topologies/four_port_seed \
    --workers 4 --steps-per-group 20 --epochs 20 --lr 1e-3 --device cpu
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

from common import write_json, PROJECT_ROOT  # noqa: E402
from state_layout import build_state_layout  # noqa: E402
from pinn.models import CompactDilatedCNN  # noqa: E402
from pinn.mesh_metadata import build_from_polymesh as build_mesh_meta  # noqa: E402
from pinn.flux_assembly import FluxAssembler  # noqa: E402
from pinn.losses import ResidualLossConfig  # noqa: E402
from multitopology.context import PreparedTopology, load_prepared_topology  # noqa: E402
from multitopology.worker_pool import TopologyWorkerPool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps-per-group", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    if args.device != "cpu":
        raise ValueError(
            "Local multi-topology HFDIB training currently supports --device cpu only."
        )

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float64)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = Path(PROJECT_ROOT) / dataset_dir

    output_dir = Path(args.output) if args.output else Path(PROJECT_ROOT) / "outputs" / "shared_hfdib"
    run_id = f"run_{int(time.time())}"
    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load shared mesh metadata
    from pinn.mesh_metadata import MeshMetadata
    mesh_meta = MeshMetadata.load(
        str(dataset_dir / "shared_mesh" / "mesh_metadata.npz"),
        str(dataset_dir / "shared_mesh" / "mesh_metadata.json"),
    )

    layout = build_state_layout("isothermal")

    # Build flux assembler (shared)
    flux_asm = FluxAssembler(
        owners=mesh_meta.owners,
        neighbours=mesh_meta.neighbours,
        sf_vec=mesh_meta.face_area_vectors,
        owner_weights=mesh_meta.owner_weights,
        n_cells=mesh_meta.n_cells,
        n_faces=mesh_meta.n_faces,
    )

    # Load all prepared topologies
    topo_dirs = sorted([d for d in dataset_dir.iterdir()
                       if d.is_dir() and d.name.startswith("topology_")])
    print(f"[train] {len(topo_dirs)} topologies loaded")

    prepared = []
    for td in topo_dirs:
        pt = load_prepared_topology(td, layout, mesh_meta, flux_asm)
        prepared.append(pt)

    n_channels = prepared[0].features.shape[0]
    n_topologies = len(prepared)
    worker_count = min(args.workers, n_topologies)

    # Create shared model
    model = CompactDilatedCNN(in_channels=n_channels, width=24, dilations=(1, 2, 4, 8))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: {n_params} parameters, {n_channels} input channels")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    start_epoch = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]
        print(f"[train] resumed from epoch {start_epoch}, step {global_step}")

    # Write config
    config = {
        "n_topologies": n_topologies,
        "n_params": n_params,
        "n_channels": n_channels,
        "worker_count": worker_count,
        "steps_per_group": args.steps_per_group,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "uses_flow_labels": False,
    }
    write_json(out_dir / "config.json", config)

    history = []
    per_topo_history = []
    t0 = time.perf_counter()

    # Training loop
    pool = None
    for epoch in range(start_epoch, args.epochs):
        # Shuffle topology order
        rng = np.random.default_rng(args.seed + epoch)
        topo_order = list(range(n_topologies))
        rng.shuffle(topo_order)

        # Partition into groups
        groups = [topo_order[i:i + worker_count]
                  for i in range(0, n_topologies, worker_count)]

        for group_idx, group in enumerate(groups):
            active = [prepared[i] for i in group]
            active_ids = [t.topology_id for t in active]

            # Start workers for this group
            if pool is not None:
                pool.close()
            pool = TopologyWorkerPool(prepared)
            pool.start(active_ids)

            for step in range(args.steps_per_group):
                optimizer.zero_grad()

                states = []
                for topo in active:
                    corr = model(topo.features)
                    state = topo.state_assembler.assemble(corr)
                    states.append(state)

                # Send to workers
                results = pool.evaluate(
                    active_ids,
                    [s.detach().cpu().numpy().copy() for s in states],
                )

                # Accumulate gradients
                grad_states = []
                for state, result in zip(states, results):
                    gs = torch.from_numpy(result["grad_state"]).to(
                        dtype=state.dtype, device=state.device)
                    grad_states.append(gs / len(states))

                torch.autograd.backward(states, grad_states)
                optimizer.step()

                global_step += 1

                # Log
                mean_loss = np.mean([r["loss"] for r in results])
                mean_loss_u = np.mean([r["loss_u"] for r in results])
                mean_loss_p = np.mean([r["loss_p"] for r in results])
                mean_loss_phi = np.mean([r["loss_phi"] for r in results])
                gnorm = sum(p.grad.norm().item() for p in model.parameters()
                            if p.grad is not None)
                elapsed = time.perf_counter() - t0

                hist_entry = {
                    "epoch": epoch,
                    "group_index": group_idx,
                    "global_step": global_step,
                    "active_topology_ids": active_ids,
                    "mean_loss": mean_loss,
                    "mean_loss_u": mean_loss_u,
                    "mean_loss_p": mean_loss_p,
                    "mean_loss_phi": mean_loss_phi,
                    "min_topology_loss": min(r["loss"] for r in results),
                    "max_topology_loss": max(r["loss"] for r in results),
                    "gradient_norm": gnorm,
                    "elapsed_seconds": round(elapsed, 1),
                }
                history.append(hist_entry)

                for topo, result in zip(active, results):
                    per_topo_history.append({
                        "topology_id": topo.topology_id,
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": result["loss"],
                        "loss_u": result["loss_u"],
                        "loss_p": result["loss_p"],
                        "loss_phi": result["loss_phi"],
                        "residual_norm": result["residual_norm"],
                    })

                if global_step % 10 == 0 or step == args.steps_per_group - 1:
                    print(f"[train] e{epoch} g{group_idx} s{global_step} "
                          f"loss={mean_loss:.4e} U={mean_loss_u:.2e} "
                          f"p={mean_loss_p:.2e} phi={mean_loss_phi:.2e} "
                          f"g={gnorm:.2e} t={elapsed:.0f}s", flush=True)

            # Save checkpoint after each group
            ckpt_path = out_dir / "checkpoint_latest.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "group_index": group_idx,
                "global_step": global_step,
                "config": config,
            }, ckpt_path)

    if pool is not None:
        pool.close()

    # Save final outputs
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "global_step": global_step,
        "config": config,
    }, out_dir / "checkpoint_final.pt")

    write_json(out_dir / "history.jsonl", history)
    write_json(out_dir / "per_topology_history.jsonl", per_topo_history)
    write_json(out_dir / "topology_ids.json",
                [t.topology_id for t in prepared])

    # Save final states
    final_states_dir = out_dir / "final_states"
    final_states_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for topo in prepared:
            corr = model(topo.features)
            state = topo.state_assembler.assemble(corr)
            np.save(final_states_dir / f"{topo.topology_id}.npy",
                    state.detach().numpy())

    # Final metrics
    initial_losses = history[:n_topologies] if len(history) >= n_topologies else history
    final_losses = history[-n_topologies:] if len(history) >= n_topologies else history
    write_json(out_dir / "final_metrics.json", {
        "initial_mean_loss": np.mean([h["mean_loss"] for h in initial_losses]),
        "final_mean_loss": np.mean([h["mean_loss"] for h in final_losses]),
        "total_steps": global_step,
        "wall_seconds": round(time.perf_counter() - t0, 1),
        "uses_flow_labels": False,
    })

    print(f"[train] done: {global_step} steps, "
          f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
