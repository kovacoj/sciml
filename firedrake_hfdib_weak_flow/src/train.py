"""Robust serial training CLI for the Firedrake HFDIB experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import resource
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


HISTORY_COLUMNS = [
    "global_step", "stage", "stage_step", "beta", "learning_rate", "loss_total",
    "loss_momentum", "loss_continuity", "loss_x_raw", "loss_y_raw",
    "loss_continuity_raw", "residual_x_l2", "residual_y_l2",
    "residual_continuity_l2", "divergence_l2", "inlet_flux", "outlet_flux",
    "mass_imbalance", "max_speed", "mean_speed", "pressure_min", "pressure_max",
    "gradient_norm", "seconds_per_step", "elapsed_seconds", "rss_mb",
]

_STOP_REQUESTED = False


def continuation_position(continuation: list[dict], stage: int, stage_step: int):
    """Normalize completed-stage counters to the next uncompleted optimizer step."""
    while stage < len(continuation) and stage_step >= continuation[stage]["steps"]:
        stage += 1
        stage_step = 0
    return stage, stage_step


def resolve_path(value: str | Path, config_path: Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        project_root / path,
        config_path.parent / path,
        project_root.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def checkpoint_payload(
    *, model, optimizer, config: dict, global_step: int, stage_index: int,
    stage_step: int, beta: float, Cm: float, Cc: float, elapsed_seconds: float,
    best_beta1_loss: float, git_sha: str,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "global_step": int(global_step),
        "stage_index": int(stage_index),
        "stage_step": int(stage_step),
        "beta": float(beta),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "git_sha": git_sha,
        "Cm": float(Cm),
        "Cc": float(Cc),
        "elapsed_seconds": float(elapsed_seconds),
        "best_beta1_loss": float(best_beta1_loss),
    }


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def restore_rng(checkpoint: dict) -> None:
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])


def get_git_sha(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def relative_mass_imbalance(inlet_flux: float, outlet_flux: float) -> float:
    return abs(inlet_flux + outlet_flux) / (
        abs(inlet_flux) + abs(outlet_flux) + 1.0e-12
    )


def heartbeat_payload(
    *, status: str, global_step: int, stage: int, beta: float, loss: float,
    best_beta1_loss: float, seconds_per_step: float,
    estimated_remaining_minutes: float, elapsed_seconds: float,
) -> dict[str, float | int | str]:
    return {
        "status": status,
        "global_step": int(global_step),
        "stage": int(stage),
        "beta": float(beta),
        "loss": float(loss),
        "best_beta1_loss": float(best_beta1_loss),
        "seconds_per_step": float(seconds_per_step),
        "estimated_remaining_minutes": float(estimated_remaining_minutes),
        "elapsed_minutes": float(elapsed_seconds / 60.0),
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def _request_stop(signum, frame) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _field_arrays(fields) -> tuple[np.ndarray, ...]:
    return tuple(
        value.detach().cpu().numpy()
        for value in (fields.ux, fields.uy, fields.p, fields.uibx, fields.uiby)
    )


def _diagnostics(context, ux: np.ndarray, uy: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from firedrake import FacetNormal, as_vector, assemble, div, dot

    u = as_vector((context.ux, context.uy))
    normal = FacetNormal(context.mesh)
    inlet_flux = float(assemble(
        dot(u, normal) * context.boundary_measure(context.inlet_marker)
    ))
    outlet_flux = float(assemble(
        dot(u, normal) * context.boundary_measure(context.outlet_marker)
    ))
    speed = np.hypot(ux, uy)
    return {
        "divergence_l2": float(assemble(div(u) ** 2 * context.measure) ** 0.5),
        "inlet_flux": inlet_flux,
        "outlet_flux": outlet_flux,
        "mass_imbalance": relative_mass_imbalance(inlet_flux, outlet_flux),
        "max_speed": float(speed.max()),
        "mean_speed": float(speed.mean()),
        "pressure_min": float(p.min()),
        "pressure_max": float(p.max()),
    }


def _save_snapshot(context, geometry, fields, beta: float, output_dir: Path, step: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from firedrake import as_vector, div, project

    xx, yy = np.meshgrid(geometry.x, geometry.y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    ux = np.asarray(context.ux.at(points), dtype=np.float64).reshape(xx.shape)
    uy = np.asarray(context.uy.at(points), dtype=np.float64).reshape(xx.shape)
    pressure = np.asarray(context.p.at(points), dtype=np.float64).reshape(xx.shape)
    divergence_function = project(div(as_vector((context.ux, context.uy))), context.Q)
    divergence = np.asarray(
        divergence_function.at(points), dtype=np.float64
    ).reshape(xx.shape)
    speed = np.hypot(ux, uy)
    arrays = _field_arrays(fields)
    np.savez_compressed(
        output_dir / f"fields_{step:04d}.npz",
        ux=arrays[0], uy=arrays[1], p=arrays[2], uibx=arrays[3], uiby=arrays[4],
        lambda_values=context.lam.dat.data_ro.copy(),
        chi_values=context.chi.dat.data_ro.copy(), step=step, beta=beta,
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 3), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (geometry.lambda_field, speed, pressure, divergence),
        ("lambda", "speed", "p", "div(u)"),
    ):
        image = axis.imshow(values, origin="lower", extent=(0, context.Lx, 0, context.Ly))
        axis.set_title(title)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.savefig(output_dir / f"preview_{step:04d}.png", dpi=160)
    plt.close(figure)


def run(config_path: Path, output_dir: Path, resume: Path | None, init_from: Path | None) -> None:
    from .forms import FiredrakeContext
    from .geometry import TPFMGeometry
    from .model import CoordinateMLP
    from .residual_bridge import FiredrakeResidualBridge
    from .state import FEFieldMapper

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    config_path = config_path.resolve()
    project_root = Path(__file__).resolve().parents[1]
    git_sha = get_git_sha(project_root)
    config = json.loads(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = resolve_path(config["dataset"], config_path, project_root)
    if not dataset.exists():
        raise FileNotFoundError(f"dataset not found: {dataset}")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    geometry = TPFMGeometry(
        dataset, sample_index=int(config["topology_index"]),
        spacing=float(config["spacing"]),
    )
    context = FiredrakeContext(
        geometry, int(config["nx"]), int(config["ny"]),
        velocity_degree=int(config["velocity_degree"]),
        pressure_degree=int(config["pressure_degree"]),
        quadrature_degree=int(config["quadrature_degree"]),
        nu=float(config["nu"]), ell=config["ell"], uin=float(config["uin"]),
        pout=float(config["pout"]), inlet_marker=int(config["inlet_marker"]),
        outlet_marker=int(config["outlet_marker"]),
        wall_markers=config["wall_markers"],
    )
    mapper = FEFieldMapper(
        context, geometry, u_scale=float(config["u_scale"]),
        p_scale=float(config["p_scale"]),
    )
    model = CoordinateMLP(
        input_dim=4, width=int(config["network_width"]),
        depth=int(config["network_depth"]),
    )
    continuation = config["continuation"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(continuation[0]["lr"]))
    bridge = FiredrakeResidualBridge(context, gamma=float(config["gamma"]))

    global_step = stage_index = stage_step = 0
    elapsed_offset = 0.0
    best_beta1_loss = float("inf")
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        global_step = int(checkpoint["global_step"])
        stage_index = int(checkpoint["stage_index"])
        stage_step = int(checkpoint["stage_step"])
        elapsed_offset = float(checkpoint["elapsed_seconds"])
        best_beta1_loss = float(checkpoint["best_beta1_loss"])
        bridge.Cm, bridge.Cc = float(checkpoint["Cm"]), float(checkpoint["Cc"])
        context.inv_Cm.assign(1.0 / bridge.Cm)
        context.inv_Cc.assign(1.0 / bridge.Cc)
        restore_rng(checkpoint)
    else:
        initial = init_from or (
            resolve_path(config["init_from"], config_path, project_root)
            if config.get("init_from") else None
        )
        if initial is not None:
            checkpoint = torch.load(initial, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
        bridge.initialize_normalization(mapper.evaluate(model))

    stage_index, stage_step = continuation_position(
        continuation, stage_index, stage_step
    )
    print(f"dataset={dataset}")
    print(f"normalization Cm={bridge.Cm:.12e} Cc={bridge.Cc:.12e}")
    warmup_fields = mapper.evaluate(model)
    bridge.evaluate_loss(warmup_fields, beta=float(continuation[min(stage_index, len(continuation)-1)]["beta"]))
    bridge.evaluate_loss_and_gradients(
        warmup_fields, beta=float(continuation[min(stage_index, len(continuation)-1)]["beta"])
    )
    print("JIT warm-up complete")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    history_path = output_dir / "history.csv"
    history_exists = history_path.exists() and history_path.stat().st_size > 0
    history_file = history_path.open("a", newline="", buffering=1)
    writer = csv.DictWriter(history_file, fieldnames=HISTORY_COLUMNS)
    if not history_exists:
        writer.writeheader()
        history_file.flush()

    started = time.monotonic()
    warning_streak = 0
    timed_steps = 0
    total_steps = sum(int(stage["steps"]) for stage in continuation)
    last_metrics = None
    last_seconds_per_step = float("nan")
    last_estimated_minutes = float("nan")
    last_beta = float(continuation[min(stage_index, len(continuation) - 1)]["beta"])

    def elapsed() -> float:
        return elapsed_offset + time.monotonic() - started

    def save_checkpoint(path: Path, beta: float) -> None:
        atomic_torch_save(checkpoint_payload(
            model=model, optimizer=optimizer, config=config, global_step=global_step,
            stage_index=stage_index, stage_step=stage_step, beta=beta,
            Cm=bridge.Cm, Cc=bridge.Cc, elapsed_seconds=elapsed(),
            best_beta1_loss=best_beta1_loss, git_sha=git_sha,
        ), path)

    try:
        while stage_index < len(continuation):
            stage = continuation[stage_index]
            beta, learning_rate = float(stage["beta"]), float(stage["lr"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            while stage_step < int(stage["steps"]):
                fields = mapper.evaluate(model)
                metrics, fe_gradients = bridge.evaluate_loss_and_gradients(fields, beta)
                arrays = _field_arrays(fields)
                if not np.isfinite(metrics.loss) or not all(np.all(np.isfinite(a)) for a in arrays):
                    raise FloatingPointError("non-finite loss or field")
                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(
                    (fields.ux, fields.uy, fields.p, fields.uibx, fields.uiby),
                    tuple(torch.as_tensor(getattr(fe_gradients, name), dtype=torch.float64)
                          for name in ("ux", "uy", "p", "uibx", "uiby")),
                )
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
                if not np.isfinite(gradient_norm):
                    raise FloatingPointError("non-finite gradient norm")
                ux, uy, p, _, _ = arrays
                max_speed = float(np.hypot(ux, uy).max())
                pressure_abs_max = float(np.abs(p).max())
                abort_threshold = (
                    max_speed > 2.0 or pressure_abs_max > 10.0
                    or gradient_norm > 1.0e8
                )
                warning_streak = warning_streak + 1 if abort_threshold else 0
                if max_speed > 1.0 or abort_threshold:
                    print(
                        f"WARNING step={global_step + 1} max_speed={max_speed:.6e} "
                        f"max_abs_pressure={pressure_abs_max:.6e} "
                        f"gradient_norm={gradient_norm:.6e} streak={warning_streak}"
                    )
                if warning_streak >= 5:
                    raise FloatingPointError("unsafe field or gradient threshold for 5 steps")
                diagnostics = _diagnostics(context, ux, uy, p)
                optimizer.step()
                global_step += 1
                stage_step += 1
                timed_steps += 1
                last_beta = beta
                last_metrics = metrics
                timed_elapsed = time.monotonic() - started
                last_seconds_per_step = timed_elapsed / timed_steps
                last_estimated_minutes = (
                    last_seconds_per_step * (total_steps - global_step) / 60.0
                )
                if timed_steps == 10:
                    projected_total_minutes = (
                        elapsed() + last_seconds_per_step * (total_steps - global_step)
                    ) / 60.0
                    print(f"projected total minutes: {projected_total_minutes:.3f}")
                    atomic_json(output_dir / "projection.json", {
                        "global_step": global_step,
                        "projected_total_minutes": projected_total_minutes,
                    })

                if global_step % int(config["log_every"]) == 0:
                    elapsed_seconds = elapsed()
                    row = {
                        "global_step": global_step, "stage": stage_index,
                        "stage_step": stage_step, "beta": beta,
                        "learning_rate": learning_rate,
                        "loss_total": metrics.loss,
                        "loss_momentum": metrics.loss_x + metrics.loss_y,
                        "loss_continuity": metrics.loss_continuity,
                        "loss_x_raw": metrics.raw_loss_x,
                        "loss_y_raw": metrics.raw_loss_y,
                        "loss_continuity_raw": metrics.raw_loss_continuity,
                        "residual_x_l2": metrics.residual_x_l2,
                        "residual_y_l2": metrics.residual_y_l2,
                        "residual_continuity_l2": metrics.residual_continuity_l2,
                        **diagnostics,
                        "gradient_norm": gradient_norm,
                        "seconds_per_step": last_seconds_per_step,
                        "elapsed_seconds": elapsed_seconds,
                        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                    }
                    writer.writerow(row)
                    history_file.flush()
                    print(
                        f"step={global_step}/{total_steps} beta={beta:g} "
                        f"loss={metrics.loss:.6e} grad={gradient_norm:.3e} "
                        f"seconds_per_step={last_seconds_per_step:.3f}"
                    )

                if global_step % int(config["field_save_every"]) == 0:
                    _save_snapshot(context, geometry, fields, beta, output_dir, global_step)
                if global_step % int(config["checkpoint_every"]) == 0:
                    save_checkpoint(output_dir / "checkpoint_latest.pt", beta)
                if global_step % 100 == 0:
                    save_checkpoint(output_dir / f"checkpoint_{global_step:04d}.pt", beta)
                if beta == 1.0 and metrics.loss < best_beta1_loss:
                    best_beta1_loss = metrics.loss
                    save_checkpoint(output_dir / "checkpoint_best_beta1.pt", beta)

                if global_step % 5 == 0:
                    atomic_json(output_dir / "heartbeat.json", heartbeat_payload(
                        status="running", global_step=global_step, stage=stage_index,
                        beta=beta, loss=metrics.loss,
                        best_beta1_loss=best_beta1_loss,
                        seconds_per_step=last_seconds_per_step,
                        estimated_remaining_minutes=last_estimated_minutes,
                        elapsed_seconds=elapsed(),
                    ))
                if _STOP_REQUESTED:
                    save_checkpoint(output_dir / "checkpoint_latest.pt", beta)
                    atomic_json(output_dir / "heartbeat.json", heartbeat_payload(
                        status="interrupted", global_step=global_step, stage=stage_index,
                        beta=beta, loss=metrics.loss,
                        best_beta1_loss=best_beta1_loss,
                        seconds_per_step=last_seconds_per_step,
                        estimated_remaining_minutes=last_estimated_minutes,
                        elapsed_seconds=elapsed(),
                    ))
                    return

            stage_index += 1
            stage_step = 0

        final_beta = float(continuation[-1]["beta"])
        save_checkpoint(output_dir / "checkpoint_latest.pt", final_beta)
        atomic_json(output_dir / "heartbeat.json", heartbeat_payload(
            status="done", global_step=global_step, stage=stage_index,
            beta=final_beta, loss=last_metrics.loss if last_metrics else float("nan"),
            best_beta1_loss=best_beta1_loss,
            seconds_per_step=last_seconds_per_step,
            estimated_remaining_minutes=0.0, elapsed_seconds=elapsed(),
        ))
        atomic_json(output_dir / "DONE.json", {
            "global_step": global_step, "elapsed_seconds": elapsed(),
            "best_beta1_loss": best_beta1_loss,
        })
    except BaseException as error:
        failure_traceback = traceback.format_exc()
        try:
            beta = float(continuation[min(stage_index, len(continuation)-1)]["beta"])
            save_checkpoint(output_dir / "checkpoint_latest.pt", beta)
        finally:
            atomic_json(output_dir / "FAILED.json", {
                "error": repr(error), "traceback": failure_traceback,
                "global_step": global_step, "elapsed_seconds": elapsed(),
            })
            atomic_json(output_dir / "heartbeat.json", heartbeat_payload(
                status="failed", global_step=global_step, stage=stage_index,
                beta=last_beta,
                loss=last_metrics.loss if last_metrics else float("nan"),
                best_beta1_loss=best_beta1_loss,
                seconds_per_step=last_seconds_per_step,
                estimated_remaining_minutes=last_estimated_minutes,
                elapsed_seconds=elapsed(),
            ))
        raise
    finally:
        history_file.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path)
    arguments = parser.parse_args(argv)
    run(arguments.config, arguments.output_dir, arguments.resume, arguments.init_from)


if __name__ == "__main__":
    main()
