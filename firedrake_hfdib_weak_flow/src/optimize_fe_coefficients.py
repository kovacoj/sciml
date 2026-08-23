"""Optimize Case0 finite-element coefficients against the shared weak residual."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from .benchmark_setup import load_config_geometry
from .compare_reference import compare
from .residual_bridge import FiredrakeResidualBridge
from .state import NeuralFields
from .train import _diagnostics, atomic_json


def run(config_path: Path, output_dir: Path, reference: Path | None = None) -> dict:
    """Run coefficient-only residual optimization without a neural network."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    config, _, _, _, _, context, _ = load_config_geometry(config_path)
    if context.formulation != "h1_weak":
        raise ValueError("FE coefficient optimization requires h1_weak")
    continuation = config["continuation"]
    if any(float(stage["beta"]) != 0.0 for stage in continuation):
        raise ValueError("FE coefficient optimization currently supports Case0 beta=0 only")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_ux = torch.zeros(context.S.dim(), requires_grad=True)
    raw_uy = torch.zeros(context.S.dim(), requires_grad=True)
    raw_p = torch.zeros(context.Q.dim(), requires_grad=True)
    parameters = (raw_ux, raw_uy, raw_p)
    optimizer_name = config.get("optimizer", "adam")
    velocity_mask = torch.from_numpy(context.velocity_mask)
    ux_lift = torch.from_numpy(context.ux_lift)
    zeros = torch.zeros(context.S.dim())
    velocity_scale = float(config.get("u_scale", 0.1))
    pressure_scale = float(config.get("p_scale", 0.2))

    def fields() -> NeuralFields:
        return NeuralFields(
            ux_lift + velocity_scale * velocity_mask * raw_ux,
            velocity_scale * velocity_mask * raw_uy,
            pressure_scale * raw_p,
            zeros,
            zeros,
        )

    bridge = FiredrakeResidualBridge(context, gamma=float(config.get("gamma", 1.0)))
    bridge.initialize_normalization(fields())
    atomic_json(output_dir / "normalization.json", {
        "Cm": bridge.Cm,
        "Cc": bridge.Cc,
        "gamma": bridge.gamma,
        "formulation": context.formulation,
    })
    initial = bridge.evaluate_loss(fields(), beta=0.0)
    history = []
    step = 0
    started = time.monotonic()
    optimizer_result = None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=float(continuation[0]["lr"]))
        for stage in continuation:
            for group in optimizer.param_groups:
                group["lr"] = float(stage["lr"])
            for _ in range(int(stage["steps"])):
                current = fields()
                metrics, gradients = bridge.evaluate_loss_and_gradients(current, beta=0.0)
                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(
                    (current.ux, current.uy, current.p),
                    tuple(
                        torch.from_numpy(getattr(gradients, name))
                        for name in ("ux", "uy", "p")
                    ),
                )
                optimizer.step()
                step += 1
                if step == 1 or step % int(config.get("log_every", 10)) == 0:
                    arrays = tuple(
                        value.detach().numpy()
                        for value in (current.ux, current.uy, current.p)
                    )
                    history.append({
                        "step": step,
                        **asdict(metrics),
                        **_diagnostics(context, *arrays),
                    })
    elif optimizer_name == "scipy_lbfgs":
        from scipy.optimize import minimize

        free = np.flatnonzero(context.velocity_mask)
        n_free = len(free)
        n_pressure = context.Q.dim()
        x0 = np.zeros(2 * n_free + n_pressure, dtype=np.float64)
        last = {}

        def set_coefficients(values: np.ndarray) -> None:
            with torch.no_grad():
                raw_ux.zero_()
                raw_uy.zero_()
                raw_ux[free] = torch.from_numpy(values[:n_free])
                raw_uy[free] = torch.from_numpy(values[n_free:2 * n_free])
                raw_p.copy_(torch.from_numpy(values[2 * n_free:]))

        def objective(values: np.ndarray):
            set_coefficients(np.asarray(values, dtype=np.float64))
            current = fields()
            metrics, gradients = bridge.evaluate_loss_and_gradients(current, beta=0.0)
            gradient = np.concatenate((
                velocity_scale * gradients.ux[free],
                velocity_scale * gradients.uy[free],
                pressure_scale * gradients.p,
            ))
            last.update(values=np.asarray(values).copy(), metrics=metrics,
                        gradient_norm=float(np.linalg.norm(gradient)))
            return metrics.loss, gradient

        def callback(intermediate) -> None:
            nonlocal step
            step += 1
            if step == 1 or step % int(config.get("log_every", 10)) == 0:
                set_coefficients(np.asarray(intermediate, dtype=np.float64))
                current = fields()
                metrics = bridge.evaluate_loss(current, beta=0.0)
                arrays = tuple(value.detach().numpy() for value in (
                    current.ux, current.uy, current.p
                ))
                history.append({
                    "step": step,
                    **asdict(metrics),
                    **_diagnostics(context, *arrays),
                })

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": int(config.get("maxiter", 1000)),
                "maxfun": int(config.get("maxfun", 2000)),
                "ftol": float(config.get("ftol", 1.0e-15)),
                "gtol": float(config.get("gtol", 1.0e-10)),
                "maxls": int(config.get("maxls", 50)),
            },
        )
        set_coefficients(result.x)
        step = int(result.nit)
        optimizer_result = {
            "name": optimizer_name,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "gradient_norm": float(np.linalg.norm(result.jac)),
        }
    else:
        raise ValueError(f"unknown FE coefficient optimizer: {optimizer_name}")

    final_fields = fields()
    final = bridge.evaluate_loss(final_fields, beta=0.0)
    arrays = {
        name: getattr(final_fields, name).detach().numpy().copy()
        for name in ("ux", "uy", "p")
    }
    np.savez_compressed(output_dir / "final_fields.npz", **arrays)
    comparison = (
        compare(output_dir / "final_fields.npz", reference)
        if reference is not None else None
    )
    result = {
        "steps": step,
        "elapsed_seconds": time.monotonic() - started,
        "initial": asdict(initial),
        "final": asdict(final),
        "diagnostics": _diagnostics(context, arrays["ux"], arrays["uy"], arrays["p"]),
        "comparison": comparison,
        "optimizer": optimizer_result or {"name": optimizer_name},
        "history": history,
        "config": config,
    }
    atomic_json(output_dir / "metrics.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    arguments = parser.parse_args(argv)
    run(arguments.config, arguments.output_dir, arguments.reference)


if __name__ == "__main__":
    main()
