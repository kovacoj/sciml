"""Refine a trained Case0 coordinate MLP with exact external L-BFGS gradients."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import minimize

from .benchmark_setup import load_config_geometry
from .model import CoordinateMLP
from .residual_bridge import FiredrakeResidualBridge
from .train import _diagnostics, atomic_json, validate_checkpoint_geometry


def run(config_path: Path, checkpoint_path: Path, output_dir: Path) -> dict:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    config, _, _, _, _, context, mapper = load_config_geometry(config_path)
    if context.formulation != "h1_weak":
        raise ValueError("neural L-BFGS refinement requires h1_weak")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_geometry(config, checkpoint)
    model = CoordinateMLP(
        input_dim=4,
        width=int(config["network_width"]),
        depth=int(config["network_depth"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    bridge = FiredrakeResidualBridge(context, gamma=float(config["gamma"]))
    bridge.Cm = float(checkpoint["Cm"])
    bridge.Cc = float(checkpoint["Cc"])
    context.inv_Cm.assign(1.0 / bridge.Cm)
    context.inv_Cc.assign(1.0 / bridge.Cc)
    parameters = tuple(model.parameters())
    shapes = tuple(parameter.shape for parameter in parameters)
    sizes = tuple(parameter.numel() for parameter in parameters)

    def flatten_parameters() -> np.ndarray:
        return np.concatenate([
            parameter.detach().numpy().reshape(-1) for parameter in parameters
        ])

    def assign_parameters(values: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for parameter, shape, size in zip(parameters, shapes, sizes):
                parameter.copy_(torch.from_numpy(values[offset:offset + size].reshape(shape)))
                offset += size

    evaluations = 0
    last_metrics = None

    def objective(values: np.ndarray):
        nonlocal evaluations, last_metrics
        assign_parameters(np.asarray(values, dtype=np.float64))
        fields = mapper.evaluate(model)
        metrics, gradients = bridge.evaluate_loss_and_gradients(fields, beta=0.0)
        model.zero_grad(set_to_none=True)
        torch.autograd.backward(
            (fields.ux, fields.uy, fields.p, fields.uibx, fields.uiby),
            tuple(torch.from_numpy(getattr(gradients, name)) for name in (
                "ux", "uy", "p", "uibx", "uiby"
            )),
        )
        gradient = np.concatenate([
            parameter.grad.detach().numpy().reshape(-1) for parameter in parameters
        ])
        evaluations += 1
        last_metrics = metrics
        return metrics.loss, gradient

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = minimize(
        objective,
        flatten_parameters(),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(config.get("lbfgs_maxiter", 1000)),
            "maxfun": int(config.get("lbfgs_maxfun", 2000)),
            "ftol": float(config.get("lbfgs_ftol", 1.0e-15)),
            "gtol": float(config.get("lbfgs_gtol", 1.0e-10)),
            "maxls": int(config.get("lbfgs_maxls", 50)),
            "maxcor": int(config.get("lbfgs_maxcor", 20)),
        },
    )
    assign_parameters(result.x)
    fields = mapper.evaluate(model)
    final = bridge.evaluate_loss(fields, beta=0.0)
    arrays = {
        name: getattr(fields, name).detach().numpy().copy()
        for name in ("ux", "uy", "p", "uibx", "uiby")
    }
    np.savez_compressed(output_dir / "final_fields.npz", **arrays)
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "Cm": bridge.Cm,
        "Cc": bridge.Cc,
        "source_checkpoint": str(checkpoint_path),
    }, output_dir / "checkpoint_lbfgs.pt")
    report = {
        "elapsed_seconds": time.monotonic() - started,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "gradient_norm": float(np.linalg.norm(result.jac)),
        "final": asdict(final),
        "diagnostics": _diagnostics(context, arrays["ux"], arrays["uy"], arrays["p"]),
        "source_checkpoint": str(checkpoint_path),
        "evaluations": evaluations,
    }
    atomic_json(output_dir / "metrics.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run(arguments.config, arguments.checkpoint, arguments.output_dir)


if __name__ == "__main__":
    main()
