"""Compare neural FE coefficient snapshots with an independent direct reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _checkpoint_fields(path: Path, config_path: Path | None) -> dict[str, np.ndarray]:
    if config_path is None:
        raise ValueError("--config is required when --neural is a checkpoint")
    import torch

    from .benchmark_setup import load_config_geometry
    from .model import CoordinateMLP
    from .train import validate_checkpoint_geometry

    config, _, _, _, _, _, mapper = load_config_geometry(config_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_geometry(config, checkpoint)
    model = CoordinateMLP(
        input_dim=4,
        width=int(config["network_width"]),
        depth=int(config["network_depth"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    fields = mapper.evaluate(model)
    return {
        name: getattr(fields, name).detach().cpu().numpy()
        for name in ("ux", "uy", "p")
    }


def compare(
    neural_path: Path, reference_path: Path, config_path: Path | None = None
) -> dict:
    checkpoint_fields = (
        _checkpoint_fields(neural_path, config_path)
        if neural_path.suffix in {".pt", ".pth"} else None
    )
    neural_archive = None if checkpoint_fields is not None else np.load(neural_path)
    neural = checkpoint_fields if checkpoint_fields is not None else neural_archive
    try:
        with np.load(reference_path) as reference:
            result = {"norm": "simple_unweighted_FE_coefficient_relative_l2"}
            for name in ("ux", "uy"):
                actual = np.asarray(neural[name], dtype=np.float64)
                expected = np.asarray(reference[name], dtype=np.float64)
                if actual.shape != expected.shape:
                    raise ValueError(
                        f"{name} coefficient shapes differ: {actual.shape} != {expected.shape}"
                    )
                result[f"{name}_relative_l2"] = float(
                    np.linalg.norm(actual - expected)
                    / (np.linalg.norm(expected) + 1.0e-30)
                )
            actual_p = np.asarray(neural["p"], dtype=np.float64)
            expected_p = np.asarray(reference["p"], dtype=np.float64)
            if actual_p.shape != expected_p.shape:
                raise ValueError(
                    f"p coefficient shapes differ: {actual_p.shape} != {expected_p.shape}"
                )
            actual_p = actual_p - actual_p.mean()
            expected_p = expected_p - expected_p.mean()
            result["pressure_gauge_centered_relative_l2"] = float(
                np.linalg.norm(actual_p - expected_p)
                / (np.linalg.norm(expected_p) + 1.0e-30)
            )
            speed = np.hypot(neural["ux"], neural["uy"])
            result["neural_max_speed"] = float(speed.max())
            result["neural_pressure_range"] = float(np.ptp(neural["p"]))
        return result
    finally:
        if neural_archive is not None:
            neural_archive.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = compare(arguments.neural, arguments.reference, arguments.config)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
