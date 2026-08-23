import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.train import (
    HISTORY_COLUMNS,
    atomic_torch_save,
    checkpoint_payload,
    continuation_position,
    get_git_sha,
    heartbeat_payload,
    relative_mass_imbalance,
    restore_rng,
    validate_checkpoint_geometry,
)


EXPECTED_HISTORY_COLUMNS = [
    "global_step", "stage", "stage_step", "beta", "learning_rate", "loss_total",
    "loss_momentum", "loss_continuity", "loss_x_raw", "loss_y_raw",
    "loss_continuity_raw", "residual_x_l2", "residual_y_l2",
    "residual_continuity_l2", "divergence_l2", "inlet_flux", "outlet_flux",
    "mass_imbalance", "continuity_constant_test", "absolute_mass_defect",
    "max_speed", "mean_speed", "pressure_min", "pressure_max",
    "gradient_norm", "seconds_per_step", "elapsed_seconds", "rss_mb",
]


def test_history_columns_are_exact():
    assert HISTORY_COLUMNS == EXPECTED_HISTORY_COLUMNS


def test_continuation_position_never_duplicates_completed_stage():
    continuation = [
        {"beta": 0.0, "steps": 3, "lr": 1e-3},
        {"beta": 1.0, "steps": 2, "lr": 1e-3},
    ]
    assert continuation_position(continuation, 0, 2) == (0, 2)
    assert continuation_position(continuation, 0, 3) == (1, 0)
    assert continuation_position(continuation, 1, 2) == (2, 0)


def test_checkpoint_round_trip_restores_optimizer_counters_and_rng(tmp_path):
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(torch.ones((1, 2), dtype=torch.float64)).sum().backward()
    optimizer.step()
    payload = checkpoint_payload(
        model=model, optimizer=optimizer, config={"name": "test"}, global_step=7,
        stage_index=1, stage_step=2, beta=0.5, Cm=2.0, Cc=3.0,
        elapsed_seconds=4.0, best_beta1_loss=5.0, git_sha="abc123",
    )
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(payload, path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["global_step"] == 7
    assert loaded["stage_index"] == 1
    assert loaded["stage_step"] == 2
    required = {
        "model_state", "optimizer_state", "config", "global_step", "stage_index",
        "stage_step", "beta", "Cm", "Cc", "torch_rng", "numpy_rng",
        "python_rng", "git_sha", "elapsed_seconds",
    }
    assert set(loaded) == required | {"best_beta1_loss"}
    assert loaded["optimizer_state"]["state"]

    expected_python = random.random()
    expected_numpy = np.random.random()
    expected_torch = torch.rand(1)
    restore_rng(loaded)
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_explicit_checkpoint_geometry_identity_must_match():
    config = {
        "geometry_kind": "manufactured_empty",
        "benchmark_case": "manufactured_empty",
    }
    validate_checkpoint_geometry(config, {"config": dict(config)})
    with pytest.raises(ValueError, match="geometry identity"):
        validate_checkpoint_geometry(config, {"config": {
            "geometry_kind": "manufactured_circle",
            "benchmark_case": "manufactured_circle",
        }})
    validate_checkpoint_geometry({"name": "legacy"}, {"config": {}})


def test_relative_mass_imbalance_formula():
    assert relative_mass_imbalance(-2.0, 2.0) == 0.0
    expected = 1.0 / (3.0 + 1.0e-12)
    assert relative_mass_imbalance(-1.0, 2.0) == expected


def test_get_git_sha_marks_repository_as_safe(monkeypatch, tmp_path):
    project_root = tmp_path / "repo" / "project"
    project_root.mkdir(parents=True)
    captured = {}

    def check_output(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return "abc123\n"

    monkeypatch.setattr("src.train.subprocess.check_output", check_output)
    assert get_git_sha(project_root) == "abc123"
    assert captured["command"] == [
        "git", "-c", f"safe.directory={project_root.parent.resolve()}",
        "rev-parse", "HEAD",
    ]
    assert captured["cwd"] == project_root.parent.resolve()


def test_heartbeat_schema_is_exact():
    payload = heartbeat_payload(
        status="running", global_step=5, stage=1, beta=0.25, loss=2.0,
        best_beta1_loss=1.0, seconds_per_step=3.0,
        estimated_remaining_minutes=4.0, elapsed_seconds=120.0,
    )
    assert list(payload) == [
        "status", "global_step", "stage", "beta", "loss", "best_beta1_loss",
        "seconds_per_step", "estimated_remaining_minutes", "elapsed_minutes", "rss_mb",
    ]
    assert payload["elapsed_minutes"] == 2.0


def test_configs_and_launcher_use_exact_operational_contracts():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.json"):
        config = json.loads(path.read_text())
        if path.name.endswith("_geometry.json") or path.name == "controlled_geometry_smoke.json":
            continue
        assert {"log_every", "checkpoint_every", "field_save_every"} <= set(config)
        assert not {"log_interval", "checkpoint_interval", "field_interval"} & set(config)
    smoke = json.loads((root / "configs/smoke.json").read_text())
    assert smoke["seed"] == 11
    for name in ("tpfm_topology_b_scratch.json", "tpfm_topology_b_transfer.json"):
        assert json.loads((root / "configs" / name).read_text())["topology_index"] == 1

    launcher = (root / "scripts/launch_nohup.sh").read_text()
    assert "RUN_NAME CONFIG [CPUS]" in launcher
    assert "OUT=$PROJECT_ROOT/outputs/$RUN_NAME" in launcher
    assert "CONTAINER=fdhfdib_$RUN_NAME" in launcher
    assert "--rm --name" in launcher
    assert "--ipc=host" in launcher
    assert "stdout.log" in launcher


def test_controlled_configs_use_spec_values_and_required_resolutions():
    root = Path(__file__).resolve().parents[1]
    configs = {
        path.stem: json.loads(path.read_text())
        for path in (root / "configs").glob("controlled_*.json")
    }
    assert (configs["controlled_geometry_smoke"]["nx"], configs["controlled_geometry_smoke"]["ny"]) == (20, 10)
    for name in (
        "controlled_topology_a", "controlled_topology_b_scratch",
        "controlled_topology_b_transfer",
    ):
        config = configs[name]
        assert (config["nx"], config["ny"]) == (40, 20)
        assert config["benchmark_mode"] == "controlled"
        assert config["domain_spec"] == "geometry/controlled_tpfm_32cell.json"
        assert not {"uin", "pout", "nu", "spacing"} & set(config)
    assert configs["controlled_topology_a"]["seed"] == 11
    assert configs["controlled_topology_a"]["gamma"] == 1.0
    direct = [{"beta": 1.0, "steps": 200, "lr": 0.0005}]
    assert configs["controlled_topology_b_scratch"]["topology_index"] == 1
    assert configs["controlled_topology_b_transfer"]["topology_index"] == 1
    assert configs["controlled_topology_b_scratch"]["continuation"] == direct
    assert configs["controlled_topology_b_transfer"]["continuation"] == direct
    assert configs["controlled_topology_b_transfer"]["init_from"] == (
        "outputs/controlled_topoA/checkpoint_best_beta1.pt"
    )


def test_manufactured_configs_are_explicit_and_neural_runs_are_fresh():
    root = Path(__file__).resolve().parents[1]
    continuation_steps = [100, 50, 50, 200]
    for case in ("empty", "circle"):
        geometry = json.loads(
            (root / f"configs/manufactured_{case}_geometry.json").read_text()
        )
        neural = json.loads(
            (root / f"configs/manufactured_{case}_neural.json").read_text()
        )
        assert (geometry["nx"], geometry["ny"]) == (20, 10)
        assert (neural["nx"], neural["ny"]) == (40, 20)
        assert neural["benchmark_mode"] == "manufactured"
        assert neural["geometry_kind"] == f"manufactured_{case}"
        assert neural["benchmark_case"] == f"manufactured_{case}"
        assert (neural["velocity_degree"], neural["pressure_degree"]) == (2, 1)
        assert (neural["network_width"], neural["network_depth"]) == (64, 4)
        assert neural["gamma"] == 1.0
        assert [stage["steps"] for stage in neural["continuation"]] == continuation_steps
        assert "init_from" not in neural
