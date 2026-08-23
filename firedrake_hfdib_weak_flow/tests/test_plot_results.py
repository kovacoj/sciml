import csv

from plot_results import TRANSFER_COLUMNS, resolve_dataset, write_transfer_csv


def test_resolve_dataset_checks_repository_parent(tmp_path):
    repo_root = tmp_path / "sciml"
    project_root = repo_root / "firedrake_hfdib_weak_flow"
    dataset = tmp_path / "tpfm_unet_reference" / "data" / "mixer_64.npz"
    project_root.mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    dataset.touch()

    resolved = resolve_dataset(
        {"dataset": "../tpfm_unet_reference/data/mixer_64.npz"}, project_root,
    )

    assert resolved == dataset


def test_transfer_csv_joins_matching_steps_with_exact_columns(tmp_path):
    def row(step, value):
        return {
            "global_step": step, "loss_total": value, "loss_momentum": value + 1,
            "loss_continuity": value + 2, "divergence_l2": value + 3,
            "mass_imbalance": value + 4, "elapsed_seconds": value + 5,
        }

    path = tmp_path / "transfer.csv"
    write_transfer_csv([row(5, 1.0), row(10, 2.0)], [row(5, 3.0), row(15, 4.0)], path)

    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == TRANSFER_COLUMNS
    assert len(rows) == 1
    assert rows[0]["step"] == "5"
    assert rows[0]["scratch_loss"] == "1.0"
    assert rows[0]["transfer_loss"] == "3.0"
