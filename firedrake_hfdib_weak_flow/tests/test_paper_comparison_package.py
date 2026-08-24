import csv
import json
from pathlib import Path

import numpy as np


def test_paper_comparison_package_is_complete_and_consistent():
    root = Path(__file__).parents[1] / "outputs/supervisor_2026_08_24/topic2_firedrake"
    if not root.exists():
        return
    with (root / "paper_morphology_matching.csv").open(newline="") as stream:
        matching = list(csv.DictReader(stream))
    expected = {"R1": "D", "R2": "E", "R3": "A"}
    for reference, topology in expected.items():
        selected = [
            row for row in matching
            if row["reference_class"] == reference and row["candidate"] == topology
        ]
        assert len(selected) == 1
        assert int(selected[0]["rank"]) == 1
        assert (root / f"fitted_gallery/topology_{topology}/metrics.json").exists()

    required = (
        "firedrake_paper_morphology_cfd_comparison",
        "firedrake_paper_morphology_cfd_comparison_relative",
        "reference_vs_firedrake_method",
        "reference_morphology_classes_vs_AtoF",
    )
    for stem in required:
        assert (root / "figures" / f"{stem}.png").stat().st_size > 0
        assert (root / "figures" / f"{stem}.pdf").stat().st_size > 0

    with np.load(root / "paper_morphology_comparison_data.npz") as data:
        np.testing.assert_array_equal(data["reference_classes"], ["R1", "R2", "R3"])
        np.testing.assert_array_equal(data["selected_topology_ids"], ["D", "E", "A"])
        assert float(data["global_velocity_max"]) > 0
        assert float(data["global_pressure_abs_max"]) > 0
        for reference in ("R1", "R2", "R3"):
            mask = data[f"{reference}_geometry_mask"].astype(bool)
            for field in ("ux", "uy", "velocity_magnitude", "pressure"):
                values = data[f"{reference}_{field}"]
                assert np.all(np.isfinite(values[mask]))
                assert np.all(np.isnan(values[~mask]))

    paper = json.loads((root / "paper_reference_figure6_metrics.json").read_text())
    assert paper["resolution"] == "64x64"
    assert paper["case_1"]["velocity_MSE_TV"] == 0.040

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["_metadata"]["morphology_selection"] == expected
    for stem in required:
        assert f"figures/{stem}.png" in manifest
        assert f"figures/{stem}.pdf" in manifest
    assert "paper_morphology_comparison_data.npz" in manifest
