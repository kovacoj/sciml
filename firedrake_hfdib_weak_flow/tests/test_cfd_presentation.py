import json
from pathlib import Path


def test_cfd_presentation_manifest_and_required_figures():
    root = Path(__file__).parents[1] / "outputs/supervisor_2026_08_24/topic2_firedrake"
    if not root.exists():
        return
    required = (
        "firedrake_cfd_gallery_common_scale.png",
        "firedrake_cfd_gallery_relative.png",
        "firedrake_streamline_gallery.png",
        "firedrake_pressure_drop_topologies.png",
        "firedrake_detailed_topology.png",
        "firedrake_3d_flow_topology_E.png",
        "firedrake_2d_vs_3d.png",
        "firedrake_fitted_mesh_examples.png",
    )
    for name in required:
        assert (root / "figures" / name).stat().st_size > 0
    manifest = json.loads((root / "manifest.json").read_text())
    for name in required:
        assert f"figures/{name}" in manifest
    for name in "ABCDEF":
        assert f"fitted_gallery/topology_{name}/fields.npz" in manifest
        assert f"fitted_gallery/topology_{name}/solution.h5" in manifest
    metrics = json.loads(
        (root / "fitted_gallery/topology_E_3d/metrics.json").read_text()
    )
    assert metrics["mass_imbalance"] < 1.0e-12
    assert metrics["velocity_dofs"] == 40527
    assert metrics["pressure_dofs"] == 2115
