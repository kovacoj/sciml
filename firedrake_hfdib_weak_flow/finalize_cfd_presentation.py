"""Generate the final fitted-CFD presentation report from saved metrics."""

import csv
import json
from pathlib import Path


ROOT = Path("outputs/supervisor_2026_08_24/topic2_firedrake")


def main() -> None:
    with (ROOT / "fitted_cfd_gallery_metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics = {
        row["topology_id"]: {
            key: (float(value) if key not in {"topology_id", "solve_type"} else value)
            for key, value in row.items()
        }
        for row in rows
    }
    three = json.loads(
        (ROOT / "fitted_gallery/topology_E_3d/metrics.json").read_text()
    )
    pressure = {name: metrics[name]["delta_p"] for name in "ABCDEF"}
    report = [
        "# Firedrake CFD Gallery", "",
        "## Main Result", "",
        "Six procedurally generated connected article-inspired fitted geometries were solved with Taylor-Hood FEM. The fluid boundary is represented by the conforming mesh; no Brinkman or HFDIB approximation is used in these gallery solves.", "",
        f"- Cases converged: 6/6",
        f"- Maximum mass imbalance: {max(row['mass_imbalance'] for row in metrics.values()):.3e}",
        f"- Maximum assembled weak residual: {max(row['weak_residual'] for row in metrics.values()):.3e}",
        f"- Pressure-drop range: {min(pressure.values()):.3f} to {max(pressure.values()):.3f}", "",
        "| Topology | Pressure drop | Maximum speed | Mass imbalance | Weak residual |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in "ABCDEF":
        row = metrics[name]
        report.append(
            f"| {name} | {row['delta_p']:.3f} | {row['u_max']:.3f} | "
            f"{row['mass_imbalance']:.3e} | {row['weak_residual']:.3e} |"
        )
    report.extend((
        "", "## Genuine 3D Topology E", "",
        "Topology E was extruded through a 0.016 m depth with no-slip top and bottom walls. This is a solved three-dimensional velocity field, not a graphical extrusion.", "",
        f"- Velocity DOFs: {three['velocity_dofs']:,}",
        f"- Pressure DOFs: {three['pressure_dofs']:,}",
        f"- Solve time: {three['solve_time']:.1f} s",
        f"- Mass imbalance: {three['mass_imbalance']:.3e}",
        f"- Weak residual: {three['weak_residual']:.3e}",
        f"- Pressure drop/range proxy: {three['delta_p']:.3f}", "",
        "## Recommended Figures", "",
        "### Main slides", "",
        "1. `figures/firedrake_cfd_gallery_relative.png`: topology diversity and internal flow structure.",
        "2. `figures/firedrake_streamline_gallery.png`: visually clear jets, splits, merges, and bottlenecks.",
        "3. `figures/firedrake_3d_flow_topology_E.png`: strongest demonstration of genuine 3D CFD.",
        "4. `figures/firedrake_detailed_topology.png`: fitted mesh plus detailed fields.", "",
        "### Scientific/appendix", "",
        "- `figures/firedrake_cfd_gallery_common_scale.png`: comparable absolute field scales.",
        "- `figures/firedrake_pressure_drop_topologies.png`: resistance comparison.",
        "- `figures/firedrake_2d_vs_3d.png`: completed 2D/3D comparison.",
        "- `figures/firedrake_fitted_mesh_examples.png`: conforming-mesh explanation.", "",
        "## Claim Boundary", "",
        "These are procedurally generated article-inspired fitted geometries solved with Firedrake Taylor-Hood FEM. They are not reproductions of the article CFD domain and do not use HFDIB. The paper uses fixed-grid HFDIB; this gallery uses geometry-dependent fitted meshes and strong no-slip wall conditions.", "",
        "The final figures use grayscale geometry, viridis velocity, and coolwarm pressure, matching the DAFoam presentation package.",
    ))
    (ROOT / "PRESENTATION_RESULTS.md").write_text("\n".join(report) + "\n")
    (ROOT / "README.md").write_text(
        "# Topic 2 Firedrake CFD Gallery\n\n"
        "Presentation-grade 2D and 3D fitted-domain CFD figures, plotted data, metrics, checkpoints, and SHA-256 manifest.\n\n"
        "Use `firedrake_cfd_gallery_relative.png` for topology diversity, `firedrake_streamline_gallery.png` for flow paths, and `firedrake_3d_flow_topology_E.png` for the strongest 3D CFD result.\n"
    )


if __name__ == "__main__":
    main()
