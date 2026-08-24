"""Reference-morphology matching and direct fitted-CFD comparison plates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import FancyArrowPatch
import numpy as np
from scipy.interpolate import griddata


ROOT = Path("outputs/supervisor_2026_08_24/topic2_firedrake")
DATA = ROOT / "fitted_gallery"
FIGURES = ROOT / "figures"
PLOTTED = ROOT / "plotted_data"
FIGURES.mkdir(parents=True, exist_ok=True)
PLOTTED.mkdir(parents=True, exist_ok=True)
NAMES = tuple("ABCDEF")
SELECTED = {"R1": "D", "R2": "E", "R3": "A"}
CLASS_NAMES = {
    "R1": "complex irregular network",
    "R2": "central merge/split",
    "R3": "parallel channels",
}

# Scores are geometry-only ordinal descriptors: 0 absent, 1 weak, 2 moderate,
# 3 strong. They are fixed before loading any CFD metrics.
SCORES = {
    "A": (1, 1, 3, 0, 0, 0, "Two straight horizontal channels and one vertical bridge."),
    "B": (2, 3, 0, 2, 0, 2, "Crossed feeders with a compact central chamber; approximately symmetric."),
    "C": (2, 3, 0, 2, 0, 3, "Two feeders merge into a broad central shared path and split again."),
    "D": (2, 1, 0, 3, 3, 0, "Asymmetric offset multi-bend paths; strongest maze-like tortuosity."),
    "E": (3, 3, 0, 2, 0, 3, "Four branches terminate at a broad central chamber; explicit merge/split."),
    "F": (2, 1, 3, 1, 2, 0, "Two parallel channels with one oblique communicating side branch."),
}
# branch, junction, parallel, tortuosity, asymmetry, chamber
CLASS_WEIGHTS = {
    "R1": (2, 2, -1, 3, 2, 0),
    "R2": (2, 3, -1, 1, 0, 3),
    "R3": (-1, -1, 3, -2, -1, -1),
}


def load_cases():
    cases, metrics = {}, {}
    for name in NAMES:
        with np.load(DATA / f"topology_{name}/fields.npz") as archive:
            cases[name] = {key: archive[key].copy() for key in archive.files}
        metrics[name] = json.loads((DATA / f"topology_{name}/metrics.json").read_text())
    return cases, metrics


def tri(case):
    xy = case["coordinates"][:, :2]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], case["cells"])


def style(axis):
    axis.set_aspect("equal"); axis.set_xlim(0, .128); axis.set_ylim(0, .128)
    axis.set_xticks([]); axis.set_yticks([])


def morphology_csv():
    rows = []
    for reference, weights in CLASS_WEIGHTS.items():
        scored = []
        for candidate in NAMES:
            values = SCORES[candidate][:6]
            score = sum(weight * value for weight, value in zip(weights, values))
            scored.append((candidate, score))
        rank = {candidate: index + 1 for index, (candidate, _) in enumerate(
            sorted(scored, key=lambda item: (-item[1], item[0]))
        )}
        for candidate, _ in scored:
            branch, junction, parallel, tortuosity, asymmetry, chamber, notes = SCORES[candidate]
            rows.append({
                "candidate": candidate, "reference_class": reference,
                "branch_score": branch, "junction_score": junction,
                "parallel_path_score": parallel, "tortuosity_score": tortuosity,
                "asymmetry_score": asymmetry, "chamber_score": chamber,
                "qualitative_notes": notes, "rank": rank[candidate],
            })
    path = ROOT / "paper_morphology_matching.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return rows


def selection_sheet(cases, rows):
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    chosen_for = {value: key for key, value in SELECTED.items()}
    for axis, name in zip(axes.flat, NAMES):
        topology = tri(cases[name])
        axis.set_facecolor("#d8d2c4")
        axis.tripcolor(topology, np.ones(len(cases[name]["coordinates"])), color="#eaf7f8")
        axis.triplot(topology, color="#66747a", lw=.18)
        title = f"Topology {name}"
        if name in chosen_for:
            reference = chosen_for[name]
            title += f"\nselected for {reference}"
            axis.text(.5, -.08, CLASS_NAMES[reference], transform=axis.transAxes,
                      ha="center", va="top", fontsize=9, color="#9b3a32")
        axis.set_title(title, weight="bold" if name in chosen_for else "normal")
        style(axis)
    figure.suptitle("Geometry-only reference morphology selection", fontsize=15)
    figure.savefig(FIGURES / "firedrake_reference_morphology_selection.png", dpi=320, facecolor="white")
    plt.close(figure)


def interpolate_case(case, resolution=200):
    xy = case["coordinates"][:, :2]
    x = np.linspace(0, .128, resolution); y = np.linspace(0, .128, resolution)
    xx, yy = np.meshgrid(x, y); topology = tri(case)
    inside = topology.get_trifinder()(xx, yy) >= 0
    ux = griddata(xy, case["velocity"][:, 0], (xx, yy), method="linear")
    uy = griddata(xy, case["velocity"][:, 1], (xx, yy), method="linear")
    pressure = griddata(xy, case["pressure"] - case["pressure"].mean(), (xx, yy), method="linear")
    speed = np.hypot(ux, uy)
    for field in (ux, uy, pressure, speed):
        field[~inside] = np.nan
    return x, y, xx, yy, ux, uy, speed, pressure, inside


def comparison_plate(cases, metrics, relative=False):
    selected = tuple((reference, SELECTED[reference]) for reference in ("R1", "R2", "R3"))
    grids = {reference: interpolate_case(cases[name]) for reference, name in selected}
    velocity_max = max(np.nanmax(grid[6]) for grid in grids.values())
    pressure_abs = max(np.nanmax(np.abs(grid[7])) for grid in grids.values())
    figure, axes = plt.subplots(3, 4, figsize=(14, 9.2), constrained_layout=True)
    velocity_artist = pressure_artist = None
    plotted = {}
    for row, (reference, name) in enumerate(selected):
        case = cases[name]; topology = tri(case)
        x, y, xx, yy, ux, uy, speed, pressure, inside = grids[reference]
        display_speed, display_pressure = speed.copy(), pressure.copy()
        if relative:
            display_speed /= np.nanmax(display_speed) + 1e-30
            finite = np.isfinite(display_pressure)
            minimum, maximum = np.nanmin(display_pressure), np.nanmax(display_pressure)
            display_pressure[finite] = (display_pressure[finite] - minimum) / (maximum - minimum + 1e-30)
            velocity_limits, pressure_limits = (0, 1), (0, 1)
        else:
            velocity_limits, pressure_limits = (0, velocity_max), (-pressure_abs, pressure_abs)
        axes[row, 0].set_facecolor("#d8d2c4")
        axes[row, 0].tripcolor(topology, np.ones(len(case["coordinates"])), color="#eaf7f8")
        axes[row, 0].triplot(topology, color="#66747a", lw=.18)
        velocity_artist = axes[row, 1].pcolormesh(
            xx, yy, display_speed, shading="auto", cmap="viridis",
            vmin=velocity_limits[0], vmax=velocity_limits[1],
        )
        axes[row, 2].pcolormesh(
            xx, yy, display_speed, shading="auto", cmap="viridis",
            vmin=velocity_limits[0], vmax=velocity_limits[1],
        )
        axes[row, 2].streamplot(x, y, ux, uy, color="white", density=1.2,
                                linewidth=.65, arrowsize=.62)
        axes[row, 2].contour(xx, yy, inside.astype(float), levels=[.5], colors="black", linewidths=.8)
        pressure_artist = axes[row, 3].pcolormesh(
            xx, yy, display_pressure, shading="auto", cmap="coolwarm",
            vmin=pressure_limits[0], vmax=pressure_limits[1],
        )
        for axis in axes[row]:
            style(axis)
        metric = metrics[name]
        axes[row, 0].set_ylabel(
            f"{reference} / {name}\n{CLASS_NAMES[reference]}\n"
            f"Δp={metric['delta_p']:.2f}, umax={metric['u_max']:.3f}\n"
            f"εmass={metric['mass_imbalance']:.1e}, R={metric['weak_residual']:.1e}",
            rotation=0, ha="right", va="center", labelpad=12, fontsize=8.5,
        )
        plotted[reference] = {
            "topology_id": name, "x": x, "y": y, "geometry_mask": inside,
            "ux": ux, "uy": uy, "velocity_magnitude": speed, "pressure": pressure,
        }
    headers = (
        "Fitted geometry", "Relative |u|" if relative else "Velocity magnitude |u|",
        "Relative |u| + streamlines" if relative else "|u| + streamlines",
        "Relative pressure" if relative else "Gauge pressure",
    )
    for axis, title in zip(axes[0], headers):
        axis.set_title(title, fontsize=12)
    figure.colorbar(velocity_artist, ax=axes[:, 1:3], fraction=.018, pad=.012)
    figure.colorbar(pressure_artist, ax=axes[:, 3], fraction=.025, pad=.02)
    figure.suptitle(
        "Reference-morphology classes mapped to direct fitted Firedrake CFD"
        + (" — relative display" if relative else " — common scientific scales"),
        fontsize=15,
    )
    figure.text(
        .5, .002,
        "Reference: fixed-grid HFDIB topology field. Ours: conforming fitted-domain Taylor-Hood FEM.",
        ha="center", fontsize=9,
    )
    suffix = "_relative" if relative else ""
    stem = f"firedrake_paper_morphology_cfd_comparison{suffix}"
    figure.savefig(FIGURES / f"{stem}.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / f"{stem}.pdf", facecolor="white")
    plt.close(figure)
    return plotted, velocity_max, pressure_abs


def method_schematic():
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    blocks = (
        (axes[0], "REFERENCE", ("8×8 bitmap", "64×64 λ", "fixed Cartesian mesh", "HFDIB", "u, p")),
        (axes[1], "OURS", ("geometry definition", "conforming fitted mesh", "Taylor-Hood P2/P1", "weak Navier-Stokes", "u, p")),
    )
    for axis, title, labels in blocks:
        axis.axis("off"); axis.set_title(title, fontsize=15, weight="bold")
        y_values = np.linspace(.86, .14, len(labels))
        for index, (y, label) in enumerate(zip(y_values, labels)):
            axis.text(.5, y, label, transform=axis.transAxes, ha="center", va="center",
                      fontsize=12, bbox=dict(boxstyle="round,pad=.45", fc="#eef4f8", ec="#355c75"))
            if index + 1 < len(labels):
                axis.annotate("", xy=(.5, y_values[index + 1] + .055), xytext=(.5, y - .055),
                              xycoords=axis.transAxes, arrowprops=dict(arrowstyle="->", lw=1.6))
    figure.text(.5, .01, "Same physical question — different spatial discretization", ha="center", fontsize=13, weight="bold")
    figure.savefig(FIGURES / "reference_vs_firedrake_method.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "reference_vs_firedrake_method.pdf", facecolor="white")
    plt.close(figure)


def family_schematic(cases):
    figure = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(.55, 1))
    reference_axis = figure.add_subplot(grid[0, :]); reference_axis.axis("off")
    positions = {"R1": .17, "R2": .50, "R3": .83}
    for reference, x in positions.items():
        reference_axis.text(x, .65, f"{reference}\n{CLASS_NAMES[reference]}",
                            transform=reference_axis.transAxes, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=.5", fc="#fff3d6", ec="#9c7532"))
        target = (ord(SELECTED[reference]) - ord("A") + .5) / 6
        reference_axis.add_patch(FancyArrowPatch(
            (x, .40), (target, -.05), transform=reference_axis.transAxes,
            arrowstyle="->", mutation_scale=14, color="#9c7532", clip_on=False,
        ))
    for column, name in enumerate(NAMES):
        axis = figure.add_subplot(grid[1, column]); topology = tri(cases[name])
        axis.set_facecolor("#d8d2c4")
        axis.tripcolor(topology, np.ones(len(cases[name]["coordinates"])), color="#eaf7f8")
        axis.triplot(topology, color="#66747a", lw=.16); axis.set_title(name, weight="bold")
        style(axis)
    figure.suptitle("Reference morphology classes and our fitted topology family", fontsize=15)
    figure.savefig(FIGURES / "reference_morphology_classes_vs_AtoF.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "reference_morphology_classes_vs_AtoF.pdf", facecolor="white")
    plt.close(figure)


def write_reference_files():
    reference = {
        "resolution": "64x64",
        "case_1": {"velocity_MSE_TV": .040, "pressure_MSE_TV": .140,
                   "global_continuity": -3.9e-6, "pressure_drop_ratio": 1.14},
        "case_2": {"velocity_MSE_TV": .025, "pressure_MSE_TV": .093,
                   "global_continuity": 3.8e-6, "pressure_drop_ratio": 1.13},
        "case_3": {"velocity_MSE_TV": .015, "pressure_MSE_TV": .053,
                   "global_continuity": -6.6e-7, "pressure_drop_ratio": 1.15},
    }
    (ROOT / "paper_reference_figure6_metrics.json").write_text(json.dumps(reference, indent=2) + "\n")
    notes = """# Paper Comparison Notes

## Reference paper

- Geometry: 8x8 bitmap -> 64x64 lambda field -> fixed-grid HFDIB.
- Lambda convention: 0 fluid, 1 solid.
- Physics: uin=0.1, nu=1e-2, pout=0, steady low-Re Navier-Stokes.
- Figure 6: three unseen 64x64 topologies.
- Columns: lambda, IBM relative velocity, U-Net relative velocity, IBM relative pressure, U-Net relative pressure.

| Case | MSE-TV u | MSE-TV p | global continuity | pressure-drop ratio |
|---|---:|---:|---:|---:|
| 1 | 0.040 | 0.140 | -3.9e-6 | 1.14 |
| 2 | 0.025 | 0.093 | 3.8e-6 | 1.13 |
| 3 | 0.015 | 0.053 | -6.6e-7 | 1.15 |

The paper reports velocity fields as visually difficult to distinguish, occasional ghost currents, more apparent pressure differences, and about 15% excessive pressure drop in these unseen cases.

## Firedrake morphology mapping

- R1 complex/tortuous network -> Topology D: asymmetric offset multi-bend paths.
- R2 merge/split/chamber -> Topology E: broad chamber connected to four branches.
- R3 parallel channels -> Topology A: two long horizontal paths with one bridge.

Reference uses fixed-grid HFDIB topology fields. Firedrake uses geometry-dependent fitted meshes and direct Taylor-Hood Navier-Stokes solves. These are comparable in physical intent, not identical discretizations or reproduced geometries.

Paper global continuity and Firedrake mass imbalance/assembled weak residual are not definition-identical. Do not claim an orders-of-magnitude improvement from juxtaposing them.
"""
    (ROOT / "PAPER_COMPARISON_NOTES.md").write_text(notes)


def save_data(plotted, velocity_max, pressure_abs):
    payload = {
        "reference_classes": np.asarray(("R1", "R2", "R3")),
        "selected_topology_ids": np.asarray(tuple(SELECTED[key] for key in ("R1", "R2", "R3"))),
        "global_velocity_max": velocity_max,
        "global_pressure_abs_max": pressure_abs,
    }
    for reference, values in plotted.items():
        for key, value in values.items():
            payload[f"{reference}_{key}"] = np.asarray(value)
    np.savez_compressed(ROOT / "paper_morphology_comparison_data.npz", **payload)


def update_docs_and_manifest():
    report_path = ROOT / "PRESENTATION_RESULTS.md"
    text = report_path.read_text()
    section = """
## Reference-paper morphology comparison

- R1 complex irregular network: topology D — asymmetric, offset, multi-bend paths with the strongest tortuosity.
- R2 central merge/split: topology E — broad central chamber with four connected feeder/exit branches.
- R3 parallel channels: topology A — two dominant horizontal channels with one communicating bridge.

Selection uses geometry-only ordinal scores in `paper_morphology_matching.csv`; CFD metrics were not used. Primary scientific fields share common velocity and gauge-pressure scales. Relative fields are explicitly presentation-only displays.
"""
    if "## Reference-paper morphology comparison" not in text:
        report_path.write_text(text.rstrip() + "\n" + section)
    readme = ROOT / "README.md"
    readme.write_text(readme.read_text().rstrip() + "\n\nMorphology comparison: `firedrake_paper_morphology_cfd_comparison_relative.png` (slide display) and `firedrake_paper_morphology_cfd_comparison.png` (common scientific scales).\n")
    manifest = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
        and "study_2d_vs_3d" not in path.parts
    }
    manifest["_metadata"] = {
        "cases_2d": 6, "has_completed_3d": True,
        "morphology_selection": SELECTED,
        "morphology_selection_basis": "geometry-only ordinal scores",
        "common_scales": True,
        "style": {"geometry": "muted gray/cyan", "velocity": "viridis",
                  "pressure": "coolwarm", "background": "white"},
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    cases, metrics = load_cases()
    rows = morphology_csv()
    selection_sheet(cases, rows)
    plotted, velocity_max, pressure_abs = comparison_plate(cases, metrics, relative=False)
    comparison_plate(cases, metrics, relative=True)
    method_schematic(); family_schematic(cases); write_reference_files()
    save_data(plotted, velocity_max, pressure_abs)
    update_docs_and_manifest()


if __name__ == "__main__":
    main()
