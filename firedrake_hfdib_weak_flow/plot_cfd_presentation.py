"""Presentation-grade figures from completed fitted Firedrake CFD artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import griddata


ROOT = Path("outputs/supervisor_2026_08_24/topic2_firedrake")
DATA = ROOT / "fitted_gallery"
FIGURES = ROOT / "figures"
PLOTTED = ROOT / "plotted_data"
FIGURES.mkdir(parents=True, exist_ok=True)
PLOTTED.mkdir(parents=True, exist_ok=True)
NAMES = tuple("ABCDEF")
VELOCITY_CMAP = "viridis"
PRESSURE_CMAP = "coolwarm"


def load_cases():
    cases, metrics = {}, {}
    for name in NAMES:
        with np.load(DATA / f"topology_{name}/fields.npz") as archive:
            cases[name] = {key: archive[key].copy() for key in archive.files}
        metrics[name] = json.loads(
            (DATA / f"topology_{name}/metrics.json").read_text()
        )
    return cases, metrics


def triangulation(case):
    xy = case["coordinates"][:, :2]
    return mtri.Triangulation(xy[:, 0], xy[:, 1], case["cells"])


def gauge_pressure(values):
    return values - np.mean(values)


def format_axis(axis):
    axis.set_aspect("equal")
    axis.set_xlim(0, .128)
    axis.set_ylim(0, .128)
    axis.set_xticks([])
    axis.set_yticks([])


def gallery(cases, relative=False):
    speeds = {name: np.linalg.norm(cases[name]["velocity"], axis=1) for name in NAMES}
    pressures = {name: gauge_pressure(cases[name]["pressure"]) for name in NAMES}
    vmax = max(values.max() for values in speeds.values())
    pmax = max(np.abs(values).max() for values in pressures.values())
    figure, axes = plt.subplots(6, 3, figsize=(9.2, 15.2), constrained_layout=True)
    velocity_artist = pressure_artist = None
    for row, name in enumerate(NAMES):
        case = cases[name]
        tri = triangulation(case)
        speed, pressure = speeds[name].copy(), pressures[name].copy()
        axes[row, 0].tripcolor(tri, np.zeros(len(speed)), cmap="Greys", vmin=0, vmax=1)
        axes[row, 0].triplot(tri, color="0.72", lw=.08)
        if relative:
            speed /= speed.max() + 1e-30
            pressure /= np.abs(pressure).max() + 1e-30
            velocity_limits, pressure_limits = (0, 1), (-1, 1)
        else:
            velocity_limits, pressure_limits = (0, vmax), (-pmax, pmax)
        velocity_artist = axes[row, 1].tripcolor(
            tri, speed, shading="gouraud", cmap=VELOCITY_CMAP,
            vmin=velocity_limits[0], vmax=velocity_limits[1],
        )
        pressure_artist = axes[row, 2].tripcolor(
            tri, pressure, shading="gouraud", cmap=PRESSURE_CMAP,
            vmin=pressure_limits[0], vmax=pressure_limits[1],
        )
        for axis in axes[row]:
            axis.triplot(tri, color="0.15", lw=.18, alpha=.45)
            format_axis(axis)
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=13, fontsize=12, weight="bold")
    headers = (
        "Fitted geometry",
        "Relative |u|" if relative else "Velocity magnitude |u| [m/s]",
        "Relative pressure" if relative else "Gauge pressure",
    )
    for axis, title in zip(axes[0], headers):
        axis.set_title(title, fontsize=12)
    figure.colorbar(velocity_artist, ax=axes[:, 1], fraction=.025, pad=.02)
    figure.colorbar(pressure_artist, ax=axes[:, 2], fraction=.025, pad=.02)
    figure.suptitle(
        "Steady Navier-Stokes flow through article-inspired fitted topologies",
        fontsize=15,
    )
    suffix = "relative" if relative else "common_scale"
    figure.savefig(FIGURES / f"firedrake_cfd_gallery_{suffix}.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / f"firedrake_cfd_gallery_{suffix}.pdf", facecolor="white")
    plt.close(figure)
    return vmax, pmax


def interpolate_fluid(case, size=180):
    xy = case["coordinates"][:, :2]
    x = np.linspace(xy[:, 0].min(), xy[:, 0].max(), size)
    y = np.linspace(xy[:, 1].min(), xy[:, 1].max(), size)
    xx, yy = np.meshgrid(x, y)
    tri = triangulation(case)
    finder = tri.get_trifinder()
    inside = finder(xx, yy) >= 0
    ux = griddata(xy, case["velocity"][:, 0], (xx, yy), method="linear")
    uy = griddata(xy, case["velocity"][:, 1], (xx, yy), method="linear")
    speed = np.hypot(ux, uy)
    ux[~inside] = np.nan; uy[~inside] = np.nan; speed[~inside] = np.nan
    return x, y, xx, yy, ux, uy, speed, inside


def streamline_gallery(cases):
    selected = ("A", "C", "E", "F")
    fields = {name: interpolate_fluid(cases[name]) for name in selected}
    vmax = max(np.nanmax(value[6]) for value in fields.values())
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    plotted = {}
    for axis, name in zip(axes.flat, selected):
        x, y, xx, yy, ux, uy, speed, inside = fields[name]
        image = axis.pcolormesh(xx, yy, speed, shading="auto", cmap=VELOCITY_CMAP, vmin=0, vmax=vmax)
        axis.streamplot(x, y, ux, uy, color="white", density=1.25, linewidth=.65, arrowsize=.65)
        axis.contour(xx, yy, inside.astype(float), levels=[.5], colors="black", linewidths=.8)
        axis.set_title(f"Topology {name}")
        format_axis(axis)
        plotted[name] = {"x": x, "y": y, "ux": ux, "uy": uy, "speed": speed, "mask": inside}
    figure.colorbar(image, ax=axes, fraction=.025, label="|u| [m/s]")
    figure.suptitle("Fitted-topology velocity paths", fontsize=15)
    figure.savefig(FIGURES / "firedrake_streamline_gallery.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "firedrake_streamline_gallery.pdf", facecolor="white")
    plt.close(figure)
    np.savez_compressed(
        PLOTTED / "firedrake_streamline_gallery_data.npz",
        selected=np.asarray(selected), vmax=vmax,
        **{f"{name}_{key}": value for name, values in plotted.items() for key, value in values.items()},
    )


def pressure_drop(metrics):
    values = [metrics[name]["delta_p"] for name in NAMES]
    figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    bars = axis.bar(NAMES, values, color="#2878b5")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    axis.set_xlabel("Topology")
    axis.set_ylabel("Pressure range / drop proxy")
    axis.set_title("Hydraulic resistance varies across fitted topologies")
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(FIGURES / "firedrake_pressure_drop_topologies.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "firedrake_pressure_drop_topologies.pdf", facecolor="white")
    plt.close(figure)
    with (PLOTTED / "firedrake_pressure_drop_topologies.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(("topology", "delta_p")); writer.writerows(zip(NAMES, values))


def detailed_topology(cases, name="E"):
    case = cases[name]; tri = triangulation(case); xy = case["coordinates"][:, :2]
    speed = np.linalg.norm(case["velocity"], axis=1); pressure = gauge_pressure(case["pressure"])
    x, y, xx, yy, ux, uy, speed_grid, inside = interpolate_fluid(case)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    axes[0, 0].triplot(tri, color="#355c75", lw=.3)
    axes[0, 0].set_title("Fitted geometry and triangular mesh")
    image = axes[0, 1].pcolormesh(xx, yy, speed_grid, shading="auto", cmap=VELOCITY_CMAP)
    axes[0, 1].streamplot(x, y, ux, uy, color="white", density=1.25, linewidth=.65, arrowsize=.6)
    axes[0, 1].contour(xx, yy, inside.astype(float), levels=[.5], colors="black", linewidths=.8)
    axes[0, 1].set_title("Velocity magnitude and streamlines")
    figure.colorbar(image, ax=axes[0, 1], fraction=.046)
    image = axes[1, 0].tripcolor(tri, pressure, shading="gouraud", cmap=PRESSURE_CMAP)
    axes[1, 0].triplot(tri, color="0.1", lw=.15, alpha=.35)
    axes[1, 0].set_title("Gauge pressure")
    figure.colorbar(image, ax=axes[1, 0], fraction=.046)
    step = max(1, len(xy) // 300)
    axes[1, 1].tripcolor(tri, speed, shading="gouraud", cmap=VELOCITY_CMAP)
    axes[1, 1].quiver(
        xy[::step, 0], xy[::step, 1], case["velocity"][::step, 0], case["velocity"][::step, 1],
        color="white", scale=2.4, width=.003,
    )
    axes[1, 1].set_title("Subsampled velocity vectors")
    for axis in axes.flat:
        format_axis(axis)
    figure.suptitle(f"Detailed fitted CFD solution: topology {name}", fontsize=15)
    figure.savefig(FIGURES / "firedrake_detailed_topology.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "firedrake_detailed_topology.pdf", facecolor="white")
    plt.close(figure)


def fitted_mesh_examples(cases):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, name in zip(axes, ("B", "E")):
        tri = triangulation(cases[name])
        axis.triplot(tri, color="#355c75", lw=.28)
        axis.set_title(f"Topology {name}")
        format_axis(axis)
    figure.suptitle("Geometry-conforming Taylor-Hood meshes", fontsize=15)
    figure.savefig(FIGURES / "firedrake_fitted_mesh_examples.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "firedrake_fitted_mesh_examples.pdf", facecolor="white")
    plt.close(figure)


def plot_3d(cases, metrics):
    case_path = DATA / "topology_E_3d"
    with np.load(case_path / "fields_3d.npz") as archive:
        xyz = archive["coordinates"].copy(); velocity = archive["velocity"].copy(); pressure = archive["pressure"].copy()
    three = json.loads((case_path / "metrics.json").read_text())
    speed = np.linalg.norm(velocity, axis=1)
    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    axis = figure.add_subplot(221, projection="3d")
    sample = np.arange(0, len(xyz), max(1, len(xyz) // 6000))
    image3d = axis.scatter(xyz[sample, 0], xyz[sample, 1], xyz[sample, 2], c=speed[sample], s=3, cmap=VELOCITY_CMAP, alpha=.55)
    axis.set_title("A. 3D fitted fluid volume"); axis.set_xlabel("x"); axis.set_ylabel("y"); axis.set_zlabel("z")
    figure.colorbar(image3d, ax=axis, shrink=.65, label="|u|")
    z_mid = np.unique(xyz[:, 2])[len(np.unique(xyz[:, 2])) // 2]
    for panel, field, cmap, title, z_value in (
        (222, speed, VELOCITY_CMAP, "B. Midplane velocity", z_mid),
        (223, gauge_pressure(pressure), PRESSURE_CMAP, "C. Midplane pressure", z_mid),
    ):
        axis = figure.add_subplot(panel)
        mask = np.isclose(xyz[:, 2], z_value)
        image = axis.scatter(xyz[mask, 0], xyz[mask, 1], c=field[mask], s=12, cmap=cmap)
        axis.set_aspect("equal"); axis.set_title(title); axis.set_xlabel("x"); axis.set_ylabel("y")
        figure.colorbar(image, ax=axis)
    axis = figure.add_subplot(224)
    mask = np.isclose(xyz[:, 0], .064, atol=.003)
    image = axis.scatter(xyz[mask, 1], xyz[mask, 2], c=speed[mask], s=18, cmap=VELOCITY_CMAP)
    axis.set_title("D. Through-depth velocity variation"); axis.set_xlabel("y"); axis.set_ylabel("z")
    figure.colorbar(image, ax=axis, label="|u|")
    note = (
        f"{three['velocity_dofs']:,} velocity DOFs | {three['pressure_dofs']:,} pressure DOFs | "
        f"{three['solve_time']:.1f} s | mass {three['mass_imbalance']:.2e}"
    )
    figure.suptitle("Topology E: genuine three-dimensional Navier-Stokes flow\n" + note, fontsize=14)
    figure.savefig(FIGURES / "firedrake_3d_flow_topology_E.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "firedrake_3d_flow_topology_E.pdf", facecolor="white")
    plt.close(figure)

    two = cases["E"]
    two_xy = two["coordinates"][:, :2]; two_tri = triangulation(two)
    two_speed = np.linalg.norm(two["velocity"], axis=1); two_pressure = gauge_pressure(two["pressure"])
    mid = np.isclose(xyz[:, 2], z_mid)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].tripcolor(two_tri, two_speed, shading="gouraud", cmap=VELOCITY_CMAP); axes[0, 0].set_title("2D |u|")
    image = axes[0, 1].scatter(xyz[mid, 0], xyz[mid, 1], c=speed[mid], s=12, cmap=VELOCITY_CMAP); axes[0, 1].set_title("3D midplane |u|")
    axes[1, 0].tripcolor(two_tri, two_pressure, shading="gouraud", cmap=PRESSURE_CMAP); axes[1, 0].set_title("2D pressure")
    axes[1, 1].scatter(xyz[mid, 0], xyz[mid, 1], c=gauge_pressure(pressure)[mid], s=12, cmap=PRESSURE_CMAP); axes[1, 1].set_title("3D midplane pressure")
    for axis in axes.flat:
        format_axis(axis)
    figure.suptitle("Topology E: 2D and completed 3D solutions")
    figure.savefig(FIGURES / "firedrake_2d_vs_3d.png", dpi=320, facecolor="white")
    figure.savefig(FIGURES / "firedrake_2d_vs_3d.pdf", facecolor="white")
    plt.close(figure)
    np.savez_compressed(
        PLOTTED / "firedrake_3d_topology_E_data.npz",
        xyz=xyz, velocity=velocity, pressure=pressure, midplane_z=z_mid,
    )
    return three


def hash_manifest(extra):
    files = []
    for path in sorted(FIGURES.iterdir()):
        if path.is_file():
            files.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    for path in sorted(PLOTTED.iterdir()):
        if path.is_file():
            files.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    (ROOT / "manifest.json").write_text(json.dumps({"style": {"geometry": "grayscale", "velocity": VELOCITY_CMAP, "pressure": PRESSURE_CMAP, "background": "white"}, **extra, "files": files}, indent=2) + "\n")


def main():
    cases, metrics = load_cases()
    vmax, pmax = gallery(cases, relative=False)
    gallery(cases, relative=True)
    streamline_gallery(cases)
    pressure_drop(metrics)
    detailed_topology(cases, "E")
    fitted_mesh_examples(cases)
    three = plot_3d(cases, metrics)
    np.savez_compressed(
        PLOTTED / "firedrake_cfd_gallery_data.npz",
        case_ids=np.asarray(NAMES), global_velocity_max=vmax, global_pressure_abs_max=pmax,
        **{f"{name}_{key}": value for name, case in cases.items() for key, value in case.items()},
    )
    hash_manifest({"cases_2d": 6, "has_completed_3d": True, "three_d_metrics": three})


if __name__ == "__main__":
    main()
