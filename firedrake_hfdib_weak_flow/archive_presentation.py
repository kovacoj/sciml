"""Collect traceable presentation figures and network checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(project: Path, destination: Path, lho_root: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    networks = destination / "networks"
    data = destination / "data"
    for directory in (figures, networks, data):
        directory.mkdir(exist_ok=True)
    copies = {
        project / "outputs/final_case0/figure_case0_fields.png": figures / "firedrake_case0_fields.png",
        project / "outputs/final_case0/figure_case0_optimization.png": figures / "firedrake_case0_convergence.png",
        project / "outputs/poisson_demo/poisson_result.png": figures / "poisson_weak_solution.png",
        project / "outputs/tpfm_port_audit/invariant_port_structure.png": figures / "tpfm_invariant_ports.png",
        project / "outputs/manufactured_circle_geometry/geometry_manufactured_circle.png": figures / "circular_hfdib_geometry.png",
        project / "outputs/brinkman_topologies/figure_brinkman_topologies.png": figures / "brinkman_topologies_ABC.png",
        project / "outputs/brinkman_topologies/figure_brinkman_alpha_gate.png": figures / "brinkman_alpha_gate.png",
        project / "outputs/fitted_topology_A_presentation/figure_fitted_topology_A_direct.png": figures / "fitted_topology_A_direct.png",
        lho_root / "outputs/eigenfunctions_n0_n5.png": figures / "lho_eigenfunctions_n0_n5.png",
        lho_root / "outputs/orthogonality_matrix.png": figures / "lho_orthogonality.png",
        lho_root / "outputs/disk_p24x96/disk_eigenfunctions_s0-3.png": figures / "disk_eigenfunctions_s0_3.png",
        lho_root / "outputs/disk_p24x96/disk_eigenfunctions_s4-7.png": figures / "disk_eigenfunctions_s4_7.png",
        lho_root / "outputs/disk_p24x96/disk_convergence.png": figures / "disk_eigenfunction_convergence.png",
        lho_root / "outputs/disk_p24x96/disk_energies.png": figures / "disk_eigenvalues.png",
        project / "outputs/final_case0/case0_summary.json": data / "case0_summary.json",
        project / "outputs/final_case0/case0_table.csv": data / "case0_table.csv",
        project / "outputs/poisson_demo/poisson_metrics.json": data / "poisson_metrics.json",
        project / "outputs/poisson_demo/poisson_training.csv": data / "poisson_training.csv",
        project / "outputs/tpfm_port_audit/invariant_port_structure.json": data / "invariant_port_structure.json",
        project / "outputs/manufactured_empty_gradient_check.json": data / "manufactured_empty_gradient_check.json",
        project / "outputs/manufactured_circle_gradient_check.json": data / "manufactured_circle_gradient_check.json",
        project / "outputs/brinkman_topologies/brinkman_alpha_gate.json": data / "brinkman_alpha_gate.json",
        project / "outputs/fitted_topology_A_presentation/summary.json": data / "fitted_topology_A_summary.json",
        project / "outputs/poisson_demo/poisson_model.pt": networks / "poisson_model.pt",
        project / "outputs/manufactured_empty_stokes/checkpoint_latest.pt": networks / "stokes_mlp_adam.pt",
        project / "outputs/manufactured_empty_stokes_lbfgs/checkpoint_lbfgs.pt": networks / "stokes_mlp_lbfgs_1000.pt",
        project / "outputs/manufactured_empty_stokes_lbfgs_2000/checkpoint_lbfgs.pt": networks / "stokes_mlp_lbfgs_2000.pt",
        project / "outputs/manufactured_empty_stokes_lbfgs_3000/checkpoint_lbfgs.pt": networks / "stokes_mlp_lbfgs_3000.pt",
        project / "configs/manufactured_empty_stokes.json": networks / "stokes_config.json",
        project / "outputs/manufactured_empty_stokes/normalization.json": networks / "stokes_normalization.json",
    }
    manifest_files = []
    for source, target in copies.items():
        if not source.exists():
            continue
        shutil.copy2(source, target)
        manifest_files.append({
            "path": str(target.relative_to(destination)),
            "source": str(source),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        })
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project.parent, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    manifest = {
        "git_sha": git_sha,
        "networks_reproducible": [
            "networks/poisson_model.pt",
            "networks/stokes_mlp_adam.pt",
            "networks/stokes_mlp_lbfgs_1000.pt",
            "networks/stokes_mlp_lbfgs_2000.pt",
            "networks/stokes_mlp_lbfgs_3000.pt",
        ],
        "legacy_figure_only": {
            "pattern": "figures/*eigen* and figures/lho_orthogonality.png",
            "reason": "The historical OpenFOAM/LibTorch run did not serialize model parameters.",
        },
        "files": manifest_files,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--destination", type=Path, default=Path("outputs/supervisor_2026_08_24/topic2_firedrake"))
    parser.add_argument("--lho-root", type=Path, default=Path(__file__).resolve().parent.parent / "openfoam_libtorch_lho")
    arguments = parser.parse_args()
    collect(arguments.project.resolve(), arguments.destination.resolve(), arguments.lho_root.resolve())


if __name__ == "__main__":
    main()
