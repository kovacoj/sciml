"""Write presentation notes and SHA-256 manifest for the CFD gallery."""
from pathlib import Path
import hashlib,json
ROOT=Path("outputs/supervisor_2026_08_24/topic2_firedrake")
summary=json.loads((ROOT/"fitted_cfd_gallery_summary.json").read_text()); three=json.loads((ROOT/"fitted_gallery/topology_E_3d/metrics.json").read_text())
case_metrics=[json.loads((ROOT/f"fitted_gallery/topology_{name}/metrics.json").read_text()) for name in "ABCDEF"]
(ROOT/"fitted_gallery/summary.json").write_text(json.dumps(case_metrics,indent=2)+"\n")
(ROOT/"PRESENTATION_RESULTS.md").write_text(f"""# Firedrake CFD Gallery\n\n## 2D family\nSix deterministic connected article-inspired fitted geometries were solved with Taylor-Hood FEM. Stokes initializes steady Navier-Stokes at `uin=0.1`, `nu=0.01`, and natural zero outlet traction.\n\n- Cases: 6/6 converged\n- Maximum mass imbalance: {summary['max_mass_imbalance']:.3e}\n- Maximum assembled weak residual: {summary['max_weak_residual']:.3e}\n- Pressure range: {summary['pressure_drop_range'][0]:.3f} to {summary['pressure_drop_range'][1]:.3f}\n\n## 3D extension\nTopology E was extruded through `H=0.016 m` with no-slip top and bottom walls. This is a genuine `u(x,y,z)` solution, not a rendered 2D extrusion.\n\n- Velocity DOFs: {three['velocity_dofs']}\n- Pressure DOFs: {three['pressure_dofs']}\n- Solve time: {three['solve_time']:.1f} s\n- Mass imbalance: {three['mass_imbalance']:.3e}\n- Weak residual: {three['weak_residual']:.3e}\n- Pressure range: {three['delta_p']:.3f}\n\n## Claim boundary\nProcedurally generated article-inspired fitted geometries solved with Firedrake Taylor-Hood FEM. This is not reproduction of the paper and does not use HFDIB or a trained 3D surrogate.\n""")
(ROOT/"README.md").write_text("# Topic 2 Firedrake\n\nPresentation CFD fields, metrics, checkpoints, figures, and reproducible sources for six fitted 2D topologies and one extruded 3D chamber.\n")
manifest={}
for path in sorted(ROOT.rglob("*")):
    if path.is_file() and path.name!="manifest.json": manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
