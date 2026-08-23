"""Solve deterministic article-inspired fitted CFD topologies."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from .brinkman_mesh import INLET_MARKER, OUTLET_MARKER, WALL_MARKER
from .fitted_channel_mesh import write_fitted_topology
from .train import relative_mass_imbalance

def solve_case(name: str, output: Path, nominal_h: float=.002) -> dict:
    from firedrake import (CheckpointFile, Constant, DirichletBC, FacetNormal, Function,
        FunctionSpace, Mesh, NonlinearVariationalProblem, NonlinearVariationalSolver,
        TestFunctions, VectorFunctionSpace, assemble, div, dot, ds, dx, split, sqrt)
    from .weak_forms import navier_stokes_weak_form
    output.mkdir(parents=True,exist_ok=True)
    mesh=Mesh(str(write_fitted_topology(output/"mesh.msh",name,nominal_h)))
    V=VectorFunctionSpace(mesh,"CG",2); Q=FunctionSpace(mesh,"CG",1); Z=V*Q
    state=Function(Z,name=f"topology_{name}"); u,p=split(state); v,q=TestFunctions(Z)
    bcs=[DirichletBC(Z.sub(0),Constant((.1,0)),INLET_MARKER),DirichletBC(Z.sub(0),Constant((0,0)),WALL_MARKER)]
    elapsed=0.; stages=[]
    for beta in (0.,1.):
        residual=navier_stokes_weak_form(u,p,v,q,Constant(.01),Constant(beta),dx)
        solver=NonlinearVariationalSolver(NonlinearVariationalProblem(residual,state,bcs=bcs),solver_parameters={"snes_type":"newtonls","snes_rtol":1e-10,"snes_atol":1e-11,"snes_max_it":30,"ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type":"mumps","mat_type":"aij"})
        started=time.perf_counter(); solver.solve(); stage_time=time.perf_counter()-started; elapsed+=stage_time
        stages.append({"beta":beta,"seconds":stage_time,"reason":int(solver.snes.getConvergedReason())})
    velocity,pressure=state.subfunctions; normal=FacetNormal(mesh)
    residual=navier_stokes_weak_form(u,p,v,q,Constant(.01),Constant(1.),dx)
    assembled=assemble(residual,bcs=bcs)
    with assembled.dat.vec_ro as vec: weak=float(vec.norm())
    qi=float(assemble(dot(velocity,normal)*ds(INLET_MARKER))); qo=float(assemble(dot(velocity,normal)*ds(OUTLET_MARKER)))
    velocity_p1=Function(VectorFunctionSpace(mesh,"CG",1)).interpolate(velocity)
    pressure_p1=Function(Q).interpolate(pressure)
    coords=mesh.coordinates.dat.data_ro.copy(); cells=mesh.coordinates.function_space().cell_node_map().values.copy()
    values_u=velocity_p1.dat.data_ro.copy(); values_p=pressure_p1.dat.data_ro.copy()
    np.savez_compressed(output/"fields.npz",coordinates=coords,cells=cells,velocity=values_u,pressure=values_p)
    with CheckpointFile(str(output/"solution.h5"),"w") as checkpoint:
        checkpoint.save_mesh(mesh); checkpoint.save_function(velocity,name="velocity"); checkpoint.save_function(pressure,name="pressure")
    speed=np.linalg.norm(values_u,axis=1)
    report={"topology_id":name,"mesh_cells":int(mesh.cell_set.size),"mesh_vertices":len(coords),"solve_type":"steady_navier_stokes_stokes_initialized","solve_time":elapsed,"stages":stages,"Q_in":qi,"Q_out":qo,"mass_imbalance":relative_mass_imbalance(qi,qo),"divergence_L2":float(assemble(div(velocity)**2*dx)**.5),"u_max":float(speed.max()),"u_mean":float(speed.mean()),"delta_p":float(values_p.max()-values_p.min()),"weak_residual":weak,"velocity_dofs":V.dim(),"pressure_dofs":Q.dim()}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n"); return report

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--topologies",default="A,B,C,D,E,F"); parser.add_argument("--nominal-h",type=float,default=.002)
    args=parser.parse_args(); reports=[]
    for name in args.topologies.split(","):
        reports.append(solve_case(name.strip(),args.output/f"topology_{name.strip()}",args.nominal_h))
    (args.output/"summary.json").write_text(json.dumps(reports,indent=2)+"\n")
if __name__=="__main__": main()
