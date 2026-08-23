"""Extrude fitted chamber topology E and solve genuine 3D flow."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from .brinkman_mesh import INLET_MARKER,OUTLET_MARKER,WALL_MARKER
from .fitted_channel_mesh import write_fitted_topology
from .train import relative_mass_imbalance

def run(output:Path,height=.016,layers=4):
    from firedrake import (CheckpointFile,Constant,DirichletBC,ExtrudedMesh,FacetNormal,Function,FunctionSpace,Mesh,NonlinearVariationalProblem,NonlinearVariationalSolver,TestFunctions,VectorFunctionSpace,assemble,div,dot,ds_v,dx,split)
    from .weak_forms import navier_stokes_weak_form
    output.mkdir(parents=True,exist_ok=True); base=Mesh(str(write_fitted_topology(output/"base_mesh.msh","E",.004))); mesh=ExtrudedMesh(base,layers=layers,layer_height=height/layers)
    V=VectorFunctionSpace(mesh,"CG",2); Q=FunctionSpace(mesh,"CG",1); Z=V*Q; state=Function(Z,name="topology_E_3d"); u,p=split(state); v,q=TestFunctions(Z)
    bcs=[DirichletBC(Z.sub(0),Constant((.1,0,0)),INLET_MARKER),DirichletBC(Z.sub(0),Constant((0,0,0)),WALL_MARKER),DirichletBC(Z.sub(0),Constant((0,0,0)),"top"),DirichletBC(Z.sub(0),Constant((0,0,0)),"bottom")]
    elapsed=0.; stages=[]
    for beta in (0.,1.):
        residual=navier_stokes_weak_form(u,p,v,q,Constant(.01),Constant(beta),dx); solver=NonlinearVariationalSolver(NonlinearVariationalProblem(residual,state,bcs=bcs),solver_parameters={"snes_type":"newtonls","snes_rtol":1e-9,"snes_atol":1e-10,"snes_max_it":30,"ksp_type":"preonly","pc_type":"lu","pc_factor_mat_solver_type":"mumps","mat_type":"aij"})
        started=time.perf_counter(); solver.solve(); seconds=time.perf_counter()-started; elapsed+=seconds; stages.append({"beta":beta,"seconds":seconds,"reason":int(solver.snes.getConvergedReason())})
    velocity,pressure=state.subfunctions; residual=navier_stokes_weak_form(u,p,v,q,Constant(.01),Constant(1),dx); assembled=assemble(residual,bcs=bcs)
    with assembled.dat.vec_ro as vec: weak=float(vec.norm())
    normal=FacetNormal(mesh); qi=float(assemble(dot(velocity,normal)*ds_v(INLET_MARKER))); qo=float(assemble(dot(velocity,normal)*ds_v(OUTLET_MARKER)))
    V1=VectorFunctionSpace(mesh,"CG",1); Q1=FunctionSpace(mesh,"CG",1); uv=Function(V1).interpolate(velocity); pv=Function(Q1).interpolate(pressure); coords=mesh.coordinates.dat.data_ro.copy(); values=uv.dat.data_ro.copy(); pressure_values=pv.dat.data_ro.copy()
    np.savez_compressed(output/"fields_3d.npz",coordinates=coords,velocity=values,pressure=pressure_values)
    with CheckpointFile(str(output/"solution_3d.h5"),"w") as c: c.save_mesh(mesh); c.save_function(velocity,name="velocity"); c.save_function(pressure,name="pressure")
    report={"topology_id":"E_extruded_3d","height":height,"layers":layers,"mesh_cells":int(mesh.cell_set.size),"solve_time":elapsed,"stages":stages,"Q_in":qi,"Q_out":qo,"mass_imbalance":relative_mass_imbalance(qi,qo),"divergence_L2":float(assemble(div(velocity)**2*dx)**.5),"u_max":float(np.linalg.norm(values,axis=1).max()),"delta_p":float(np.ptp(pressure_values)),"weak_residual":weak,"velocity_dofs":V.dim(),"pressure_dofs":Q.dim()}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--height",type=float,default=.016); p.add_argument("--layers",type=int,default=4); a=p.parse_args(); run(a.output,a.height,a.layers)
if __name__=="__main__":main()
