"""Audit sequential cold/neural SIMPLE trajectories on frozen held-out cases."""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np
PYTHON_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PYTHON_ROOT))
from common import PROJECT_ROOT,hfdib_signed_distance_options
from dafoam_bridge import DAFoamResidualBridge
from diagnostics.warmstart_common import build_state_assembler,predict_full_state,relative_field_errors
from state_layout import build_isothermal_layout
from unet.generate_case import write_signed_distance_file
from tpfm_reference.evaluate_warmstart_convergence import load_model,residual_metrics,atomic_json

STEPS=(0,1,2,5,10,20)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--cases",default="290,291"); p.add_argument("--output",required=True); a=p.parse_args()
    ds=Path(a.dataset); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); model=load_model(Path(a.checkpoint))
    from pinn.mesh_metadata import MeshMetadata
    mesh=MeshMetadata.load(str(ds/"shared/mesh_metadata.npz"),str(ds/"shared/mesh_metadata.json")); w0=np.load(ds/"shared/base_state_k0.npy"); assembler,_=build_state_assembler(mesh,build_isothermal_layout(mesh.n_cells,mesh.n_faces),w0)
    rows=[]; state_checks=[]
    from mpi4py import MPI
    for index in [int(v) for v in a.cases.split(",")]:
        case=f"topology_{index:04d}"; td=ds/case; case_dir=td/"case"; subprocess.run(["blockMesh","-case",str(case_dir)],check=True,capture_output=True); write_signed_distance_file(str(case_dir),np.load(td/"signed_distance.npy")); os.chdir(case_dir)
        bridge=DAFoamResidualBridge(str(case_dir),hfdib_signed_distance_options(str(case_dir),inlet_patches=["inletLower","inletUpper"],outlet_patches=["outletLower","outletUpper"]),comm=MPI.COMM_SELF); bridge.solver(); reference=np.ascontiguousarray(bridge.solver.getStates().copy(),dtype=np.float64); cold_norm=float(np.linalg.norm(bridge.residual(w0))); neural,_=predict_full_state(model,np.load(td/"lambda.npy"),assembler)
        for label,start in (("cold",w0),("neural",neural)):
            state=start.copy(); previous=state.copy()
            for k in range(21):
                if k>0:
                    state=bridge.simple_step(state); state_checks.append({"case_id":case,"initialization":label,"k":k,"step_delta_norm":float(np.linalg.norm(state-previous))}); previous=state.copy()
                if k in STEPS:
                    residual=bridge.residual(state); fields=relative_field_errors(state,reference,mesh.n_cells); blocks=residual_metrics(residual,cold_norm,mesh.n_cells); rows.append({"case_id":case,"initialization":label,"k":k,**fields,**blocks})
        del bridge
    assert all(item["step_delta_norm"]>1e-14 for item in state_checks)
    with (out/"equal_budget_audit.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(out/"equal_budget_audit.json",{"rows":rows,"state_checks":state_checks}); cold0=[r["rel_u"] for r in rows if r["initialization"]=="cold" and r["k"]==0]; cold5=[r["rel_u"] for r in rows if r["initialization"]=="cold" and r["k"]==5]; atomic_json(out/"status.json",{"status":"PASS","cold_k0_mean":float(np.mean(cold0)),"cold_k5_mean":float(np.mean(cold5)),"distinct":not np.isclose(np.mean(cold0),np.mean(cold5))})
if __name__=="__main__":main()
