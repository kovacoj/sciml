"""Benchmark frozen solver-distilled predictions with paper-style metrics."""
from __future__ import annotations
import argparse,csv,json,re,sys
from pathlib import Path
import numpy as np
PYTHON_ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(PYTHON_ROOT))
from diagnostics.warmstart_common import build_state_assembler
from state_layout import build_isothermal_layout
from tpfm_reference.paper_metrics import paper_style_metrics,physical_metrics

def foam(path,vector):
    source=path.read_text(); match=re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;",source,re.S); body=match.group(2)
    values=np.array([tuple(map(float,row.split())) for row in re.findall(r"\(([^()]+)\)",body)]) if vector else np.fromstring(body,sep=" ")
    if len(values)!=int(match.group(1)): raise ValueError(path)
    return values
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--cases",default="290,291,292,293,294,295,296,297"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    import torch; torch.set_default_dtype(torch.float64)
    from pinn.mesh_metadata import MeshMetadata
    from unet.factory import build_model
    payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=build_model(payload.get("architecture","simple"),**payload.get("model_kwargs",{})); model.load_state_dict(payload["model_state_dict"]); model.eval()
    mesh=MeshMetadata.load(str(a.dataset/"shared/mesh_metadata.npz"),str(a.dataset/"shared/mesh_metadata.json")); base=np.load(a.dataset/"shared/base_state_k0.npy"); assembler,_=build_state_assembler(mesh,build_isothermal_layout(mesh.n_cells,mesh.n_faces),base)
    rows=[]; saved={}
    for number in [int(v) for v in a.cases.split(",")]:
        case=f"topology_{number:04d}"; td=a.dataset/case; lam=np.load(td/"lambda.npy"); reference_u=foam(td/"case/5000/U",True).reshape(64,64,3); reference_p=foam(td/"case/5000/p",False).reshape(64,64)
        lam_t=torch.from_numpy(lam).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            cell,phi=model(lam_t); fluid=(lam_t[:,0]<.5).to(cell.dtype).unsqueeze(1); cell=cell*torch.cat([fluid,fluid,torch.ones_like(fluid)],dim=1); corrections=cell.squeeze(0).permute(1,2,0).reshape(-1,3); predicted=assembler.assemble(corrections,phi.squeeze(0)).detach().numpy()
        teacher=np.load(a.dataset/"solver_targets"/case/"state_k020.npy")
        saved[f"{case}_lambda"]=lam; saved[f"{case}_reference_u"]=reference_u; saved[f"{case}_reference_p"]=reference_p
        inlet=np.zeros((64,64),bool); outlet=np.zeros((64,64),bool); inlet[:,0]=lam[:,0]<.5; outlet[:,-1]=lam[:,-1]<.5
        for method,state in (("solver_distilled",predicted),("teacher_W20",teacher)):
            u=state[:3*mesh.n_cells].reshape(64,64,3); pressure=state[3*mesh.n_cells:4*mesh.n_cells].reshape(64,64); speed=np.linalg.norm(u[...,:2],axis=2); reference_speed=np.linalg.norm(reference_u[...,:2],axis=2); um=paper_style_metrics(speed,reference_speed); pm=paper_style_metrics(pressure-pressure.mean(),reference_p-reference_p.mean()); physical=physical_metrics(u,reference_u,pressure,reference_p,inlet,outlet)
            rows.append({"case_id":case,"method":method,"paper_velocity_mse":um["mse"],"paper_velocity_tv":um["tv"],"paper_velocity_mse_tv":um["mse_tv"],"paper_pressure_mse":pm["mse"],"paper_pressure_tv":pm["tv"],"paper_pressure_mse_tv":pm["mse_tv"],**physical})
            saved[f"{case}_{method}_u"]=u; saved[f"{case}_{method}_p"]=pressure
    with (a.output/"per_case_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    aggregate={}
    for method in ("solver_distilled","teacher_W20"):
        selected=[r for r in rows if r["method"]==method]; aggregate[method]={key:{"median":float(np.median([r[key] for r in selected])),"max":float(np.max([r[key] for r in selected]))} for key in ("paper_velocity_mse_tv","relative_velocity_error","paper_pressure_mse_tv","relative_pressure_error","pressure_drop_relative_error")}
    aggregate["paper_reported"]={"velocity_mse_tv_range":[.015,.040],"pressure_mse_tv_range":[.053,.140],"pressure_drop_error_range":[.13,.15]}; aggregate["metric_definition"]={"normalization":"paper_figure6 prediction min/max","tv":"reconstructed mean absolute forward differences","exact_reference_implementation":False}; (a.output/"aggregate_metrics.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    np.savez_compressed(a.output/"benchmark_fields.npz",**saved)
if __name__=="__main__":main()
