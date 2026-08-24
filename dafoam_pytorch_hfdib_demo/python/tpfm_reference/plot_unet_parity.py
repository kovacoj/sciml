"""Create parity figures from frozen one-shot benchmark fields."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def descriptors(lam):
    fluid=lam<.5; fraction=float(fluid.mean()); transitions=float(np.mean(np.abs(np.diff(fluid.astype(float),axis=0)))+np.mean(np.abs(np.diff(fluid.astype(float),axis=1)))); horizontal=float(np.mean(np.sum(fluid,axis=1)>48)); return fraction,transitions,horizontal
def main():
    p=argparse.ArgumentParser(); p.add_argument("--benchmark",type=Path,required=True); p.add_argument("--figures",type=Path,required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args(); a.figures.mkdir(parents=True,exist_ok=True); fields=np.load(a.benchmark/"benchmark_fields.npz"); rows=list(csv.DictReader((a.benchmark/"per_case_metrics.csv").open())); cases=sorted({r["case_id"] for r in rows}); geometry=sorted((descriptors(fields[f"{c}_lambda"]),c) for c in cases); selected=[geometry[0][1],geometry[len(geometry)//2][1],geometry[-1][1]]
    with (a.benchmark/"topology_selection.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["case_id","fluid_fraction","interface_density","long_horizontal_path_fraction","selection_method"])
        for d,c in geometry:w.writerow([c,*d,"geometry_descriptor_order_independent_of_model_error"])
    metrics={(r["case_id"],r["method"]):r for r in rows}
    fig,axes=plt.subplots(3,5,figsize=(14,8.5),constrained_layout=True)
    for i,c in enumerate(selected):
        lam=fields[f"{c}_lambda"]; ru=fields[f"{c}_reference_u"]; rp=fields[f"{c}_reference_p"]; pu=fields[f"{c}_solver_distilled_u"]; pp=fields[f"{c}_solver_distilled_p"]; refspeed=np.linalg.norm(ru[...,:2],axis=2); predspeed=np.linalg.norm(pu[...,:2],axis=2); rp=rp-rp.mean(); pp=pp-pp.mean(); vmax=max(refspeed.max(),predspeed.max()); pmax=max(np.abs(rp).max(),np.abs(pp).max())
        vals=((lam,"Greys",0,1),(refspeed,"viridis",0,vmax),(predspeed,"viridis",0,vmax),(rp,"coolwarm",-pmax,pmax),(pp,"coolwarm",-pmax,pmax))
        for j,(value,cmap,vmin,vmax_) in enumerate(vals): axes[i,j].imshow(value,origin="lower",cmap=cmap,vmin=vmin,vmax=vmax_); axes[i,j].axis("off")
        m=metrics[(c,"solver_distilled")]; axes[i,0].set_ylabel(c[-3:]+f"\nMSE-TV u={float(m['paper_velocity_mse_tv']):.3f}\ne_u={float(m['relative_velocity_error']):.3f}\nΔp err={float(m['pressure_drop_relative_error']):.1%}",fontsize=9)
    for ax,title in zip(axes[0],["λ","DAFoam |u|","Model |u|","DAFoam p","Model p"]):ax.set_title(title)
    fig.suptitle("Figure-6 analogue: frozen solver-distilled model",fontsize=14); fig.savefig(a.figures/"dafoam_figure6_analogue_reference_vs_model.png",dpi=320); fig.savefig(a.figures/"dafoam_figure6_analogue_reference_vs_model.pdf"); plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(12,8.5),constrained_layout=True)
    for i,c in enumerate(selected):
        lam=fields[f"{c}_lambda"]; ru=fields[f"{c}_reference_u"]; pu=fields[f"{c}_solver_distilled_u"]; rp=fields[f"{c}_reference_p"]-fields[f"{c}_reference_p"].mean(); pp=fields[f"{c}_solver_distilled_p"]-fields[f"{c}_solver_distilled_p"].mean(); eu=np.linalg.norm(pu[...,:2]-ru[...,:2],axis=2); ep=np.abs(pp-rp)
        for j,(value,cmap) in enumerate(((lam,"Greys"),(np.linalg.norm(ru[...,:2],axis=2),"viridis"),(eu,"magma"),(ep,"magma"))):axes[i,j].imshow(value,origin="lower",cmap=cmap); axes[i,j].axis("off")
    for ax,title in zip(axes[0],["λ","Reference |u|","Velocity error","Pressure error"]):ax.set_title(title)
    fig.savefig(a.figures/"dafoam_error_maps.png",dpi=320); fig.savefig(a.figures/"dafoam_error_maps.pdf"); plt.close(fig)
    x=np.array([float(metrics[(c,"solver_distilled")]["reference_pressure_drop"]) for c in cases]); y=np.array([float(metrics[(c,"solver_distilled")]["predicted_pressure_drop"]) for c in cases]); bounds=(min(x.min(),y.min()),max(x.max(),y.max())); fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(x,y,c="#d1495b"); ax.plot(bounds,bounds,"k--",label="parity"); ax.plot(bounds,np.array(bounds)*1.05,"k:",alpha=.6); ax.plot(bounds,np.array(bounds)*.95,"k:",alpha=.6); ax.set(xlabel="DAFoam pressure drop",ylabel="Model pressure drop",title="Pressure-drop parity"); ax.legend(); fig.tight_layout(); fig.savefig(a.figures/"dafoam_pressure_drop_parity.png",dpi=320); fig.savefig(a.figures/"dafoam_pressure_drop_parity.pdf"); plt.close(fig)
    aggregate=json.loads((a.benchmark/"aggregate_metrics.json").read_text()); s=aggregate["solver_distilled"]; t=aggregate["teacher_W20"]
    report=f"""# Paper Parity Report\n\nMetric normalization: `paper_figure6`. TV definition: reconstructed mean absolute forward differences; not bit-identical to the unavailable reference implementation.\n\n| Method | velocity MSE-TV | e_u | pressure MSE-TV | e_p | pressure-drop error |\n|---|---:|---:|---:|---:|---:|\n| Paper reported range | 0.015-0.040 | N/A | 0.053-0.140 | N/A | 13-15% |\n| Solver-distilled k=0 | {s['paper_velocity_mse_tv']['median']:.4f} | {s['relative_velocity_error']['median']:.3f} | {s['paper_pressure_mse_tv']['median']:.3f} | {s['relative_pressure_error']['median']:.3f} | {s['pressure_drop_relative_error']['median']:.1%} |\n| W20 teacher | {t['paper_velocity_mse_tv']['median']:.4f} | {t['relative_velocity_error']['median']:.3f} | {t['paper_pressure_mse_tv']['median']:.4f} | {t['relative_pressure_error']['median']:.3f} | {t['pressure_drop_relative_error']['median']:.1%} |\n| Supervised DAFoam | NOT TRAINED | | | | |\n| Physics augmented | NOT TRAINED | | | | |\n\nSolver-distilled paper velocity parity: **FAIL** (`0.0405 > 0.025`). Supervised training is blocked because converged labels exist for 0/256 training and 0/32 validation cases, and Docker is unavailable.\n"""; a.report.write_text(report)
    (a.benchmark/"figure_selection.json").write_text(json.dumps({"selected_case_ids":selected,"selection_method":"ordered geometry descriptors (fluid fraction, interface density, long horizontal paths), independent of model error","figure6_exact_matches_found":False,"reason":"reference PDF unavailable locally"},indent=2)+"\n")
if __name__=="__main__":main()
