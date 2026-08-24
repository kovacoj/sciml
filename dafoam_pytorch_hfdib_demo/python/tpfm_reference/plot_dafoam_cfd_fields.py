"""Plot converged DAFoam/HFDIB fields from persisted OpenFOAM outputs."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASES=[f"topology_{i:04d}" for i in (290,291,293,294,296,297)]
def foam(path,vector):
    source=path.read_text(); match=re.search(r"internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;",source,re.S); body=match.group(2)
    values=np.array([tuple(map(float,row.split())) for row in re.findall(r"\(([^()]+)\)",body)]) if vector else np.fromstring(body,sep=" ")
    if len(values)!=int(match.group(1)): raise ValueError(path)
    return values
def fields(dataset,case):
    td=dataset/case; lam=np.load(td/"lambda.npy"); u=foam(td/"case/5000/U",True).reshape(64,64,3); p=foam(td/"case/5000/p",False).reshape(64,64); p-=p.mean(); speed=np.linalg.norm(u[...,:2],axis=2); return lam,u,p,speed
def gallery(dataset,figures,relative=False):
    data={case:fields(dataset,case) for case in CASES}; vmax=max(v[3].max() for v in data.values()); pmax=max(np.abs(v[2]).max() for v in data.values())
    fig,axes=plt.subplots(6,3,figsize=(9,16),constrained_layout=True)
    for row,case in enumerate(CASES):
        lam,u,p,speed=data[case]
        if relative: speed=speed/(speed.max()+1e-30); p=(p-p.min())/(np.ptp(p)+1e-30); sv=(0,1); pv=(0,1); pcmap="coolwarm"
        else: sv=(0,vmax); pv=(-pmax,pmax); pcmap="coolwarm"
        axes[row,0].imshow(lam,origin="lower",cmap="Greys",vmin=0,vmax=1); imv=axes[row,1].imshow(speed,origin="lower",cmap="viridis",vmin=sv[0],vmax=sv[1]); imp=axes[row,2].imshow(p,origin="lower",cmap=pcmap,vmin=pv[0],vmax=pv[1])
        axes[row,0].set_ylabel(case[-3:],rotation=0,labelpad=14,weight="bold")
        for ax in axes[row]: ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[0,0].set_title("Topology λ"); axes[0,1].set_title("Relative velocity" if relative else "Velocity magnitude"); axes[0,2].set_title("Relative pressure" if relative else "Gauge pressure")
    fig.colorbar(imv,ax=axes[:,1],fraction=.025,pad=.02); fig.colorbar(imp,ax=axes[:,2],fraction=.025,pad=.02)
    suffix="relative" if relative else "common_scale"; fig.suptitle("DAFoam/HFDIB flow on held-out TPFM topologies",fontsize=15); fig.savefig(figures/f"dafoam_tpfm_cfd_gallery_{suffix}.png",dpi=320); fig.savefig(figures/f"dafoam_tpfm_cfd_gallery_{suffix}.pdf"); plt.close(fig)
    np.savez_compressed(figures/"dafoam_tpfm_cfd_gallery_data.npz",case_ids=np.array(CASES),lambda_fields=np.array([data[c][0] for c in CASES]),velocity=np.array([data[c][1] for c in CASES]),pressure=np.array([data[c][2] for c in CASES]),velocity_limit=vmax,pressure_limit=pmax,state_source="case/5000 converged OpenFOAM fields")
def other_figures(dataset,figures,metrics):
    data={case:fields(dataset,case) for case in CASES}; ranges={c:np.ptp(v[2]) for c,v in data.items()}; ordered=sorted(CASES,key=lambda c:ranges[c]); selected=[ordered[0],ordered[len(ordered)//2],ordered[-1]]; x=np.arange(64); y=np.arange(64)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,case in zip(axes,selected):
        lam,u,p,s=data[case]; ax.imshow(s,origin="lower",cmap="viridis"); fluid=lam<.5; ux=np.where(fluid,u[...,0],np.nan); uy=np.where(fluid,u[...,1],np.nan); ax.streamplot(x,y,ux,uy,color="white",density=1.1,linewidth=.55,arrowsize=.7); ax.contour(lam,levels=[.5],colors="black",linewidths=.6); ax.set_title(case[-3:]); ax.axis("off")
    fig.savefig(figures/"dafoam_streamlines.png",dpi=320); fig.savefig(figures/"dafoam_streamlines.pdf"); plt.close(fig)
    case=selected[1]; lam,u,p,s=data[case]; fig,axes=plt.subplots(2,2,figsize=(9,8),constrained_layout=True); axes[0,0].imshow(lam,origin="lower",cmap="Greys"); axes[0,0].set_title("Topology λ"); axes[0,1].imshow(s,origin="lower",cmap="viridis"); axes[0,1].streamplot(x,y,np.where(lam<.5,u[...,0],np.nan),np.where(lam<.5,u[...,1],np.nan),color="white",density=1); axes[0,1].set_title("|u| + streamlines"); axes[1,0].imshow(p,origin="lower",cmap="coolwarm"); axes[1,0].set_title("Gauge pressure"); axes[1,1].imshow(s,origin="lower",cmap="viridis",alpha=.8); skip=5; axes[1,1].quiver(x[::skip],y[::skip],u[::skip,::skip,0],u[::skip,::skip,1],color="white",scale=1.5); axes[1,1].set_title("Velocity vectors")
    for ax in axes.ravel(): ax.axis("off"); fig.savefig(figures/"dafoam_detailed_flow_case.png",dpi=320); fig.savefig(figures/"dafoam_detailed_flow_case.pdf"); plt.close(fig)
    rows=list(csv_dict(metrics)); fig,ax=plt.subplots(figsize=(6,6)); xs=np.array([float(r["cold_k5_rel_U"]) for r in rows]); ys=np.array([float(r["neural_k5_rel_U"]) for r in rows]); bounds=(min(xs.min(),ys.min())*.95,max(xs.max(),ys.max())*1.05); ax.scatter(xs,ys,c="#d1495b",s=55); ax.plot(bounds,bounds,"--",color="black"); ax.set(xlim=bounds,ylim=bounds,xlabel="Cold k=5 relative velocity error",ylabel="Neural k=5 relative velocity error",title="Finite-budget velocity accuracy"); fig.tight_layout(); fig.savefig(figures/"dafoam_k5_error_paired.png",dpi=320); fig.savefig(figures/"dafoam_k5_error_paired.pdf"); plt.close(fig)
    (figures/"figure_selection.json").write_text(json.dumps({"streamline_selection_rule":"minimum, median, maximum converged gauge-pressure range","cases":selected,"detailed_case":case,"warmstart_flow_comparison":"NOT_GENERATED_NO_DIRECT_AUDIT_T5_STATES"},indent=2)+"\n")
def csv_dict(path):
    import csv
    with path.open() as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--figures",type=Path,required=True); p.add_argument("--metrics",type=Path,required=True); a=p.parse_args(); a.figures.mkdir(parents=True,exist_ok=True); gallery(a.dataset,a.figures,False); gallery(a.dataset,a.figures,True); other_figures(a.dataset,a.figures,a.metrics)
if __name__=="__main__":main()
