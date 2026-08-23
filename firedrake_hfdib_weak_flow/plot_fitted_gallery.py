"""Create article-style figures from fitted-gallery FEM outputs."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

ROOT=Path("outputs/supervisor_2026_08_24/topic2_firedrake"); DATA=ROOT/"fitted_gallery"; FIG=ROOT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
names="ABCDEF"
cases={}; metrics=[]
for name in names:
    case=np.load(DATA/f"topology_{name}"/"fields.npz"); cases[name]=case
    metrics.append(json.loads((DATA/f"topology_{name}"/"metrics.json").read_text()))
speed_max=max(np.linalg.norm(cases[n]["velocity"],axis=1).max() for n in names)
pmin=min(cases[n]["pressure"].min() for n in names); pmax=max(cases[n]["pressure"].max() for n in names)

def gallery(relative=False):
    fig,axes=plt.subplots(6,3,figsize=(9,16),constrained_layout=True)
    for row,name in enumerate(names):
        d=cases[name]; xy=d["coordinates"][:,:2]; tri=mtri.Triangulation(xy[:,0],xy[:,1],d["cells"])
        speed=np.linalg.norm(d["velocity"],axis=1); pressure=d["pressure"]-np.mean(d["pressure"])
        axes[row,0].tripcolor(tri,np.ones(len(xy)),cmap="Greys",vmin=0,vmax=1)
        if relative:
            speed=(speed-speed.min())/(np.ptp(speed)+1e-30); pressure=(pressure-pressure.min())/(np.ptp(pressure)+1e-30)
            uv=(0,1); pv=(0,1)
        else: uv=(0,speed_max); pv=(pmin,pmax)
        axes[row,1].tripcolor(tri,speed,shading="gouraud",cmap="turbo",vmin=uv[0],vmax=uv[1])
        axes[row,2].tripcolor(tri,pressure,shading="gouraud",cmap="coolwarm",vmin=pv[0],vmax=pv[1])
        axes[row,0].set_ylabel(name,rotation=0,labelpad=12,fontsize=12,weight="bold")
        for ax in axes[row]: ax.set_aspect("equal"); ax.set_xlim(0,.128); ax.set_ylim(0,.128); ax.axis("off")
    for ax,title in zip(axes[0],["Fitted geometry ($\\lambda$-like)","relative $|u|$" if relative else "$|u|$ (m/s)","relative pressure" if relative else "$p$ (kinematic)"]): ax.set_title(title)
    suffix="relative" if relative else "common_scale"; fig.savefig(FIG/f"fitted_cfd_gallery_{suffix}.png",dpi=220); fig.savefig(FIG/f"fitted_cfd_gallery_{suffix}.pdf"); plt.close(fig)
gallery(False); gallery(True)

fig,ax=plt.subplots(figsize=(7,4)); ax.bar(names,[m["delta_p"] for m in metrics],color="#277da1"); ax.set(xlabel="Topology",ylabel="Pressure range / drop proxy",title="Fixed-inlet topology resistance comparison"); fig.tight_layout(); fig.savefig(FIG/"fitted_cfd_pressure_drop.png",dpi=220); plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
for ax,name in zip(axes,"BEF"):
    d=cases[name]; xy=d["coordinates"][:,:2]; tri=mtri.Triangulation(xy[:,0],xy[:,1],d["cells"]); speed=np.linalg.norm(d["velocity"],axis=1)
    ax.tripcolor(tri,speed,shading="gouraud",cmap="turbo",vmin=0,vmax=speed_max); ax.triplot(tri,color="white",lw=.08,alpha=.25); ax.set_title(f"Topology {name}"); ax.set_aspect("equal"); ax.axis("off")
fig.savefig(FIG/"fitted_cfd_streamlines.png",dpi=220); plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True); d=cases["E"]; xy=d["coordinates"][:,:2]; tri=mtri.Triangulation(xy[:,0],xy[:,1],d["cells"])
axes[0].triplot(tri,lw=.25,color="#264653"); axes[0].set_title("Conforming triangular mesh")
axes[1].tripcolor(tri,np.linalg.norm(d["velocity"],axis=1),shading="gouraud",cmap="turbo"); axes[1].set_title("Taylor-Hood velocity solution")
for ax in axes: ax.set_aspect("equal"); ax.axis("off")
fig.savefig(FIG/"fitted_cfd_mesh_example.png",dpi=220); plt.close(fig)
columns=["topology_id","mesh_cells","mesh_vertices","solve_type","solve_time","Q_in","Q_out","mass_imbalance","divergence_L2","u_max","u_mean","delta_p","weak_residual"]
with (ROOT/"fitted_cfd_gallery_metrics.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore"); writer.writeheader(); writer.writerows(metrics)
(ROOT/"fitted_cfd_gallery_summary.json").write_text(json.dumps({"cases":len(metrics),"max_mass_imbalance":max(m["mass_imbalance"] for m in metrics),"max_weak_residual":max(m["weak_residual"] for m in metrics),"pressure_drop_range":[min(m["delta_p"] for m in metrics),max(m["delta_p"] for m in metrics)]},indent=2)+"\n")
