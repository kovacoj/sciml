"""Visualize the extruded chamber's genuine three-dimensional solution."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

ROOT=Path("outputs/supervisor_2026_08_24/topic2_firedrake"); CASE=ROOT/"fitted_gallery/topology_E_3d"; FIG=ROOT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
d=np.load(CASE/"fields_3d.npz"); xyz=d["coordinates"]; velocity=d["velocity"]; pressure=d["pressure"]; speed=np.linalg.norm(velocity,axis=1)
fig=plt.figure(figsize=(12,9)); ax=fig.add_subplot(221,projection="3d"); sample=np.arange(0,len(xyz),max(1,len(xyz)//5000)); scatter=ax.scatter(xyz[sample,0],xyz[sample,1],xyz[sample,2],c=speed[sample],s=4,cmap="turbo",alpha=.55); ax.set_title("A. Extruded chamber fluid volume"); ax.set(xlabel="x",ylabel="y",zlabel="z"); fig.colorbar(scatter,ax=ax,shrink=.65,label="$|u|$")
for panel,(z,title,field,cmap) in enumerate(((.008,"B. Midplane velocity",speed,"turbo"),(.004,"C. Pressure slice z=H/4",pressure,"coolwarm")),start=2):
    axis=fig.add_subplot(2,2,panel); mask=np.isclose(xyz[:,2],z,atol=1e-8); sc=axis.scatter(xyz[mask,0],xyz[mask,1],c=field[mask],s=14,cmap=cmap); axis.set_aspect("equal"); axis.set_title(title); axis.set(xlabel="x",ylabel="y"); fig.colorbar(sc,ax=axis)
axis=fig.add_subplot(224); mask=np.isclose(xyz[:,0],.064,atol=.003); axis.scatter(xyz[mask,1],xyz[mask,2],c=speed[mask],s=20,cmap="turbo"); axis.set_title("D. Cross-thickness velocity"); axis.set(xlabel="y",ylabel="z"); axis.axhline(0,color="black",lw=.6); axis.axhline(.016,color="black",lw=.6)
fig.tight_layout(); fig.savefig(FIG/"fitted_cfd_3d_chamber.png",dpi=220); fig.savefig(FIG/"fitted_cfd_3d_chamber.pdf"); plt.close(fig)

two=json.loads((ROOT/"fitted_gallery/topology_E/metrics.json").read_text()); three=json.loads((CASE/"metrics.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(9,4)); axes[0].bar(["2D fitted","3D extruded"],[two["delta_p"],three["delta_p"]],color=["#457b9d","#e76f51"]); axes[0].set_ylabel("Pressure range / drop proxy"); axes[0].set_title("Additional 3D wall friction")
zlevels=np.unique(xyz[:,2]); profile=[]
for z in zlevels:
    mask=np.isclose(xyz[:,2],z)&np.isclose(xyz[:,0],.064,atol=.003)&(xyz[:,1]>.045)&(xyz[:,1]<.083); profile.append(np.mean(speed[mask]) if np.any(mask) else np.nan)
axes[1].plot(profile,zlevels,"o-"); axes[1].set(xlabel="Mean $|u|$ near chamber center",ylabel="z",title="No-slip cross-thickness profile")
fig.tight_layout(); fig.savefig(FIG/"fitted_cfd_2d_3d_comparison.png",dpi=220); plt.close(fig)
