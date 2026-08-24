"""Build additive SVG figures and tables from validated Stokes artifacts only."""
from __future__ import annotations
import csv,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/stokes_preconditioning_research/final_figures"
RUNS=ROOT/"outputs/stokes_preconditioning_research/runs"
OUT.mkdir(parents=True,exist_ok=True)
COLORS={"raw":"#d1495b","dual":"#277da1","correction":"#2a9d8f","block":"#7b2cbf","jacobi_ls":"#f4a261","oracle":"#222222"}

coefficient={"raw":[.520,.908,.978],"dual":[.00698,.0520,.224],"correction":[.000143,.00313,.00607]}; meshes=["16x8","32x16","64x32"]
methods=[("Raw",.978,1.015,2.37,"raw"),("Dual",.224,.711,7.49,"dual"),("Jacobi-LS",.998,1.001,4.63,"jacobi_ls"),("Block",.189,.606,12.32,"block"),("Exact correction",.00607,.429,10.16,"correction"),("Oracle",.00523,.427,1.84,"oracle")]
mlp={mesh:{name:[] for name in ("raw","dual","correction")} for mesh in meshes[:2]}
for path in sorted(RUNS.glob("mlp_*_seed*.json")):
    payload=json.loads(path.read_text()); mesh=f"{payload['problem']['nx']}x{payload['problem']['ny']}"
    for result in payload["results"]: mlp[mesh][result["loss"]].append(result["relative_velocity_error"])

def svg_start(title,w=900,h=560):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">{title}</text>']
def text(x,y,value,size=14,anchor="middle",weight="normal",color="#222"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{value}</text>'
def save(name,lines): (OUT/name).write_text("\n".join(lines+["</svg>"])+"\n")
def line_plot(name,title,series,xlabels):
    lines=svg_start(title); left,top,width,height=90,70,740,390; values=[v for ys in series.values() for v in ys]; lo,hi=math.log10(min(values))-.2,math.log10(max(values))+.2
    y=lambda v:top+height*(hi-math.log10(v))/(hi-lo); x=lambda i:left+width*i/(len(xlabels)-1)
    lines += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+height}" stroke="#333"/>',f'<line x1="{left}" y1="{top+height}" x2="{left+width}" y2="{top+height}" stroke="#333"/>']
    for i,label in enumerate(xlabels): lines.append(text(x(i),top+height+28,label))
    for method,ys in series.items():
        points=" ".join(f"{x(i)},{y(v)}" for i,v in enumerate(ys)); lines.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[method]}" stroke-width="3"/>')
        for i,v in enumerate(ys): lines.append(f'<circle cx="{x(i)}" cy="{y(v)}" r="5" fill="{COLORS[method]}"/>')
    for j,method in enumerate(series): lines.append(text(660+j*0,90+j*24,method,14,"start","bold",COLORS[method]))
    lines.append(text(28,270,"relative velocity error (log)",14,"middle")); save(name,lines)
line_plot("coefficient_mesh_sensitivity.svg","Coefficient optimization under mesh refinement",coefficient,meshes)

lines=svg_start("Five-seed coordinate-MLP mesh sensitivity"); left,top,width,height=90,75,730,380; allv=[v for m in mlp.values() for vals in m.values() for v in vals]; lo,hi=min(allv)*.8,max(allv)*1.15
for mi,mesh in enumerate(meshes[:2]):
    base=left+mi*width; lines.append(text(base+160,top+height+40,mesh,16,weight="bold"))
    for j,method in enumerate(("raw","dual","correction")):
        vals=sorted(mlp[mesh][method]); cx=base+55+j*105
        for k,v in enumerate(vals):
            cy=top+height*(hi-v)/(hi-lo); lines.append(f'<circle cx="{cx+(k-2)*5}" cy="{cy}" r="5" fill="{COLORS[method]}" opacity=".65"/>')
        median=vals[len(vals)//2]; cy=top+height*(hi-median)/(hi-lo); lines.append(f'<line x1="{cx-25}" y1="{cy}" x2="{cx+25}" y2="{cy}" stroke="{COLORS[method]}" stroke-width="4"/>'); lines.append(text(cx,top+height+18,method,12))
lines.append(text(30,270,"relative velocity error",14)); save("mlp_mesh_sensitivity.svg",lines)

lines=svg_start("Exact PDE correction closely tracks the FE-error oracle"); left,top,width,height=100,75,700,380; vals=[m[1] for m in methods]; lo,hi=-3,0.1
for i,(label,value,_,_,key) in enumerate(methods):
    x=left+i*width/(len(methods)-1); y=top+height*(hi-math.log10(value))/(hi-lo); lines.append(f'<line x1="{x}" y1="{top+height}" x2="{x}" y2="{y}" stroke="{COLORS[key]}" stroke-width="28"/>'); lines.append(text(x,top+height+22,label,11)); lines.append(text(x,y-10,f"{value:.4g}",12,weight="bold"))
lines.append(text(30,270,"velocity error (log)",14)); save("oracle_equivalence.svg",lines)

lines=svg_start("Accuracy versus optimization cost"); left,top,width,height=90,70,730,390; xmax=max(m[3] for m in methods)*1.1; lo,hi=-3,0.1
for label,value,_,seconds,key in methods:
    x=left+width*seconds/xmax; y=top+height*(hi-math.log10(value))/(hi-lo); lines.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{COLORS[key]}"/>'); lines.append(text(x+10,y-10,label,12,"start"))
lines.append(text(455,top+height+38,"wall time (s)",15)); lines.append(text(25,270,"velocity error (log)",14)); save("accuracy_vs_cost.svg",lines)

lines=svg_start("Operator preconditioning changes the neural optimization problem",1100,600); boxes=[(130,140,"Raw residual","||AU-b||²","bad conditioning"),(390,140,"Dual residual","rᵀG⁻¹r","better"),(650,140,"Exact PDE correction","Aδ = b-AU","δ = Uₕ-U"),(910,140,"Correction loss","||δ||²","oracle error without labels")]
for x,y,a,b,c in boxes:
    lines.append(f'<rect x="{x-105}" y="{y-60}" width="210" height="150" rx="16" fill="#edf6f9" stroke="#277da1" stroke-width="2"/>'); lines += [text(x,y-22,a,16,weight="bold"),text(x,y+14,b,18),text(x,y+52,c,13)]
for x in (250,510,770): lines.append(f'<path d="M{x} 155 L{x+35} 155" stroke="#333" stroke-width="3" marker-end="url(#arrow)"/>')
lines.insert(2,'<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>'); lines.append(text(550,390,"Discrete proposition: if Aₕ is invertible, Aₕδ=bₕ-AₕU implies δ=Uₕ-U.",20,weight="bold")); lines.append(text(550,435,"Exact correction changes the Hessian from 2AₕᵀAₕ to the chosen error norm 2Mₓ.",17)); save("operator_preconditioning_chain.svg",lines)

with (OUT/"coefficient_results.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["method","velocity_error","pressure_error","wall_time_s"]); w.writerows((a,b,c,d) for a,b,c,d,_ in methods)
with (OUT/"mlp_seed_results.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["mesh","method","seed_index","velocity_error"])
    for mesh,data in mlp.items():
        for method,vals in data.items():
            for i,v in enumerate(vals): w.writerow([mesh,method,i,v])
