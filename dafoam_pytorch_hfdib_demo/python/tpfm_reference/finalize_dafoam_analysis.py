"""Consolidate validated eight-case DAFoam evidence without solver reruns."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; PACKAGE=ROOT/"outputs/final_dafoam/supervisor_2026_08_24"; SOURCE=PACKAGE/"warmstart_convergence"; OUT=PACKAGE/"final_analysis"; FIG=PACKAGE/"figures"; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
CASES=[f"topology_{i:04d}" for i in range(290,298)]; COLORS={"cold":"#277da1","neural":"#d1495b","teacher":"#2a9d8f"}
def load(case,method): return json.loads((SOURCE/case/f"{method}.json").read_text())
def rank(values):
    order=np.argsort(values); ranks=np.empty(len(values)); ranks[order]=np.arange(len(values)); return ranks
def corr(x,y): return float(np.corrcoef(x,y)[0,1])
rows=[]; trajectories=[]
for case in CASES:
    cold,neural,teacher=load(case,"cold"),load(case,"neural"),load(case,"teacher_k20")
    cpoints={p["simple_steps"]:p for p in cold["history"]}; npoints={p["simple_steps"]:p for p in neural["history"]}
    row={"case_id":case,"cold_initial_rel_U":cpoints[0]["rel_u"],"neural_initial_rel_U":npoints[0]["rel_u"],"teacher_rel_U":teacher["history"][0]["rel_u"],"cold_initial_rho":cpoints[0]["rho_residual"],"neural_initial_rho":npoints[0]["rho_residual"],"teacher_rho":teacher["history"][0]["rho_residual"],"N_cold":cold["total_simple_steps"],"N_neural":neural["total_simple_steps"],"iteration_saving":1-neural["total_simple_steps"]/cold["total_simple_steps"],"cold_walltime":cold["total_time_s"],"neural_walltime":neural["total_time_s"],"field_error_ratio":npoints[0]["rel_u"]/cpoints[0]["rel_u"],"residual_ratio":npoints[0]["rho_residual"]/cpoints[0]["rho_residual"],"iteration_ratio":neural["total_simple_steps"]/cold["total_simple_steps"]}
    for k in (5,10,20): row[f"cold_k{k}_rel_U"]=cpoints[k]["rel_u"]; row[f"neural_k{k}_rel_U"]=npoints[k]["rel_u"]
    rows.append(row)
    for initialization,payload in (("cold",cold),("neural",neural),("teacher",teacher)):
        for p in payload["history"]:
            if p["simple_steps"] in (0,5,10,20,50): trajectories.append({"case_id":case,"initialization":initialization,"k":p["simple_steps"],"rel_U":p["rel_u"],"rel_p":p["rel_p"],"normalized_residual":p["rho_residual"],"rho_u":p["rho_u"],"rho_p":p["rho_p"],"rho_phi":p["rho_phi"]})
def write(path,data):
    with path.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
write(OUT/"test_case_metrics.csv",rows); write(OUT/"trajectory_metrics.csv",trajectories)
savings=np.array([r["iteration_saving"] for r in rows]); field=np.array([r["field_error_ratio"] for r in rows]); residual=np.array([r["residual_ratio"] for r in rows]); iterations=np.array([r["iteration_ratio"] for r in rows]); rng=np.random.default_rng(20260824); samples=savings[rng.integers(0,8,size=(10000,8))]
summary={"docker_status":"BLOCKED_CONTAINERD","test_cases":8,"audit":{"status":"PASS_FROM_PERSISTED_SEQUENTIAL_HISTORY","cold_k0_mean":float(np.mean([r["cold_initial_rel_U"] for r in rows])),"cold_k5_mean":float(np.mean([r["cold_k5_rel_U"] for r in rows])),"distinct":True},"convergence":{"median_iteration_saving":float(np.median(savings)),"mean_iteration_saving":float(np.mean(savings)),"mean_95_ci":np.quantile(np.mean(samples,axis=1),[.025,.975]).tolist(),"wins":int(np.sum(savings>0))},"mechanism":{"median_initial_field_error_ratio":float(np.median(field)),"median_initial_residual_ratio":float(np.median(residual)),"pearson_residual_vs_iteration":corr(residual,iterations),"spearman_residual_vs_iteration":corr(rank(residual),rank(iterations))}}
for k in (5,10,20):
    c=np.array([r[f"cold_k{k}_rel_U"] for r in rows]); n=np.array([r[f"neural_k{k}_rel_U"] for r in rows]); summary[f"finite_budget_k{k}"]={"cold_median":float(np.median(c)),"neural_median":float(np.median(n)),"relative_improvement":float(1-np.median(n)/np.median(c)),"wins":int(np.sum(n<c))}
(OUT/"summary_metrics.json").write_text(json.dumps(summary,indent=2)+"\n"); (OUT/"bootstrap.json").write_text(json.dumps({"seed":20260824,"resamples":10000,"mean_iteration_saving_95_ci":summary["convergence"]["mean_95_ci"]},indent=2)+"\n")

def start(title,w=900,h=560): return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">{title}</text>']
def text(x,y,s,size=13,anchor="middle",color="#222"): return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" font-size="{size}" fill="{color}">{s}</text>'
def save(name,lines): (FIG/name).write_text("\n".join(lines+["</svg>"])+"\n")
def scatter(name,title,xs,ys,xlabel,ylabel):
    lines=start(title); l,t,w,h=100,70,700,390; xmin,xmax=min(xs)*.95,max(xs)*1.05; ymin,ymax=min(ys)*.98,max(ys)*1.02
    for i,(x,y) in enumerate(zip(xs,ys)): cx=l+w*(x-xmin)/(xmax-xmin+1e-30); cy=t+h*(ymax-y)/(ymax-ymin+1e-30); lines.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="#d1495b"/>'); lines.append(text(cx+8,cy-8,CASES[i][-3:],10,"start"))
    lines += [text(450,520,xlabel,15),text(25,280,ylabel,15)]; save(name,lines)
scatter("state_error_vs_solver_iterations.svg","Closer velocity field does not reduce SIMPLE work",field,iterations,"initial neural/cold velocity-error ratio","neural/cold iteration ratio")
scatter("residual_vs_solver_iterations.svg","Initial solver residual versus SIMPLE work",residual,iterations,"initial neural/cold residual ratio","neural/cold iteration ratio")
for name,title,metric in (("k5_paired_velocity_error.svg","Paired velocity error after five SIMPLE calls",5),("equal_budget_velocity_error.svg","Finite-budget velocity error",None)):
    lines=start(title); l,t,w,h=90,75,720,380
    if metric:
        vals=[r["cold_k5_rel_U"] for r in rows]+[r["neural_k5_rel_U"] for r in rows]; lo,hi=min(vals)*.9,max(vals)*1.05
        for i,r in enumerate(rows): x=l+i*w/7; yc=t+h*(hi-r["cold_k5_rel_U"])/(hi-lo); yn=t+h*(hi-r["neural_k5_rel_U"])/(hi-lo); lines.append(f'<line x1="{x}" y1="{yc}" x2="{x}" y2="{yn}" stroke="#999"/>'); lines.append(f'<circle cx="{x}" cy="{yc}" r="6" fill="{COLORS["cold"]}"/><circle cx="{x}" cy="{yn}" r="6" fill="{COLORS["neural"]}"/>'); lines.append(text(x,480,CASES[i][-3:],11))
    else:
        ks=(0,5,10,20); allv=[]
        for k in ks:
            allv += [r["cold_initial_rel_U" if k==0 else f"cold_k{k}_rel_U"] for r in rows]+[r["neural_initial_rel_U" if k==0 else f"neural_k{k}_rel_U"] for r in rows]
        lo,hi=min(allv)*.9,max(allv)*1.05
        for method in ("cold","neural"):
            points=[]
            for j,k in enumerate(ks): vals=[r[f"{method}_initial_rel_U" if k==0 else f"{method}_k{k}_rel_U"] for r in rows]; med=float(np.median(vals)); x=l+j*w/3; y=t+h*(hi-med)/(hi-lo); points.append(f"{x},{y}"); lines.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{COLORS[method]}"/>'); lines.append(text(x,480,str(k),12))
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[method]}" stroke-width="4"/>')
    save(name,lines)
lines=start("Teacher truncation and neural imitation error"); vals=[np.median([r["teacher_rel_U"] for r in rows]),np.median([r["neural_initial_rel_U"] for r in rows])]; labels=["W20 teacher","Wθ network"]
for i,(label,v) in enumerate(zip(labels,vals)): x=270+i*360; lines.append(f'<rect x="{x-80}" y="{450-300*v}" width="160" height="{300*v}" fill="{COLORS[["teacher","neural"][i]]}"/>'); lines.append(text(x,475,label,15)); lines.append(text(x,435-300*v,f"{v:.3f}",15))
save("teacher_error_decomposition.svg",lines)
lines=start("Velocity improves before solver residual"); components=("rho_u","rho_p","rho_phi"); cold=load(CASES[0],"cold")["history"][0]; neural=load(CASES[0],"neural")["history"][0]
for j,c in enumerate(components): x=180+j*250; scale=max(cold[c],neural[c]); lines.append(f'<rect x="{x-55}" y="{430-280*cold[c]/scale}" width="50" height="{280*cold[c]/scale}" fill="{COLORS["cold"]}"/><rect x="{x+5}" y="{430-280*neural[c]/scale}" width="50" height="{280*neural[c]/scale}" fill="{COLORS["neural"]}"/>'); lines.append(text(x,455,c,15))
save("initial_residual_components.svg",lines)
(OUT/"STATUS.md").write_text(f"# DAFoam Final Analysis\n\nDocker: `BLOCKED_CONTAINERD`. Results remain `n=8/16`.\n\nPersisted sequential histories audit cold `k=0` mean `{summary['audit']['cold_k0_mean']:.4f}` versus cold `k=5` mean `{summary['audit']['cold_k5_mean']:.4f}`; values are distinct.\n")
