"""Regenerate SUMMARY.md from the archived run artifacts."""
import csv, glob, os, json
import numpy as np
A = os.path.dirname(os.path.abspath(__file__))
META = {  # run_group -> (task, agent label, seed)
 "R_rebrac_t1":("t1","rebrac.py",0), "R1_rebrac_t1_s1":("t1","rebrac.py",1),
 "Q_rebrac_t2":("t2","rebrac.py",0), "Q1_rebrac_t2_s1":("t2","rebrac.py",1),
 "L_rebrac_a001_t3":("t3","rebrac.py",0), "L1_rebrac_a001_t3_s1":("t3","rebrac.py",1),
 "L2_rebrac_t3_s2":("t3","rebrac.py",2),
 "M_base_t1":("t1","anq_stdfp as-is",0), "M1_base_t1_s1":("t1","anq_stdfp as-is",1),
 "O_base_t2":("t2","anq_stdfp as-is",0),
 "B_lam001_t3":("t3","anq_stdfp as-is",0), "B1_lam001_t3_s1":("t3","anq_stdfp as-is",1),
 "N_pretanh_t1":("t1","anq_stdfp + clip fix",0), "N1_pretanh_t1_s1":("t1","anq_stdfp + clip fix",1),
 "P_pretanh_t2":("t2","anq_stdfp + clip fix",0), "P1_pretanh_t2_s1":("t2","anq_stdfp + clip fix",1),
 "J_lam001_pretanh_t3":("t3","anq_stdfp + clip fix",0), "J1_pretanh_t3_s1":("t3","anq_stdfp + clip fix",1),
 "K_mani_norm_t3":("t3","mani_stdfp metric (scale-free)",0), "K1_mani_norm_t3_s1":("t3","mani_stdfp metric (scale-free)",1),
 "I_mani_weak_t3":("t3","mani_stdfp metric (unnormalised)",0), "I1_mani_weak_t3_s1":("t3","mani_stdfp metric (unnormalised)",1),
 "S_bs00_t3":("t3","anq_stdfp base_scale=0.0",0), "S1_bs00_t3_s1":("t3","anq_stdfp base_scale=0.0",1),
 "U_bs05_t3":("t3","anq_stdfp base_scale=0.5",0),
 "T_bs00_t1":("t1","anq_stdfp base_scale=0.0",0), "T1_bs00_t1_s1":("t1","anq_stdfp base_scale=0.0",1),
 "V_bs05_t1":("t1","anq_stdfp base_scale=0.5",0),
 "O1_base_t2_s1":("t2","anq_stdfp as-is",1),
 "A_lam5_t3":("t3","[retired] anq_stdfp lam=5.0",0), "D_lam5_pretanh_t3":("t3","[retired] lam=5.0 +pretanh",0),
 "C_stdfp3_t3":("t3","[retired] anq_stdfp3 (=ReBRAC)",0), "E_rebrac_a5_t3":("t3","[retired] rebrac alpha=5.0",0),
 "G_mani_data_t30":("t3","[retired] mani strong-lam",0), "F_mani_lam01_t3":("t3","[retired] mani base-anchor",0),
 "temp01_t2":("t2","[other-workspace]",0),
}
rows=[]
for g in sorted(os.listdir(f"{A}/runs")):
    f=f"{A}/runs/{g}/eval.csv"
    if not os.path.exists(f): continue
    try: v=[float(r['success']) for r in csv.DictReader(open(f)) if r.get('success')]
    except Exception: continue
    if not v: continue
    task,agent,seed = META.get(g,("?","?",0))
    rows.append(dict(group=g,task=task,agent=agent,seed=seed,n=len(v),
                     last3=np.mean(v[-3:]),last5=np.mean(v[-5:]),curve=v,done=len(v)>=11))
out=[]
out.append("# antmaze-giant experiment archive\n")
out.append("All runs: 1M offline steps, `--discount=0.995`, 50 eval episodes, eval every 100k.\n")
out.append("Machine: single RTX 3090, env `/home/sukchul/miniconda3/envs/fql`, driver `main.py`.\n")
out.append("\n## Headline table (completed runs, mean over seeds of last-5 evals)\n")
agg={}
for r in rows:
    if not r["done"] or r["agent"].startswith(("[","?")): continue
    agg.setdefault((r["agent"],r["task"]),[]).append(r["last5"])
agents=sorted({a for a,_ in agg}, key=lambda a:-np.mean([np.mean(v) for (aa,_),v in agg.items() if aa==a]))
out.append("| agent | t1 | t2 | t3 |")
out.append("|---|---|---|---|")
for a in agents:
    cells=[]
    for t in ("t1","t2","t3"):
        v=agg.get((a,t))
        cells.append(f"**{np.mean(v):.3f}** ({len(v)}sd)" if v else "—")
    out.append(f"| `{a}` | " + " | ".join(cells) + " |")
out.append("\n## All runs\n")
out.append("| run group | task | agent | seed | n | last3 | last5 | curve |")
out.append("|---|---|---|---|---|---|---|---|")
for r in sorted(rows,key=lambda r:(r["task"],r["agent"],r["seed"])):
    st="" if r["done"] else " *(running)*"
    out.append(f"| `{r['group']}` | {r['task']} | {r['agent']}{st} | {r['seed']} | {r['n']} | "
               f"{r['last3']:.3f} | {r['last5']:.3f} | {' '.join(f'{x:.2f}' for x in r['curve'])} |")
open(f"{A}/SUMMARY.md","w").write("\n".join(out)+"\n")
print(f"SUMMARY.md written: {len(rows)} runs, {sum(1 for r in rows if r['done'])} complete")
