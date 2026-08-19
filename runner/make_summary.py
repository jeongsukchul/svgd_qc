"""Generate SUMMARY.md for the antmaze-giant anq_rfs / anq_stdfp / rebrac runs."""
import csv, glob, json, os
import numpy as np

ROOT = "/workspace/svgd_qc/exp/beat"
OUT = os.environ.get("SUMMARY_OUT", "/workspace/svgd_qc/hf_upload/SUMMARY.md")

LABEL = {
    "A_rebrac": "rebrac.py (baseline)", "W_reb": "rebrac.py (baseline)",
    "B_cur": "anq_stdfp (archive best cfg)", "E_sharp": "anq_stdfp drift_temps=0.3",
    "R01": "anq_rfs pretanh+data+kl", "R02": "anq_rfs pretanh+data+none",
    "R03": "anq_rfs pretanh+RESIDUAL anchor", "R04": "anq_rfs absolute+data+kl",
    "R05": "anq_rfs absolute+data+none", "R06": "anq_rfs absolute+RESIDUAL anchor",
    "R07": "anq_rfs action+data+kl  <-- winner", "R08": "anq_rfs pretanh live-base",
    "W_rfs": "anq_rfs action+data+kl  <-- winner", "V_abs": "anq_rfs absolute+data+none",
}
def label(g):
    for k in sorted(LABEL, key=len, reverse=True):
        if g.startswith(k):
            return LABEL[k]
    return g.split("_")[0]

rows = []
for ev in sorted(glob.glob(f"{ROOT}/*/*/*/*/eval.csv")):
    d = os.path.dirname(ev)
    group = ev.split("/")[len(ROOT.split("/")) + 1]
    try:
        recs = list(csv.DictReader(open(ev)))
        v = [float(r["success"]) for r in recs if r.get("success")]
    except Exception:
        continue
    if not v:
        continue
    fl = json.load(open(os.path.join(d, "flags.json"))) if os.path.exists(os.path.join(d, "flags.json")) else {}
    env = fl.get("env_name", "")
    task = "t" + env.split("task")[-1].split("-")[0] if "task" in env else "t?"
    rows.append(dict(group=group, task=task, seed=fl.get("seed", 0), label=label(group),
                     n=len(v), eval_eps=fl.get("eval_episodes", 50),
                     last3=float(np.mean(v[-3:])), last5=float(np.mean(v[-5:])),
                     best=max(v), curve=v, done=len(v) >= 10))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = []
L.append("# antmaze-giant: beating ReBRAC with a unified latent+residual actor\n")
L.append("All runs: `antmaze-giant-navigate-singletask-task{1,2,3}-v0`, 1M offline steps,")
L.append("`--discount=0.995`, eval every 100k, `horizon_length=1`, `best_of_n=1`.\n")
L.append("Machine: 8x RTX 4080 SUPER. Driver `main.py` with `--offline_scan_chunk=25`")
L.append("(fuses 25 updates into one `lax.scan` dispatch; verified bit-identical to the")
L.append("per-step loop, `max|diff| = 0`).\n")

agg = {}
for r in rows:
    if r["done"]:
        agg.setdefault((r["label"], r["task"]), []).append(r["last5"])
L.append("\n## Headline (mean over seeds of last-5 evals)\n")
L.append("| agent | t1 | t2 | t3 |")
L.append("|---|---|---|---|")
labels = sorted({a for a, _ in agg}, key=lambda a: -np.mean([np.mean(v) for (aa, _), v in agg.items() if aa == a]))
for a in labels:
    cells = []
    for t in ("t1", "t2", "t3"):
        v = agg.get((a, t))
        cells.append(f"**{np.mean(v):.3f}** ({len(v)}sd)" if v else "—")
    L.append(f"| `{a}` | " + " | ".join(cells) + " |")

L.append("\n## All runs\n")
L.append("| run group | task | agent | sd | evals | eval eps | last3 | last5 | best | curve |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["task"], -r["last5"])):
    st = "" if r["done"] else " *(running)*"
    L.append(f"| `{r['group']}`{st} | {r['task']} | {r['label']} | {r['seed']} | {r['n']} | "
             f"{r['eval_eps']} | {r['last3']:.3f} | {r['last5']:.3f} | {r['best']:.2f} | "
             + " ".join(f"{x:.2f}" for x in r["curve"]) + " |")
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(rows)} runs)")
