"""Summarize all runs under exp/beat: success curve per run group."""
import csv, glob, json, os, sys
import numpy as np

ROOT = "/workspace/svgd_qc/exp/beat"
rows = []
for ev in sorted(glob.glob(f"{ROOT}/*/*/*/*/eval.csv")):
    d = os.path.dirname(ev)
    group = ev.split("/")[len(ROOT.split("/")) + 1]
    try:
        recs = list(csv.DictReader(open(ev)))
        v = [float(r["success"]) for r in recs if r.get("success")]
        steps = [int(float(r["step"])) for r in recs if r.get("success")]
    except Exception:
        continue
    if not v:
        continue
    fl = {}
    fp = os.path.join(d, "flags.json")
    if os.path.exists(fp):
        fl = json.load(open(fp))
    env = fl.get("env_name", "?")
    task = env.split("task")[-1].split("-")[0] if "task" in env else "?"
    rows.append(dict(group=group, task="t" + task, seed=fl.get("seed", "?"),
                     n=len(v), step=max(steps) if steps else 0,
                     last3=float(np.mean(v[-3:])), last5=float(np.mean(v[-5:])),
                     best=max(v), curve=v))

rows.sort(key=lambda r: (r["task"], r["group"]))
print(f"{'run group':28s} {'task':4s} {'sd':>2s} {'n':>3s} {'step':>7s} {'last3':>6s} {'last5':>6s} {'best':>5s}  curve")
for r in rows:
    print(f"{r['group']:28s} {r['task']:4s} {str(r['seed']):>2s} {r['n']:3d} {r['step']:7d} "
          f"{r['last3']:6.3f} {r['last5']:6.3f} {r['best']:5.2f}  " + " ".join(f"{x:.2f}" for x in r["curve"]))
if not rows:
    print("(no eval.csv found yet)")
