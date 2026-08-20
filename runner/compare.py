"""Paired anq_rfs vs rebrac comparison across tasks and seeds."""
import csv, glob, json, os, math
import numpy as np

ROOT = "/workspace/svgd_qc/exp/beat"
runs = {}
for ev in sorted(glob.glob(f"{ROOT}/*/*/*/*/eval.csv")):
    d = os.path.dirname(ev)
    g = ev.split("/")[len(ROOT.split("/")) + 1]
    try:
        v = [float(r["success"]) for r in csv.DictReader(open(ev)) if r.get("success")]
    except Exception:
        continue
    if not v:
        continue
    fp = os.path.join(d, "flags.json")
    fl = json.load(open(fp)) if os.path.exists(fp) else {}
    env = fl.get("env_name", "")
    task = "t" + env.split("task")[-1].split("-")[0] if "task" in env else "t?"
    runs[g] = dict(task=task, seed=fl.get("seed", 0), n=len(v),
                   last5=float(np.mean(v[-5:])), done=len(v) >= 10, curve=v)

ARMS = [("anq_rfs (action+data+kl)", "W_rfs"), ("rebrac.py", "W_reb"),
        ("anq_rfs (absolute+data+none)", "V_abs")]
TASKS = sorted({r["task"] for r in runs.values()})

def stat(vals):
    if not vals:
        return None
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, s, len(vals)

print("=" * 78)
print("PAIRED COMPARISON  (mean +/- sem of last-5 evals over seeds; * = incomplete)")
print("=" * 78)
print(f"{'arm':32s} " + " ".join(f"{t:>13s}" for t in TASKS))
table = {}
for label, pre in ARMS:
    cells = []
    for t in TASKS:
        vals = [r["last5"] for g, r in runs.items() if g.startswith(pre) and r["task"] == t]
        allc = [r["done"] for g, r in runs.items() if g.startswith(pre) and r["task"] == t]
        st = stat(vals)
        table[(pre, t)] = st
        if st is None:
            cells.append(f"{'--':>13s}")
        else:
            m, s, n = st
            mark = "" if all(allc) else "*"
            cells.append(f"{m:.3f}+-{s:.3f}{mark}({n})".rjust(13))
    print(f"{label:32s} " + " ".join(cells))

print("-" * 78)
print("delta (anq_rfs - rebrac), paired by seed where both exist:")
for t in TASKS:
    pairs, partial = [], 0
    for g, r in runs.items():
        if not g.startswith("W_rfs") or r["task"] != t:
            continue
        mate = [rr for gg, rr in runs.items()
                if gg.startswith("W_reb") and rr["task"] == t and rr["seed"] == r["seed"]]
        if not mate:
            continue
        # Only pair runs where BOTH arms have finished.  Pairing a finished run
        # against a run that is 2 evals in reports the other arm's warm-up as if
        # it were its final score, which manufactures a huge fake delta.
        if r["done"] and mate[0]["done"]:
            pairs.append((r["seed"], r["last5"], mate[0]["last5"]))
        else:
            partial += 1
    if not pairs:
        print(f"  {t}: no complete seed pairs yet ({partial} pair(s) still running)")
        continue
    d = [a - b for _, a, b in pairs]
    m = float(np.mean(d))
    s = float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    detail = ", ".join(f"s{sd}: {a:.3f}v{b:.3f}" for sd, a, b in sorted(pairs))
    extra = f"  (+{partial} pair(s) still running)" if partial else ""
    sig = ""
    if len(d) > 1 and s > 0:
        from scipy import stats as _st
        tt, pv = _st.ttest_1samp(d, 0.0)
        sig = f"  t={tt:.2f} p={pv:.3f}"
        # Rank-based backup: a single catastrophic run (e.g. rebrac t3 seed 5
        # scored exactly 0.000 over all 10 evals) dominates a mean-based test,
        # so also report how many seeds favour anq_rfs and a Wilcoxon p.
        wins = sum(1 for x in d if x > 0)
        sig += f"  [{wins}/{len(d)} seeds+]"
        if len(d) >= 6:
            try:
                _, wp = _st.wilcoxon(d)
                sig += f" wilcoxon p={wp:.3f}"
            except Exception:
                pass
        sig += "" if pv < 0.05 else "  (n.s.)"
    print(f"  {t}: {m:+.3f} +- {s:.3f}  (n={len(d)})   [{detail}]{extra}{sig}")
# ---- pooled across tasks -------------------------------------------------
# Per-task tests are underpowered at these n.  Pooling raw deltas would be
# dominated by t3 (deltas ~0.12) over t2 (~0.015), so instead test the *sign*
# of every complete pair, blocked across tasks: under the null that the two
# agents are equivalent, each pair is a coin flip.
from scipy import stats as _st
allpairs = []
for t in TASKS:
    for g, r in runs.items():
        if not g.startswith("W_rfs") or r["task"] != t or not r["done"]:
            continue
        mate = [rr for gg, rr in runs.items()
                if gg.startswith("W_reb") and rr["task"] == t and rr["seed"] == r["seed"] and rr["done"]]
        if mate:
            allpairs.append((t, r["seed"], r["last5"] - mate[0]["last5"]))
if allpairs:
    d = [x[2] for x in allpairs]
    wins = sum(1 for x in d if x > 0)
    sp = _st.binomtest(wins, len(d), 0.5).pvalue
    print(f"POOLED over all tasks/seeds: anq_rfs wins {wins}/{len(d)} pairs   "
          f"sign-test p={sp:.4f}")
    try:
        _, wp = _st.wilcoxon(d)
        print(f"  (wilcoxon on pooled deltas p={wp:.4f} -- scale-mixed, sign test is the cleaner read)")
    except Exception:
        pass
print("=" * 78)
inc = [g for g, r in runs.items() if g.startswith(("W_", "V_")) and not r["done"]]
if inc:
    print(f"still running ({len(inc)}): " + ", ".join(sorted(inc)))
