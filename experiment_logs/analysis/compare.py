"""t3 metric-vs-isotropic comparison, aligned on matched steps."""
import csv, glob
import numpy as np
ARMS = [
    ("L_rebrac_a001_t3",   "rebrac.py",             "iso",    0.01,   0),
    ("B_lam001_t3",        "anq_stdfp",             "iso",    0.01,   0),
    ("B1_lam001_t3_s1",    "anq_stdfp",             "iso",    0.01,   1),
    ("J_lam001_pretanh_t3","anq_stdfp +pretanh",    "iso",    0.01,   0),
    ("I_mani_weak_t3",     "mani_stdfp",            "METRIC", 1.3e-4, 0),
    ("I1_mani_weak_t3_s1", "mani_stdfp",            "METRIC", 1.3e-4, 1),
    ("K_mani_norm_t3",     "mani scale-free",       "METRIC", 0.01,   0),
]
cur, qm = {}, {}
for g, name, pen, lam, sd in ARMS:
    p = glob.glob(f"exp/beat/ant2/{g}/*/*/eval.csv")
    if p:
        rows = list(csv.DictReader(open(p[0])))
        cur[g] = {int(float(r['step'])): float(r['success']) for r in rows if r.get('success')}
    q = glob.glob(f"exp/beat/ant2/{g}/*/*/offline_agent.csv")
    if q:
        rows = list(csv.DictReader(open(q[0])))
        k = next((k for k in rows[0] if k.endswith("/q_mean")), None)
        if k: qm[g] = {int(float(r['step'])): float(r[k]) for r in rows}

steps = sorted(set().union(*[set(c) for c in cur.values()])) if cur else []
common = [s for s in steps if all(s in cur.get(g, {}) for g, *_ in ARMS if g in cur)]
print(f"success by step  (aligned; every arm has data through {max(common)//1000 if common else 0}k)")
print(f"{'agent':22s} {'pen':>6s} {'sd':>2s} | " + " ".join(f"{s//1000:>4d}k" for s in steps))
for g, name, pen, lam, sd in ARMS:
    if g not in cur: continue
    print(f"{name:22s} {pen:>6s} {sd:2d} | " +
          " ".join(f"{cur[g][s]:5.2f}" if s in cur[g] else "    -" for s in steps))
if common:
    print(f"\nmean over the {len(common)} matched steps (through {max(common)//1000}k):")
    agg = {}
    for g, name, pen, lam, sd in ARMS:
        if g not in cur: continue
        agg.setdefault((name, pen), []).append(np.mean([cur[g][s] for s in common]))
    for (name, pen), xs in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {name:22s} {pen:>6s}  {np.mean(xs):.3f}  seeds {[round(x,3) for x in xs]}")
qsteps = [200000, 400000, 600000, 800000, 1000000]
print(f"\nq_mean (-200 = never reaches goal; compare only at matched steps)")
print(f"{'agent':22s} {'sd':>2s} | " + " ".join(f"{s//1000:>6d}k" for s in qsteps))
for g, name, pen, lam, sd in ARMS:
    if g not in qm: continue
    print(f"{name:22s} {sd:2d} | " + " ".join(
        f"{qm[g][s]:7.1f}" if s in qm[g] else f"{'-':>7s}" for s in qsteps))
print("\nreference (other workspace, 1M t3): rebrac 0.20 | anq_stdfp 0.14 | anq_stdfp3(modified) 0.39")
