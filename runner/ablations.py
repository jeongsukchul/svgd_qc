"""Aggregate every t3 ablation arm into one ranked table."""
import csv, glob
import numpy as np

ROOT = "/workspace/svgd_qc/exp/beat"
runs = {}
for ev in glob.glob(f"{ROOT}/*/*/*/*/eval.csv"):
    g = ev.split("/")[len(ROOT.split("/")) + 1]
    v = [float(r["success"]) for r in csv.DictReader(open(ev)) if r.get("success")]
    if len(v) >= 10:
        runs.setdefault(g, []).append(float(np.mean(v[-5:])))

ARMS = [
    ("winner: action + data + KL, stochastic z", "W_rfs_t3"),
    ("live base (no stop-grad on base input)",   "LB_"),
    ("target_multiplier 0.03",                   "TM03_"),
    ("target_multiplier 0.06",                   "TM06_"),
    ("target_multiplier 0.25",                   "TM25_"),
    ("DETERMINISTIC z + KL",                     "DETK_"),
    ("DETERMINISTIC z, no reg",                  "DET_"),
    ("stochastic z, no reg",                     "NOR_"),
    ("SAC entropy on latent",                    "SE_ent"),
    ("sigreg coeff 0.1 (det z)",                 "SIGc01"),
    ("sigreg coeff 1.0 (det z)",                 "SIG_det"),
    ("sigreg coeff 10  (det z)",                 "SIGc10"),
    ("sigreg (stochastic z)",                    "SIGS_"),
]
rows = []
for label, pre in ARMS:
    v = [x for g, l in runs.items() if g.startswith(pre) for x in l]
    if v:
        sem = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        rows.append((label, len(v), float(np.mean(v)), sem, sorted(v, reverse=True)))
rows.sort(key=lambda r: -r[2])
print(f"{'arm':44s} {'n':>2s} {'mean':>7s} {'sem':>6s}   seeds")
for label, n, m, s, v in rows:
    print(f"{label:44s} {n:2d} {m:7.3f} {s:6.3f}   " + " ".join(f"{x:.2f}" for x in v))
