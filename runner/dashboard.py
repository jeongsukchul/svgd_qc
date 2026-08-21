"""Full-metric dashboard: eval + the training diagnostics that matter, per arm.

Diagnostic columns and what they detect:
  mse    generated_to_data_mse   decoder collapse (<0.03 on cube = overfit) / underfit
  kl     latent_kl               in-support component activity vs its budget
  drms   delta_rms               out-of-support component activity
  q      critic/q_mean (final)   critic divergence (past the TD fixed point)
  dq     q drift last 20%        late critic instability
"""
import csv, glob, json, sys
import numpy as np

only = sys.argv[1] if len(sys.argv) > 1 else ""
arms = {}
for ev in sorted(glob.glob("exp/beat/ant2/*/*/*/eval.csv")):
    g = ev.split("/")[3]
    if "_mislaunched" in ev or (only and only not in g):
        continue
    try:
        fl = json.load(open(ev.replace("eval.csv", "flags.json")))
    except Exception:
        continue
    env = "CUBE" if fl["env_name"].startswith("cube") else "ANT"
    task = fl["env_name"].split("task")[-1].split("-")[0]
    v = [float(r["success"]) for r in csv.DictReader(open(ev)) if r.get("success")]
    if not v:
        continue
    diag = {}
    oa = ev.replace("eval.csv", "offline_agent.csv")
    try:
        rows = list(csv.DictReader(open(oa)))
        def col(k):
            vals = [float(r[k]) for r in rows if r.get(k)]
            return vals
        mse = col("actor/generated_to_data_mse")
        kl = col("actor/latent_kl")
        dr = col("actor/delta_rms")
        q = col("critic/q_mean")
        diag = dict(
            mse=mse[-1] if mse else None,
            kl=kl[-1] if kl else None,
            drms=dr[-1] if dr else None,
            q=q[-1] if q else None,
            dq=(q[-1] - q[int(len(q)*0.8)]) if q else None,
        )
    except Exception:
        pass
    arms.setdefault((env, task, g.rsplit("_s", 1)[0]), []).append(
        dict(n=len(v), last5=float(np.mean(v[-5:])), best=max(v), curve=v, **diag))

hdr = f"{'arm':22s} {'n':>2s} {'last5':>6s} {'best':>5s} | {'mse':>6s} {'kl':>6s} {'drms':>7s} {'q':>8s} {'dq':>7s}"
cur = None
out = []
for (env, task, base), l in arms.items():
    done = [r for r in l if r["n"] >= 10]
    if not done:
        continue
    m = float(np.mean([r["last5"] for r in done]))
    out.append((env, task, m, base, done))
out.sort(key=lambda x: (x[0], x[1], -x[2]))
for env, task, m, base, done in out:
    key = f"{env}-t{task}"
    if key != cur:
        print(f"=== {key} ==="); print(hdr); cur = key
    d0 = done[0]
    fmt = lambda k, w, p: (f"{np.mean([r[k] for r in done if r[k] is not None]):{w}.{p}f}"
                           if any(r.get(k) is not None for r in done) else " " * w)
    print(f"{base:22s} {len(done):2d} {m:6.3f} {np.mean([r['best'] for r in done]):5.2f} | "
          f"{fmt('mse',6,4)} {fmt('kl',6,2)} {fmt('drms',7,4)} {fmt('q',8,1)} {fmt('dq',7,1)}")
