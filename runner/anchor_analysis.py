"""Why does the data anchor work on antmaze but kill cube?

A1: matched-state action spread at each domain's OPERATING point (antmaze h=1,
    cube h=5 chunks).  The anchor pulls the policy toward THE logged action
    (chunk); if near-duplicate states carry very different chunks, that pull is
    toward an arbitrary mode.
A2: how far the trained decoder sits from the per-state logged action,
    relative to action scale (from the runs' own logged generated_to_data_mse).
"""
import numpy as np, csv, glob

def matched_state_spread(path, name, horizons):
    d = np.load(path)
    obs = d["observations"][:200000].astype(np.float32)
    act = d["actions"][:200000].astype(np.float32)
    term = d["terminals"][:200000]
    z = (obs - obs.mean(0)) / (obs.std(0) + 1e-6)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(z) - 10, 600, replace=False)
    for h in horizons:
        # build h-step chunks (skip windows crossing terminals)
        ok = np.ones(len(act) - h, bool)
        for j in range(h - 1):
            ok &= term[j:len(term) - h + j] == 0
        diffs = []
        for i in idx:
            if i >= len(ok) or not ok[i]:
                continue
            dist = np.sum((z - z[i]) ** 2, axis=1)
            nn = np.argsort(dist)[1:9]
            nn = nn[(nn < len(ok))]
            nn = nn[ok[nn]][:8]
            if len(nn) < 4 or dist[nn[-1]] > 0.5 * z.shape[1]:
                continue
            chunks = np.stack([np.concatenate([act[n + j] for j in range(h)]) for n in nn])
            diffs.append(np.mean(np.std(chunks, axis=0)))
        chunk_scale = np.std(act) * 1.0
        r = np.mean(diffs) / chunk_scale
        print(f"  {name} h={h}: matched-state chunk spread = {np.mean(diffs):.4f} "
              f"(action std {chunk_scale:.3f})  ratio={r:.3f}  n={len(diffs)}")

print("A1: matched-state action-chunk spread (the anchor's ambiguity)")
matched_state_spread("/root/.ogbench/data/antmaze-giant-navigate-v0.npz", "antmaze", (1, 5))
matched_state_spread("/root/.ogbench/data/cube-double-play-v0.npz", "cube   ", (1, 5))

print("\nA2: decoder distance to the logged action, relative to action variance")
for tag, pat, dstd in (("antmaze winner (W_rfs_t3_s0)", "exp/beat/ant2/W_rfs_t3_s0/*/*/offline_agent.csv", 0.6957),
                       ("cube winner (CDG_tm0125_s0) ", "exp/beat/ant2/CDG_tm0125_t2_s0/*/*/offline_agent.csv", 0.4106)):
    p = glob.glob(pat)
    if not p:
        print(f"  {tag}: no csv"); continue
    rows = list(csv.DictReader(open(p[0])))
    k = "actor/generated_to_data_mse"
    v = [float(r[k]) for r in rows if r.get(k)]
    mse = np.mean(v[-20:])
    print(f"  {tag}: final mse={mse:.4f}, action var={dstd**2:.3f} -> "
          f"normalised distance = {mse/dstd**2:.3f}")
