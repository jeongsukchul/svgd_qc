import numpy as np, sys, glob, os
def two_means_separation(A):
    X = A - A.mean(0)
    pc = np.linalg.svd(X, full_matrices=False)[2][0]
    p = X @ pc; order = np.sort(p); best = -1
    for i in range(len(p)//4, 3*len(p)//4):
        lo, hi = order[:i], order[i:]
        w = (lo.var()*len(lo) + hi.var()*len(hi))/len(p)
        if w < 1e-12: continue
        s = (hi.mean()-lo.mean())/np.sqrt(w)
        best = max(best, s)
    return best

def analyse(path, REF=200_000, NQ=800, K=64):
    rng = np.random.default_rng(0)
    d = np.load(path); obs, act, term = d['observations'], d['actions'], d['terminals']
    traj = np.concatenate([[0], np.cumsum(term)[:-1]])
    Z = (obs - obs.mean(0))/(obs.std(0)+1e-6)
    REF = min(REF, len(Z))
    ridx = rng.choice(len(Z), REF, replace=False)
    R, Ra, Rt = Z[ridx], act[ridx], traj[ridx]
    sep_nn, sep_rand, cond, dist = [], [], [], []
    for qi in rng.choice(len(Z), NQ, replace=False):
        dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
        nn = np.argpartition(dd, K)[:K]
        sep_nn.append(two_means_separation(Ra[nn]))
        sep_rand.append(two_means_separation(Ra[rng.choice(REF, K, replace=False)]))
        cond.append(Ra[nn].std(0).mean()); dist.append(np.median(dd[nn]))
    return (np.median(sep_nn), np.median(sep_rand), np.median(cond),
            act.std(0).mean(), np.median(dist), obs.shape[1], act.shape[1])

print(f"{'dataset':28s} {'sep|s':>6s} {'sep_rand':>9s} {'excess':>7s} {'cond_sd/marg':>13s} {'nn_dist':>8s}")
print(f"{'(unimodal null = 2.82)':28s}")
for name in ['antmaze-giant-navigate-v0','antmaze-large-navigate-v0','humanoidmaze-large-navigate-v0',
             'cube-single-play-v0','cube-double-play-v0','cube-triple-play-v0','scene-play-v0']:
    p = os.path.expanduser(f'~/.ogbench/data/{name}.npz')
    if not os.path.exists(p): continue
    s, sr, c, m, dm, od, ad = analyse(p)
    print(f"{name:28s} {s:6.2f} {sr:9.2f} {s-2.82:7.2f} {c/m*100:12.0f}% {dm:8.2f}")
    sys.stdout.flush()
