"""Is p(a|s) on antmaze-giant multimodal enough for a generative policy to matter?

For query states, gather nearest neighbours from OTHER trajectories (independent
visits to the same state) and ask two things about their actions:
  1. how much action spread survives conditioning on s   (room for any policy class)
  2. is that spread one blob or two                      (room for a *generative* class)
"""
import numpy as np
rng = np.random.default_rng(0)
d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
obs, act, term = d['observations'], d['actions'], d['terminals']
traj = np.concatenate([[0], np.cumsum(term)[:-1]])          # trajectory id per step
mu, sd = obs.mean(0), obs.std(0) + 1e-6
Z = (obs - mu) / sd

REF, NQ, K = 300_000, 1500, 64
ridx = rng.choice(len(Z), REF, replace=False)
R, Ra, Rt = Z[ridx], act[ridx], traj[ridx]
qidx = rng.choice(len(Z), NQ, replace=False)

def two_means_separation(A):
    """Separation of the best 2-cluster split along PC1, in within-cluster sigmas."""
    X = A - A.mean(0)
    pc = np.linalg.svd(X, full_matrices=False)[2][0]
    p = X @ pc
    order = np.sort(p)
    best, bi = -1, None
    for i in range(len(p) // 4, 3 * len(p) // 4):           # balanced splits only
        lo, hi = order[:i], order[i:]
        w = (lo.var() * len(lo) + hi.var() * len(hi)) / len(p)
        if w < 1e-12: continue
        s = (hi.mean() - lo.mean()) / np.sqrt(w)
        if s > best: best, bi = s, i
    return best

cond_sd, sep_nn, sep_rand, dists = [], [], [], []
for qi in qidx:
    q, qt = Z[qi], traj[qi]
    dd = np.linalg.norm(R - q, axis=1)
    dd[Rt == qt] = np.inf                                    # independent visits only
    nn = np.argpartition(dd, K)[:K]
    A = Ra[nn]
    dists.append(np.median(dd[nn]))
    cond_sd.append(A.std(0).mean())
    sep_nn.append(two_means_separation(A))
    sep_rand.append(two_means_separation(Ra[rng.choice(REF, K, replace=False)]))

cond_sd, sep_nn, sep_rand, dists = map(np.array, (cond_sd, sep_nn, sep_rand, dists))
marg = act.std(0).mean()
print(f"neighbourhoods: K={K} from other trajectories, median obs distance "
      f"{np.median(dists):.3f} (normalised, {obs.shape[1]} dims)")
print(f"marginal action sd            {marg:.3f}")
print(f"conditional action sd  |s     {np.median(cond_sd):.3f}   "
      f"({np.median(cond_sd)/marg*100:.0f}% of marginal survives conditioning)")
print(f"2-cluster separation, neighbours   {np.median(sep_nn):.2f} sigma")
print(f"2-cluster separation, random sets  {np.median(sep_rand):.2f} sigma  (unimodal-mixture control)")
print(f"fraction of neighbourhoods with separation > random-set median: "
      f"{(sep_nn > np.median(sep_rand)).mean()*100:.0f}%")
