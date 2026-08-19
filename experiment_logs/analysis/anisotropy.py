"""Does p(a|s) have anisotropic spread for a generator metric to exploit?

Isotropic conditional spread => the metric reduces to a scaled identity and
mani_stdfp can do nothing ReBRAC's ||.||^2 cannot.
"""
import numpy as np, os
rng = np.random.default_rng(0)

def spectrum(name, REF=200_000, NQ=800, K=64):
    d = np.load(os.path.expanduser(f'~/.ogbench/data/{name}.npz'))
    obs, act, term = d['observations'], d['actions'], d['terminals']
    traj = np.concatenate([[0], np.cumsum(term)[:-1]])
    Z = (obs - obs.mean(0)) / (obs.std(0) + 1e-6)
    ridx = rng.choice(len(Z), min(REF, len(Z)), replace=False)
    R, Ra, Rt = Z[ridx], act[ridx], traj[ridx]
    prs, conds, eigs = [], [], []
    for qi in rng.choice(len(Z), NQ, replace=False):
        dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
        nn = np.argpartition(dd, K)[:K]
        C = np.cov(Ra[nn].T)
        w = np.clip(np.linalg.eigvalsh(C), 1e-12, None)[::-1]
        prs.append(w.sum()**2 / (w**2).sum())          # participation ratio
        conds.append(w[0] / w[-1])
        eigs.append(w / w.sum())
    # marginal, for reference
    wm = np.clip(np.linalg.eigvalsh(np.cov(act.T)), 1e-12, None)[::-1]
    pr_m = wm.sum()**2 / (wm**2).sum()
    e = np.median(np.array(eigs), axis=0)
    print(f"\n{name}   (action_dim={act.shape[1]}, isotropic PR would be {act.shape[1]})")
    print(f"  conditional participation ratio  {np.median(prs):.2f}"
          f"   marginal {pr_m:.2f}")
    print(f"  conditional cond. number lmax/lmin  {np.median(conds):.0f}")
    print(f"  median normalised eigenvalues: {np.round(e,3)}")
    print(f"  top-2 directions hold {e[:2].sum()*100:.0f}% of conditional variance")

for n in ['antmaze-giant-navigate-v0', 'cube-double-play-v0', 'scene-play-v0']:
    spectrum(n)
