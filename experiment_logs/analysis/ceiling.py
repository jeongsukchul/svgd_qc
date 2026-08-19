"""How well can a 64-sample covariance even identify its own top eigenvector?

Split the neighbourhood in half and measure overlap between the two halves.
That is the reliability ceiling any generator alignment score is measured against.
"""
import numpy as np
rng = np.random.default_rng(0)
d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
obs, act, term = d['observations'], d['actions'], d['terminals']
traj = np.concatenate([[0], np.cumsum(term)[:-1]])
Z = (obs - obs.mean(0)) / (obs.std(0) + 1e-6)
ridx = rng.choice(len(Z), 200_000, replace=False)
R, Ra, Rt = Z[ridx], act[ridx], traj[ridx]
def ov(A, B, k):
    va = np.linalg.eigh(A)[1][:, ::-1][:, :k]; vb = np.linalg.eigh(B)[1][:, ::-1][:, :k]
    return (np.linalg.svd(va.T @ vb, compute_uv=False) ** 2).sum() / k
c = {1: [], 2: [], 3: []}
for qi in rng.choice(len(Z), 400, replace=False):
    dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
    nn = np.argpartition(dd, 64)[:64]; rng.shuffle(nn)
    A, B = np.cov(Ra[nn[:32]].T), np.cov(Ra[nn[32:]].T)
    for k in (1, 2, 3): c[k].append(ov(A, B, k))
print("split-half reliability of the local data covariance (the measurement ceiling):")
for k in (1, 2, 3):
    print(f"  k={k}: {np.median(c[k]):.3f}   (generator scored 0.484 / 0.568 / 0.642)")
