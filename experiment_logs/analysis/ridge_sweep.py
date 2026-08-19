"""Does the generator metric predict how far out-of-distribution a move goes?

At a FIXED step size an isotropic penalty is blind to direction by construction.
The metric claims to know which directions are safe.  Test it against the local
behaviour distribution the policy actually has to stay inside.
"""
import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np, pickle, os
from scipy.stats import spearmanr
from agents.anq_stdfp import ANQSTDFPAgent, get_config
SP = os.path.dirname(os.path.abspath(__file__))
d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
obs_np, act_np, term = d['observations'], d['actions'], d['terminals']
OBS = jnp.asarray(obs_np)
c = get_config(); c["horizon_length"]=1; c["drift_temps"]=3.0
agent = ANQSTDFPAgent.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), c)
params = pickle.load(open(f"{SP}/drift_bc.pkl","rb"))
full = dict(agent.network.params); full["modules_actor_drift"] = params["modules_actor_drift"]
agent = agent.replace(network=agent.network.replace(params=full))
jac = jax.jit(jax.vmap(jax.jacrev(
    lambda o,n: agent.network.select("actor_drift")(o,n), argnums=1), in_axes=(None,0)))

rng = np.random.default_rng(0)
traj = np.concatenate([[0], np.cumsum(term)[:-1]])
Z = (obs_np - obs_np.mean(0))/(obs_np.std(0)+1e-6)
ridx = rng.choice(len(Z), 200_000, replace=False)
R, Ra, Rt = Z[ridx], act_np[ridx], traj[ridx]
K, NDIR, NZ = 96, 256, 32
RIDGES=[1e-1,1e-2,1e-3,1e-4]
import collections
RES=collections.defaultdict(list)
rho_metric, rho_iso, ratio_med = [], [], []
for qi in rng.choice(len(Z), 250, replace=False):
    dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
    nn = np.argpartition(dd, K)[:K]
    C = np.cov(Ra[nn].T) + 1e-4*np.eye(8)              # local behaviour covariance
    Cinv = np.linalg.inv(C)
    J = np.asarray(jac(OBS[qi], jnp.asarray(rng.normal(size=(NZ,8)).astype(np.float32))))
    JJt = np.einsum("nik,njk->ij", J, J)/NZ
    u = rng.normal(size=(NDIR,8)); u /= np.linalg.norm(u,axis=1,keepdims=True)
    true_cost = np.einsum("ni,ij,nj->n", u, Cinv, u)
    for RIDGE in RIDGES:
        Minv = np.linalg.inv(JJt + RIDGE*np.eye(8))
        mc = np.einsum("ni,ij,nj->n", u, Minv, u)
        RES[RIDGE].append((spearmanr(mc, true_cost).statistic,
                           true_cost[np.argmin(mc)]/np.median(true_cost)))
print(f"manifold_ridge sweep  ({len(RES[RIDGES[0]])} states x {NDIR} directions, fixed step size)")
print(f"{'ridge':>8s} {'rho':>8s} {'best-dir cost':>14s}")
for R in RIDGES:
    a=np.array(RES[R]); print(f"{R:8.0e} {np.median(a[:,0]):+8.3f} {np.median(a[:,1]):14.2f}")
print("  (isotropic penalty: rho = 0.000 at any ridge, by construction)")
