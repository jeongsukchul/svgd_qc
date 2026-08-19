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
rho_metric, rho_iso, ratio_med = [], [], []
for qi in rng.choice(len(Z), 250, replace=False):
    dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
    nn = np.argpartition(dd, K)[:K]
    C = np.cov(Ra[nn].T) + 1e-4*np.eye(8)              # local behaviour covariance
    Cinv = np.linalg.inv(C)
    J = np.asarray(jac(OBS[qi], jnp.asarray(rng.normal(size=(NZ,8)).astype(np.float32))))
    M = np.einsum("nik,njk->ij", J, J)/NZ + 1e-2*np.eye(8)
    M = M * 8.0/np.trace(M)                            # scale-free, as the agent now uses
    Minv = np.linalg.inv(M)
    u = rng.normal(size=(NDIR,8)); u /= np.linalg.norm(u,axis=1,keepdims=True)  # FIXED step size
    true_cost   = np.einsum("ni,ij,nj->n", u, Cinv, u)   #真 out-of-distribution cost
    metric_cost = np.einsum("ni,ij,nj->n", u, Minv, u)   # what mani penalises
    iso_cost    = np.ones(NDIR)                          # what ReBRAC penalises
    rho_metric.append(spearmanr(metric_cost, true_cost).statistic)
    rho_iso.append(0.0)
    # how much cheaper is the metric's preferred direction, in true cost?
    ratio_med.append(true_cost[np.argmin(metric_cost)] / np.median(true_cost))
print(f"at fixed step size, over {len(rho_metric)} states x {NDIR} directions:")
print(f"  Spearman(metric penalty, true out-of-distribution cost) = {np.median(rho_metric):+.3f}")
print(f"  Spearman(isotropic penalty, same)                       = {rho_iso[0]:+.3f}  (blind by construction)")
print(f"  the metric's cheapest direction costs {np.median(ratio_med):.2f}x the median direction")
print(f"  fraction of states where metric correlation > 0: {np.mean(np.array(rho_metric)>0)*100:.0f}%")
