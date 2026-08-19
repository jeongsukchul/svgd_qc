"""Which drift_temps gives the best *metric*, not just the best BC fit?

The generator's J J^T is what mani_stdfp's trust region is made of.  A decoder
that fits the mean well but collapses its latent directions yields a degenerate
metric, so BC MSE and metric quality can disagree.
"""
import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np, optax
from agents.anq_stdfp import ANQSTDFPAgent, get_config

d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
obs_np, act_np, term = d['observations'], d['actions'], d['terminals']
OBS, ACT = jnp.asarray(obs_np), jnp.asarray(act_np)
traj = np.concatenate([[0], np.cumsum(term)[:-1]])
Z = (obs_np - obs_np.mean(0)) / (obs_np.std(0) + 1e-6)

rng_np = np.random.default_rng(0)
REF, NQ, K, NZ = 150_000, 200, 64, 32
ridx = rng_np.choice(len(Z), REF, replace=False)
R, Ra, Rt = Z[ridx], act_np[ridx], traj[ridx]
qidx = rng_np.choice(len(Z), NQ, replace=False)
# local data covariance per query state, computed once
C_data = []
for qi in qidx:
    dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
    C_data.append(np.cov(Ra[np.argpartition(dd, K)[:K]].T))

def topk_overlap(A, B, k):
    va = np.linalg.eigh(A)[1][:, ::-1][:, :k]; vb = np.linalg.eigh(B)[1][:, ::-1][:, :k]
    return (np.linalg.svd(va.T @ vb, compute_uv=False) ** 2).sum() / k

def run(temp, steps=30000):
    c = get_config(); c["horizon_length"] = 1; c["drift_temps"] = temp
    agent = ANQSTDFPAgent.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), c)
    D = "modules_actor_drift"; opt = optax.adam(3e-4)
    params = {D: agent.network.params[D]}; ostate = opt.init(params)
    @jax.jit
    def step(params, ostate, rng, idx):
        batch = {"observations": OBS[idx], "actions": ACT[idx][:, None, :]}
        def lf(p):
            full = dict(agent.network.params); full[D] = p[D]
            return agent.drift_bc_loss(batch, full, rng)
        (l, info), g = jax.value_and_grad(lf, has_aux=True)(params)
        u, ostate = opt.update(g, ostate)
        return optax.apply_updates(params, u), ostate, info["generated_to_data_mse"]
    rng = jax.random.PRNGKey(0)
    for i in range(steps):
        rng, r1, r2 = jax.random.split(rng, 3)
        params, ostate, mse = step(params, ostate, r2, jax.random.randint(r1,(256,),0,len(OBS)))
    full = dict(agent.network.params); full[D] = params[D]
    agent = agent.replace(network=agent.network.replace(params=full))
    jac = jax.jit(jax.vmap(jax.jacrev(
        lambda o,n: agent.network.select("actor_drift")(o,n), argnums=1), in_axes=(None,0)))
    prs, ov1, ov2, effc = [], [], [], []
    for j, qi in enumerate(qidx):
        J = np.asarray(jac(OBS[qi], jnp.asarray(rng_np.normal(size=(NZ,8)).astype(np.float32))))
        Cg = np.einsum("nik,njk->ij", J, J) / NZ
        w = np.sort(np.clip(np.linalg.eigvalsh(Cg), 1e-14, None))[::-1]
        prs.append(w.sum()**2/(w**2).sum()); effc.append((w[0]+1e-2)/(w[-1]+1e-2))
        ov1.append(topk_overlap(C_data[j], Cg, 1)); ov2.append(topk_overlap(C_data[j], Cg, 2))
    return float(mse), np.median(prs), np.median(effc), np.median(ov1), np.median(ov2)

print(f"{'drift_temps':>11s} {'BC mse':>8s} {'gen PR':>7s} {'eff cond':>9s} {'align k=1':>10s} {'k=2':>7s}")
print(f"{'data target':>11s} {'-':>8s} {4.10:7.2f} {18:9.0f} {1.0:10.2f} {1.0:7.2f}")
for t in (0.1, 1.0, 3.0, 10.0):
    mse, pr, ec, o1, o2 = run(t)
    print(f"{t:11.1f} {mse:8.4f} {pr:7.2f} {ec:9.1f} {o1:10.3f} {o2:7.3f}", flush=True)
print(f"{'chance':>11s} {'-':>8s} {'-':>7s} {'-':>9s} {0.125:10.3f} {0.25:7.3f}")
