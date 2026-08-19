"""Does the generator metric constrain the SAME directions the data varies in?

Eigenvalue magnitudes matching is not enough: if the top subspace of J J^T is
misaligned with the local action covariance, the metric frees directions the
behaviour never takes and constrains ones it does.
"""
import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np, optax, pickle, os
from agents.anq_stdfp import ANQSTDFPAgent, get_config

SP = os.path.dirname(os.path.abspath(__file__))
d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
obs_np, act_np, term = d['observations'], d['actions'], d['terminals']
OBS, ACT = jnp.asarray(obs_np), jnp.asarray(act_np)
c = get_config(); c["horizon_length"] = 1; c["drift_temps"] = 3.0
agent = ANQSTDFPAgent.create(0, np.zeros((1, 29), np.float32), np.zeros((1, 8), np.float32), c)
DRIFT = "modules_actor_drift"
ckpt = f"{SP}/drift_bc.pkl"

if os.path.exists(ckpt):
    params = pickle.load(open(ckpt, "rb"))
else:
    opt = optax.adam(3e-4); params = {DRIFT: agent.network.params[DRIFT]}
    opt_state = opt.init(params)
    @jax.jit
    def step(params, opt_state, rng, idx):
        batch = {"observations": OBS[idx], "actions": ACT[idx][:, None, :]}
        def loss_fn(p):
            full = dict(agent.network.params); full[DRIFT] = p[DRIFT]
            return agent.drift_bc_loss(batch, full, rng)
        (l, info), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
        upd, opt_state = opt.update(g, opt_state)
        return optax.apply_updates(params, upd), opt_state, info["generated_to_data_mse"]
    rng = jax.random.PRNGKey(0)
    for i in range(30000):
        rng, r1, r2 = jax.random.split(rng, 3)
        params, opt_state, mse = step(params, opt_state, r2, jax.random.randint(r1, (256,), 0, len(OBS)))
    print(f"trained drift decoder, gen->data mse {float(mse):.4f}")
    pickle.dump(params, open(ckpt, "wb"))

full = dict(agent.network.params); full[DRIFT] = params[DRIFT]
agent = agent.replace(network=agent.network.replace(params=full))

# --- local data covariance vs generator pushforward, at the same states ---
rng = np.random.default_rng(0)
traj = np.concatenate([[0], np.cumsum(term)[:-1]])
Z = (obs_np - obs_np.mean(0)) / (obs_np.std(0) + 1e-6)
REF, NQ, K, NZ = 200_000, 300, 64, 32
ridx = rng.choice(len(Z), REF, replace=False)
R, Ra, Rt = Z[ridx], act_np[ridx], traj[ridx]
qidx = rng.choice(len(Z), NQ, replace=False)

jac = jax.jit(jax.vmap(jax.jacrev(
    lambda o, n: agent.network.select("actor_drift")(o, n), argnums=1), in_axes=(None, 0)))

def topk_overlap(A, B, k):
    """Fraction of subspace shared by the top-k eigenvectors of A and B (1 = identical)."""
    va = np.linalg.eigh(A)[1][:, ::-1][:, :k]
    vb = np.linalg.eigh(B)[1][:, ::-1][:, :k]
    return (np.linalg.svd(va.T @ vb, compute_uv=False) ** 2).sum() / k

res = {k: [] for k in (1, 2, 3)}; rand = {k: [] for k in (1, 2, 3)}
for qi in qidx:
    dd = np.linalg.norm(R - Z[qi], axis=1); dd[Rt == traj[qi]] = np.inf
    nn = np.argpartition(dd, K)[:K]
    C_data = np.cov(Ra[nn].T)
    zs = jnp.asarray(rng.normal(size=(NZ, 8)).astype(np.float32))
    J = np.asarray(jac(OBS[qi], zs))                      # (NZ, 8, 8)
    C_gen = np.einsum("nik,njk->ij", J, J) / NZ
    Q = np.linalg.qr(rng.normal(size=(8, 8)))[0]          # random orthonormal control
    for k in (1, 2, 3):
        res[k].append(topk_overlap(C_data, C_gen, k))
        rand[k].append(topk_overlap(C_data, Q @ np.diag([8,7,6,5,4,3,2,1.]) @ Q.T, k))

print(f"\ntop-k subspace overlap between local data covariance and generator J J^T"
      f"  ({NQ} states, K={K} neighbours, {NZ} z-draws)")
for k in (1, 2, 3):
    print(f"  k={k}:  generator {np.median(res[k]):.3f}    random control {np.median(rand[k]):.3f}"
          f"    (chance = k/8 = {k/8:.3f})")
