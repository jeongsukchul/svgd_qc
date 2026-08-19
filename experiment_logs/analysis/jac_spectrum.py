"""Train the drift decoder alone, then ask whether manifold_ridge swamps its metric."""
import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np, optax, functools
from agents.anq_stdfp import ANQSTDFPAgent, get_config

d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
OBS, ACT = jnp.asarray(d['observations']), jnp.asarray(d['actions'])
c = get_config(); c["horizon_length"] = 1; c["drift_temps"] = 3.0
agent = ANQSTDFPAgent.create(0, np.zeros((1, OBS.shape[1]), np.float32),
                             np.zeros((1, ACT.shape[1]), np.float32), c)

DRIFT = "modules_actor_drift"
opt = optax.adam(3e-4)
params = {DRIFT: agent.network.params[DRIFT]}
opt_state = opt.init(params)

@jax.jit
def step(params, opt_state, rng, idx):
    batch = {"observations": OBS[idx], "actions": ACT[idx][:, None, :]}
    def loss_fn(p):
        full = dict(agent.network.params); full[DRIFT] = p[DRIFT]
        return agent.drift_bc_loss(batch, full, rng)
    (l, info), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    upd, opt_state = opt.update(g, opt_state)
    return optax.apply_updates(params, upd), opt_state, l, info["generated_to_data_mse"]

rng = jax.random.PRNGKey(0)
N_STEPS = 30000
for i in range(N_STEPS):
    rng, r1, r2 = jax.random.split(rng, 3)
    idx = jax.random.randint(r1, (256,), 0, len(OBS))
    params, opt_state, l, mse = step(params, opt_state, r2, idx)
    if i % 10000 == 0 or i == N_STEPS - 1:
        print(f"  step {i:6d}  drift_loss {float(l):.4f}  gen->data mse {float(mse):.4f}", flush=True)

full = dict(agent.network.params); full[DRIFT] = params[DRIFT]
agent = agent.replace(network=agent.network.replace(params=full))

# Jacobian dG/dz at z ~ N(0, I) over held-out states
rng, r1, r2 = jax.random.split(rng, 3)
qs = jax.random.randint(r1, (512,), 0, len(OBS))
obs = OBS[qs]; z = jax.random.normal(r2, (512, ACT.shape[1]))
J = agent.generator_jacobians(obs, z) if hasattr(agent, "generator_jacobians") else \
    jax.vmap(jax.jacrev(lambda o, n: agent.network.select("actor_drift")(o, n), argnums=1))(obs, z)
JJt = np.asarray(jnp.einsum("...ik,...jk->...ij", J, J))
ev = np.sort(np.linalg.eigvalsh(JJt), axis=-1)[:, ::-1]
med = np.median(ev, axis=0)
print("\neigenvalues of J J^T (median over 512 states):")
print("  ", np.array2string(med, precision=4, suppress_small=False))
for r in (1e-1, 1e-2, 1e-3, 1e-4):
    eff = (med[0] + r) / (med[-1] + r)
    frac = (med > r).sum()
    print(f"  ridge {r:.0e}: {frac}/{len(med)} eigenvalues exceed it, "
          f"effective condition number {eff:6.1f}  (true {med[0]/med[-1]:.1f})")
print("\n  data conditional covariance for comparison: cond number 18, PR 4.10 of 8")
pr = med.sum()**2 / (med**2).sum()
print(f"  generator pushforward participation ratio {pr:.2f} of {len(med)}")
