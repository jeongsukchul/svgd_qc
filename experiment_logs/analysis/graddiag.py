import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np
from agents.anq_stdfp import ANQSTDFPAgent, get_config as cfg1
from agents.mani_stdfp import ManiSTDFPAgent, get_config as cfgm

d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
rng_np = np.random.default_rng(0); idx = rng_np.choice(len(d['observations']), 256, replace=False)
batch = {"observations": jnp.asarray(d['observations'][idx]),
         "actions": jnp.asarray(d['actions'][idx])[:, None, :],
         "valid": jnp.ones((256, 1)), "rewards": jnp.zeros((256, 1))}

def gnorm(agent, tag):
    def lf(p): return agent.refine_actor_loss(batch, p, jax.random.PRNGKey(0))
    (l, info), g = jax.value_and_grad(lf, has_aux=True)(agent.network.params)
    ref = g["modules_refine_actor"]
    flat = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(ref)])
    print(f"{tag:34s} loss={float(l):12.3f}  |grad refine_actor| = {float(jnp.linalg.norm(flat)):.3e}"
          f"   delta_rms={float(info['delta_rms']):.4f}")
    return float(jnp.linalg.norm(flat))

c = cfg1(); c["horizon_length"] = 1; c["drift_temps"] = 3.0; c["lam"] = 5.0
a1 = ANQSTDFPAgent.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), c)
g1 = gnorm(a1, "anq_stdfp lam=5 anchor=data")

c = cfg1(); c["horizon_length"] = 1; c["drift_temps"] = 3.0; c["lam"] = 5.0; c["refine_anchor"]="base"
a2 = ANQSTDFPAgent.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), c)
g2 = gnorm(a2, "anq_stdfp lam=5 anchor=base")

cm = cfgm(); cm["horizon_length"] = 1; cm["drift_temps"] = 3.0; cm["lam"] = 0.1
am = ManiSTDFPAgent.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), cm)
gm = gnorm(am, "mani_stdfp lam=0.1")

cm = cfgm(); cm["horizon_length"] = 1; cm["drift_temps"] = 3.0; cm["lam"] = 0.0
am0 = ManiSTDFPAgent.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), cm)
gm0 = gnorm(am0, "mani_stdfp lam=0 (Q term only)")
print(f"\n  ratio mani/anq for the pure-Q gradient: {gm0/g1:.4f}")
