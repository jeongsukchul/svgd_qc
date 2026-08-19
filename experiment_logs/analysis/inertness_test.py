"""Is the drift decoder actually connected to the executed action?

Perturb every parameter the generative path owns (actor_drift + noise_actor)
and see whether sample_actions moves at all.
"""
import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np
from agents.anq_stdfp3 import ANQSTDFP3Agent, get_config as cfg3
from agents.anq_stdfp import ANQSTDFPAgent, get_config as cfg1

OB, AC, N = 29, 8, 64
ex_ob = jnp.zeros((1, OB)); ex_ac = jnp.zeros((1, AC))
obs = jax.random.normal(jax.random.PRNGKey(7), (N, OB))

def perturb(agent, prefixes, scale=10.0):
    """Add large noise to every param under the given module prefixes."""
    params = agent.network.params
    hit = []
    def walk(d, path, rng):
        out = {}
        for k, v in d.items():
            p = path + [k]
            if isinstance(v, dict) or hasattr(v, "items"):
                out[k] = walk(v, p, rng)
            else:
                if any(pref in p[0] for pref in prefixes):
                    hit.append("/".join(p))
                    rng, sub = jax.random.split(rng)
                    out[k] = v + scale * jax.random.normal(sub, v.shape)
                else:
                    out[k] = v
        return out
    new = walk(params, [], jax.random.PRNGKey(99))
    return agent.replace(network=agent.network.replace(params=new)), hit

def report(name, agent):
    a0 = np.asarray(agent.sample_actions(obs, rng=jax.random.PRNGKey(0)))
    a1 = np.asarray(agent.sample_actions(obs, rng=jax.random.PRNGKey(12345)))
    pert, hit = perturb(agent, ["modules_actor_drift", "modules_noise_actor"])
    a2 = np.asarray(pert.sample_actions(obs, rng=jax.random.PRNGKey(0)))
    print(f"\n=== {name} ===")
    print(f"  perturbed {len(hit)} param tensors in the generative path")
    print(f"  max |action(z_rng=0) - action(z_rng=12345)| = {np.abs(a0-a1).max():.3e}")
    print(f"  max |action - action after perturbing decoder| = {np.abs(a0-a2).max():.3e}")
    print(f"  -> generative path is {'INERT (dead code)' if np.abs(a0-a2).max() < 1e-6 else 'CONNECTED'}")

c = cfg3(); c["horizon_length"] = 1
report("anq_stdfp3", ANQSTDFP3Agent.create(0, ex_ob, ex_ac, c))

for space in ("action", "pretanh"):
    c = cfg1(); c["horizon_length"] = 1; c["refine_residual_space"] = space
    report(f"anq_stdfp (refine_residual_space={space})",
           ANQSTDFPAgent.create(0, ex_ob, ex_ac, c))
