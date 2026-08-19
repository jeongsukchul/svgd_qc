import sys; sys.path.insert(0, "/home/sukchul/qc")
import jax, jax.numpy as jnp, numpy as np
from agents.anq_stdfp import ANQSTDFPAgent, get_config as cfg1
from agents.mani_stdfp import ManiSTDFPAgent, get_config as cfgm
d = np.load('/home/sukchul/.ogbench/data/antmaze-giant-navigate-v0.npz')
idx = np.random.default_rng(0).choice(len(d['observations']), 256, replace=False)
batch = {"observations": jnp.asarray(d['observations'][idx]),
         "actions": jnp.asarray(d['actions'][idx])[:, None, :],
         "valid": jnp.ones((256,1)), "rewards": jnp.zeros((256,1))}
def gn(agent):
    def lf(p): return agent.refine_actor_loss(batch, p, jax.random.PRNGKey(0))
    (l, info), g = jax.value_and_grad(lf, has_aux=True)(agent.network.params)
    flat = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(g["modules_refine_actor"])])
    return float(jnp.linalg.norm(flat))
def mk(cls, cfg, **kw):
    c = cfg(); c["horizon_length"]=1; c["drift_temps"]=3.0
    for k,v in kw.items(): c[k]=v
    return cls.create(0, np.zeros((1,29),np.float32), np.zeros((1,8),np.float32), c)
print(f"{'config':40s} {'|grad| total':>13s} {'Q only':>10s} {'penalty/Q':>11s}")
for tag, cls, cfg, kw in [
    ("anq_stdfp lam=5.0 anchor=data",  ANQSTDFPAgent, cfg1, dict(lam=5.0)),
    ("mani_stdfp lam=0.1 anchor=data", ManiSTDFPAgent, cfgm, dict(lam=0.1)),
    ("mani_stdfp lam=0.02 anchor=data",ManiSTDFPAgent, cfgm, dict(lam=0.02)),
    ("mani_stdfp lam=0.1 anchor=base", ManiSTDFPAgent, cfgm, dict(lam=0.1, refine_anchor="base")),
]:
    tot = gn(mk(cls, cfg, **kw)); q = gn(mk(cls, cfg, **{**kw, "lam": 0.0}))
    print(f"{tag:40s} {tot:13.3e} {q:10.3e} {(tot-q)/q:11.2f}")
