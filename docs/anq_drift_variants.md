# Value-free ANQ drift variants

These are experimental variants requested for this repository; they are not
the published ANQ algorithm. Both remove the learned value function and final
weighted-regression policy from ANQ. They use a learned refinement actor rather
than iterative action-gradient ascent.

For a behavior-cloned drift action `a_b = f(s, z)`, the refiner predicts a
bounded delta in one forward pass:

```text
raw_delta = refine_actor(s, a_b)
delta = project_to_l2_ball(refine_radius * raw_delta)
a_refined = clip(a_b + delta, -1, 1)
```

The refiner maximizes the frozen critic objective with a fixed neighborhood
penalty:

```text
L_refine = -normalized_Q_aggregate(s, a_refined)
           + refine_lambda * ||delta||^2
```

The critic target uses the slowly updated target refiner:

```text
y = reward + discount * mask
    * aggregate(target_Q(s', a_b' + target_delta'))
```

Each critic is trained with an asymmetric expectile loss on `y - Q(s, a)`.
There is no separately learned `V(s)`. `target_refine_actor` is Polyak-updated
with the same `tau` as the target critic.

## Variants

- [`agents/anq_dfp.py`](../agents/anq_dfp.py): samples `z` from a unit Gaussian.
  The drift decoder is trained only with behavior-cloning drift loss. The
  learned refinement actor provides the Q-directed action delta.
- [`agents/anq_stdfp.py`](../agents/anq_stdfp.py): adds STDFP's learned latent
  noise actor. The latent actor selects a behavior mode, the drift decoder maps
  it to action space, and the refinement actor applies the final bounded delta.

## AntMaze commands

```bash
# Gaussian behavior latent.
MUJOCO_GL=egl python main.py \
  --agent=agents/anq_dfp.py \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --discount=0.995 \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --agent.q_agg=min \
  --agent.refine_q_agg=min \
  --agent.critic_expectile=0.9 \
  --agent.refine_radius=0.2 \
  --agent.refine_lambda=5.0

# Learned latent-noise actor.
MUJOCO_GL=egl python main.py \
  --agent=agents/anq_stdfp.py \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --discount=0.995 \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --agent.q_agg=min \
  --agent.refine_q_agg=min \
  --agent.critic_expectile=0.9 \
  --agent.refine_radius=0.2 \
  --agent.refine_lambda=5.0
```

## Tuning region

Tune the learned neighborhood before the expectile:

| Parameter | First-pass values | Notes |
|---|---:|---|
| `refine_radius` | `0.05, 0.1, 0.2, 0.4` | Hard L2 bound in normalized action space. |
| `refine_lambda` | `1, 5, 10` | Soft delta penalty; raise it if the refiner stays at the radius. |
| `critic_expectile` | `0.7, 0.8, 0.9` | Higher values fit positive TD residuals more strongly. |
| `q_agg` | `min, mean` | Start with `min` for offline AntMaze targets. |
| `refine_q_agg` | `min, mean` | Start with `min` for conservative refiner gradients. |
| `refine_hidden_dims` | `(256,256)`, `(512,512)` | The two-layer 512 default is usually sufficient. |

The staged AntMaze sweep defaults to `anq_dfp`:

```bash
bash scripts/run_anq_drift_ogbench_sweep.sh

# Learned latent variant.
VARIANT=anq_stdfp bash scripts/run_anq_drift_ogbench_sweep.sh

# Refine the best radii over expectiles and seeds.
VARIANT=anq_stdfp STAGE=expectile RADII="0.1 0.2" \
EXPECTILES="0.7 0.8 0.9" SEEDS="0 1 2" \
bash scripts/run_anq_drift_ogbench_sweep.sh
```

Monitor `actor/refine_delta_norm`, `actor/refine_penalty`, and
`critic/target_delta_rms` alongside success and Q magnitude. If the delta stays
at the radius without evaluation gains, reduce the radius, increase
`refine_lambda`, or use `min` aggregation.
