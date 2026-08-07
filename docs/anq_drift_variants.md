# Critic-only ANQ drift variants

These are experimental variants requested for this repository; they are not
the published ANQ algorithm.  Both remove the learned value function,
auxiliary delta actor, and final weighted-regression policy from ANQ.

For a behavior-cloned drift action `a_b = f(s, z)`, the inner neighborhood
optimization is performed explicitly:

```text
a_0 = a_b
a_{k+1} = project_to_action_and_radius(
    a_k + refine_step_size * normalized_grad_a Q(s, a_k)
)
delta = a_K - a_b
```

The projection guarantees `||delta||_2 <= refine_radius`.  The critic target is

```text
y = reward + discount * mask * aggregate(target_Q(s', a_b' + delta'))
```

and each critic is trained with an asymmetric expectile loss on `y - Q(s, a)`.
There is no separately learned `V(s)`.

## Variants

- [`agents/anq_dfp.py`](../agents/anq_dfp.py): samples `z` from a unit Gaussian.
  The drift decoder is trained only with behavior-cloning drift loss; decoded
  actions are Q-refined for TD targets and execution.
- [`agents/anq_stdfp.py`](../agents/anq_stdfp.py): adds STDFP's learned latent
  noise actor.  It selects the behavior mode in latent space, while the decoded
  action receives the same bounded action-space refinement.  A straight-through
  delta avoids second-order gradients through the inner Q optimization.

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
  --agent.critic_expectile=0.9 \
  --agent.refine_radius=0.2

# Learned latent-noise actor.
MUJOCO_GL=egl python main.py \
  --agent=agents/anq_stdfp.py \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --discount=0.995 \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --agent.q_agg=min \
  --agent.critic_expectile=0.9 \
  --agent.refine_radius=0.2
```

## Tuning region

Tune the neighborhood before the expectile:

| Parameter | First-pass values | Notes |
|---|---:|---|
| `refine_radius` | `0.05, 0.1, 0.2, 0.4` | Main neighborhood-size control in normalized action space. |
| `refine_steps` | `1, 3, 5` | Three is the default. |
| `refine_step_size` | `0.025, 0.05, 0.1` | With normalized gradients, choose enough total movement to reach the radius. |
| `critic_expectile` | `0.7, 0.8, 0.9` | Higher more strongly fits positive TD residuals and is less conservative. |
| `q_agg` | `min, mean` | Start with `min` for offline AntMaze. |

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

Monitor `critic/target_delta_rms` alongside success and Q magnitude.  If the
delta sits at the radius while Q grows without evaluation gains, reduce the
radius or switch from `mean` to `min` aggregation.
