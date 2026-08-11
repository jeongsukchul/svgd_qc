# Value-free ANQ drift variants

These are experimental variants requested for this repository; they are not
the published ANQ algorithm. Both remove the learned value function and final
weighted-regression policy from ANQ. They use a learned refinement actor rather
than iterative action-gradient ascent.

For a behavior-cloned drift action `a_b = f(s, z)`, the refiner predicts its
delta in one forward pass. `anq_dfp` retains its L2-radius projection;
`anq_stdfp` uses the componentwise scale below:

```text
raw_delta = refine_actor(s, a_b)
delta_dfp = project_to_l2_ball(refine_radius * raw_delta)
delta_stdfp = refine_action_scale * raw_delta
a_refined = clip(a_b + delta_variant, -1, 1)
```

In `anq_stdfp`, `refine_base_source` selects how `z` is obtained before
`actor_drift` creates `a_b`:

```text
latent: z ~ noise_actor(s) * latent_noise_scale
drift:  z ~ Normal(0, I)
a_b = actor_drift(s, z)
```

The selected base path is frozen during the refinement update. The same target
critic and the same `improvement_q_agg` evaluate both sides of the improvement:

```text
target_improvement = target_Q_agg(s, a_refined)
                   - target_Q_agg(s, a_b)
penalty_weight = clip(exp(-alpha * target_improvement),
                      refine_weight_min, refine_weight_max)
L_refine = -normalized_Q_aggregate(s, a_refined)
           + lam * stop_gradient(penalty_weight) * ||delta||^2
```

The target improvement is clipped before exponentiation. Positive improvement
relaxes the delta penalty; negative improvement tightens it. Set `alpha=0` for
the original unweighted penalty.

The critic target uses the current drift and refinement actors with stopped
target gradients, then evaluates the action with the target critic:

```text
y = reward + discount * mask
    * aggregate(target_Q(s', current_refine(current_drift(s'))))
```

Each critic is trained with an asymmetric expectile loss on `y - Q(s, a)`.
There is no separately learned `V(s)`. `target_critic` is the only EMA network;
the drift, latent, and refinement actors never have target copies.

## Variants

- [`agents/anq_dfp.py`](../agents/anq_dfp.py): samples `z` from a unit Gaussian.
  The drift decoder is trained only with behavior-cloning drift loss. The
  learned refinement actor provides the Q-directed action delta. The agent is
  standalone and does not inherit from `DFPAgent`.
- [`agents/anq_stdfp.py`](../agents/anq_stdfp.py): adds STDFP's learned latent
  noise actor. The latent actor selects a behavior mode, the drift decoder maps
  it to action space, and the refinement actor applies the final bounded delta.
  The agent is standalone and does not inherit from `STDFPAgent`.

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
  --agent.improvement_q_agg=min \
  --agent.refine_base_source=latent \
  --agent.critic_expectile=0.9 \
  --agent.refine_action_scale=1.0 \
  --agent.alpha=1.0 \
  --agent.lam=5.0
```

## Tuning region

Tune the learned neighborhood before the expectile:

| Parameter | First-pass values | Notes |
|---|---:|---|
| `refine_radius` (DFP) | `0.05, 0.1, 0.2, 0.4` | Hard L2 radius for `anq_dfp`. |
| `refine_action_scale` (STDFP) | `0.25, 0.5, 1, 2` | Componentwise scale before action clipping. |
| `refine_base_source` (STDFP) | `latent, drift` | Learned latent actor or direct unit-Gaussian drift input. |
| `lam` (STDFP) | `1, 5, 10` | Soft delta penalty. |
| `alpha` (STDFP) | `0.3, 1, 3` | Target-Q-improvement sensitivity of the STDFP delta penalty. |
| `improvement_q_agg` (STDFP) | `min, mean` | Shared aggregation for base and refined target Q. |
| `improvement_clip` (STDFP) | `3, 10` | Bounds target-Q differences before exponentiation. |
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
VARIANT=anq_dfp STAGE=expectile RADII="0.1 0.2" \
EXPECTILES="0.7 0.8 0.9" SEEDS="0 1 2" \
bash scripts/run_anq_drift_ogbench_sweep.sh
```

For `anq_stdfp`, run the command above directly while sweeping
`aux_action_scale`, `alpha`, and `critic_expectile`; the legacy drift sweep
script's `RADII` grid is specific to `anq_dfp`.

Monitor `actor/refine_delta_norm`, `actor/refine_penalty`,
`actor/refine_improvement_weight`, and
`critic/target_delta_rms` alongside success and Q magnitude. If the delta stays
large without evaluation gains, reduce `refine_action_scale`, increase `lam`,
or use `min` aggregation.
