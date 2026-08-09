# ANQ2: value-free ANQ on OGBench

[`agents/anq2.py`](../agents/anq2.py) is an experimental, standalone ANQ
variant. It removes the value network and value loss entirely. Its only target
network is `target_critic`.

The four trainable/current modules are:

```text
actor            final deterministic execution policy
aux_actor        local action-delta refiner
critic           Q ensemble
target_critic    EMA critic used for TD targets and policy weights
```

## Objectives

Given a dataset action `a`:

```text
delta = aux_action_scale * aux_actor(s, a)
a_refined = clip(a + delta, -1, 1)
```

ANQ2 trains the refiner with a fixed neighborhood penalty:

```text
L_refine = -normalized_refine_Q(s, a_refined) + lam * ||delta||^2
```

Without `V(s)`, the policy-extraction advantage is the local critic
improvement:

```text
improvement = refine_Q(s, a_refined) - data_Q(s, a)
weight = clip(exp(beta * improvement), actor_weight_min, actor_weight_max)
L_actor = weight * ||actor(s) - a_refined||^2
```

For TD learning, sequence batches provide the next dataset action. ANQ2
refines that action and evaluates it with the target critic:

```text
y = reward + discount^H * mask
    * aggregate(target_Q(s_next, refine(s_next, a_data_next)))
L_critic = expectile_loss(y - Q(s, a_data), critic_expectile)
```

If a custom batch does not contain `next_actions`, the current actor action is
used as the TD base action instead.

## AntMaze command

```bash
MUJOCO_GL=egl python main.py \
  --agent=agents/anq2.py \
  --env_name=antmaze-giant-navigate-singletask-task5-v0 \
  --offline_steps=1000000 \
  --online_steps=0 \
  --discount=0.995 \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --agent.lam=0.1 \
  --agent.critic_expectile=0.7 \
  --agent.q_agg=mean \
  --agent.data_q_agg=mean \
  --agent.refine_q_agg=min \
  --agent.beta=10 \
  --agent.aux_action_scale=1 \
  --agent.actor_weight_max=100
```

## Initial tuning region

| Parameter | Values | Role |
|---|---:|---|
| `lam` | `0.03, 0.05, 0.1, 0.2, 0.3` | Main refinement-size control. |
| `critic_expectile` | `0.5, 0.7, 0.9` | Asymmetry of TD regression. |
| `q_agg` | `mean, min` | Bootstrap-target aggregation. |
| `data_q_agg` | `mean, min` | Baseline Q used for local improvement. |
| `refine_q_agg` | `min, mean` | Refiner objective and improved-action Q. |
| `beta` | `3, 10` | Local-improvement policy weight temperature. |
| `aux_action_scale` | `0.5, 1, 2` | Maximum componentwise raw refinement scale. |

Start with the command above, screen `lam`, and then test
`critic_expectile`. Because this is a new value-free objective, use at least
three seeds before treating any setting as optimal.
