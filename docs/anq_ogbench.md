# ANQ on OGBench single-task environments

The implementation in [`agents/anq.py`](../agents/anq.py) ports the objectives
from the [ANQ paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/254009e8d528f98764a060e877a1b01c-Abstract-Conference.html)
and [released implementation](https://github.com/thu-rllab/ANQ) to this
repository's JAX/Flax agent interface.  It uses OGBench's `masks` for Bellman
bootstrapping, not `terminals`: OGBench trajectories may end without task
completion, while `masks == 0` specifically denotes success.

This implementation targets state-based, bounded continuous-action OGBench
tasks.  It does not yet include a pixel encoder or support Powderworld's
discrete action space.

## Run a baseline

Manipulation (`cube`, `scene`, or `puzzle`):

```bash
MUJOCO_GL=egl python main.py \
  --agent=agents/anq.py \
  --env_name=cube-double-play-singletask-task2-v0 \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --agent.lam=3 \
  --agent.alpha=1
```

Long-horizon navigation (`pointmaze`, `antmaze`, `humanoidmaze`, or
`antsoccer`) uses the paper's sparse-reward profile:

```bash
MUJOCO_GL=egl python main.py \
  --agent=agents/anq.py \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --discount=0.995 \
  --horizon_length=1 \
  --agent.action_chunking=False \
  --agent.expectile=0.9 \
  --agent.beta=10 \
  --agent.aux_weight_max=10 \
  --agent.actor_weight_max=100 \
  --agent.lam=3 \
  --agent.alpha=1
```

The faithful ANQ setting is `horizon_length=1` and
`agent.action_chunking=False`.  The implementation also supports action chunks,
but that is an experimental extension of ANQ rather than the published method.

## Recommended fine-tuning region

Tune in stages; `lambda` and `alpha` interact, so a full Cartesian sweep is
usually wasteful.

| Parameter | First-pass region | Refine around | Interpretation |
|---|---:|---:|---|
| `agent.lam` | `0.1, 0.3, 1, 3, 5, 10` | best value divided/multiplied by about 2 | Larger means tighter neighborhoods. This is the main parameter. |
| `agent.alpha` | fix at `1` | `0.5, 1, 2`; optionally `0` as the non-adaptive ablation | Larger makes radii more sensitive to data quality. |
| `agent.expectile` | `0.7` manipulation, `0.9` navigation | `0.7, 0.8, 0.9` | Higher makes the implicit outer maximization more optimistic. |
| `agent.beta` | `3` manipulation, `10` navigation | `1, 3, 10` | Policy-extraction advantage temperature. |
| `agent.q_agg` | `mean` | `min, mean` | Aggregation for the refined-Q target used by the value loss. |
| `agent.data_q_agg` | `mean` | `min, mean` | Aggregation of `Q(s, a_data)` in the adaptive radius calculation. |
| `agent.refine_q_agg` | `min` | `min, mean` | Aggregation of Q for the refiner objective and final actor weighting. |
| `discount` | `0.99` manipulation, `0.995` long-horizon navigation | `0.99, 0.995` | Use the longer horizon only where delayed success requires it. |

Start with the `lambda` stage on one seed and at least 200k updates.  Discard
settings whose critic values or auxiliary displacement grow sharply while
success stays flat.  Re-run the best two settings with at least three seeds,
then tune `alpha`.  Final comparisons should use the full 1M updates and at
least five seeds.

Dataset quality gives a useful prior:

- narrow expert-like or demonstration-heavy data: start at `lam=5` or `10`;
- broad `play`, mixed-quality, or noisy data: start at `lam=0.3`, `1`, or `3`;
- small/reduced datasets: include `lam=0.1` and `0.3`, but watch Q divergence.

Useful stability signals are logged as `critic/q_mean`, `critic/q_max`,
`aux_actor/delta_rms`, `aux_actor/radius_weight`, and `actor/weight`.  If Q and
`delta_rms` rise without evaluation improvement, increase `lam`; if
`delta_rms` remains near zero and ANQ behaves like sample-constrained IQL,
decrease it.

The provided staged runner implements this region:

```bash
# Stage 1: one-seed lambda screen.
ENV_NAME=antmaze-large-navigate-singletask-v0 \
PROFILE=navigation STAGE=lambda \
bash scripts/run_anq_ogbench_sweep.sh

# Stage 2: finalists across lambda, alpha, and seeds.
ENV_NAME=antmaze-large-navigate-singletask-v0 \
PROFILE=navigation STAGE=alpha LAMS="1 3" ALPHAS="0.5 1 2" SEEDS="0 1 2" \
bash scripts/run_anq_ogbench_sweep.sh
```

The sweep exposes all three reductions independently. For example:

```bash
Q_AGG=mean DATA_Q_AGG=mean REFINE_Q_AGG=min \
bash scripts/run_anq_ogbench_sweep.sh
```

This mixed setting matches the best logged current-architecture AntMaze run:
the value/data paths use ensemble means, while Q-directed refinement remains
pessimistic.

The paper searched `lam` mainly in `{0.1, 5}` and fixed `alpha=1`; the wider
region above is intentional because OGBench spans substantially different
action dimensions, horizons, coverage, and data quality.
