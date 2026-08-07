<div align="center">

# [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969)

## [[website](https://colinqiyangli.github.io/qc/)]      [[pdf](https://arxiv.org/pdf/2507.07969)]

</div>

<p align="center">
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="teaser figure" src="./assets/teaser.png" width="48%">
  </a>
  <a href="https://colinqiyangli.github.io/qc/">
    <img alt="aggregated results" src="./assets/agg.png" width="48%">
  </a>
</p>


## Overview
Q-chunking runs RL on a *temporally extended action (action chunking) space* with an expressive behavior constraint to leverage prior data for improved exploration and online sample efficiency.

## Installation
```bash
pip install -r requirements.txt
```


## Datasets
For robomimic, we assume the datasets are located at `~/.robomimic/lift/mh/low_dim_v15.hdf5`, `~/.robomimic/can/mh/low_dim_v15.hdf5`, and `~/.robomimic/square/mh/low_dim_v15.hdf5`. The datasets can be downloaded from https://robomimic.github.io/docs/datasets/robomimic_v0.1.html (under Method 2: Using Direct Download Links - Multi-Human (MH)).

For cube-quadruple, we use the 100M-size offline dataset. It can be downloaded from https://github.com/seohongpark/horizon-reduction via
```bash
wget -r -np -nH --cut-dirs=2 -A "*.npz" https://rail.eecs.berkeley.edu/datasets/ogbench/cube-quadruple-play-100m-v0/
```
and include this flag in the command line `--ogbench_dataset_dir=[realpath/to/your/cube-quadruple-play-100m-v0/]` to make sure it is using the 100M-size dataset.

## Reproducing paper results

We include the example command for all the methods we evaluate in our paper below. For `scene` and `puzzle-3x3` domains, use `--sparse=True`. We also release our plot data at [plot_data/README.md](plot_data/README.md).

```bash
# QC
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=32 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# BFN-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# BFN
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.actor_type=best-of-n --agent.actor_num_samples=4 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# QC-FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# FQL-n
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# FQL
MUJOCO_GL=egl python main.py --run_group=reproduce --agent.alpha=100 --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1

# RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=1 

# RLPD-AC
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5

# QC-RLPD
MUJOCO_GL=egl python main_online.py --env_name=cube-triple-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.bc_alpha=0.01
```

## Drift Flow Matching agent

`agents/dfm.py` is a time-conditioned mean-velocity policy. Unlike DFP's
direct noise-to-action prediction, it parameterizes transport as
`x_r = x_t + (r - t) * velocity(observation, x_t, t, r)`. At training time,
`time_grid_ratio` controls the fraction of groups sampled from adjacent
intervals of the inference grid; the remaining groups use uniformly sampled
time pairs. At inference, `num_flow_steps` controls the number of iterative
transports from time 0 to 1. The defaults are 10 steps and a 0.5 grid ratio.
The loss is selected through `agent.drift_backend`:

```bash
# Existing utils/drift_loss.py calculation
MUJOCO_GL=egl python main.py --agent=agents/dfm.py --agent.drift_backend=drift_loss --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=5 --online_steps=0 --agent.actor_num_samples=16

# Grouped Sinkhorn calculation from the DFM pseudocode
MUJOCO_GL=egl python main.py --agent=agents/dfm.py --agent.drift_backend=sinkhorn --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=5 --online_steps=0 --agent.actor_num_samples=16

# Conservative Gaussian log-KDE scalar loss
MUJOCO_GL=egl python main.py --agent=agents/dfm.py --agent.drift_backend=log_kde --agent.log_kde_bandwidth=0.4 --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=5 --online_steps=0 --agent.actor_num_samples=16
```

`actor_num_samples` controls the number of candidates ranked by the critic.
The Sinkhorn version uses `temp_pos`, `temp_neg`, and `sinkhorn_iters`; use a
positive odd iteration count so the final truncated plan has completed a row
projection. The `drift_loss.py` version uses `drift_temps`. These temperature
values are not numerically interchangeable because `drift_loss.py` rescales
distances and forces internally. The log-KDE version implements Algorithm 1's
Gaussian leave-one-out KDE objective and uses `log_kde_bandwidth`.

Set `num_flow_steps=1` and `time_grid_ratio=1.0` to recover endpoint-only
training and one-step inference.

The same log-KDE objective is also available for the original DFP direct
noise-to-action policy (DFP does not use the Sinkhorn backend):

```bash
MUJOCO_GL=egl python main.py --agent=agents/dfp.py --agent.drift_backend=log_kde --agent.log_kde_bandwidth=0.4 --env_name=cube-triple-play-singletask-task2-v0 --horizon_length=5 --online_steps=0 --agent.actor_num_samples=16
```

## Adaptive Neighborhood-constrained Q learning (ANQ)

A native JAX/Flax ANQ agent for OGBench single-task training is available in
`agents/anq.py`.  It ports the paper's value, critic, adaptive-neighborhood,
and weighted policy-extraction objectives and respects OGBench's task-completion
`masks`.  For a baseline command, task-family profiles, and a staged
hyperparameter region, see [docs/anq_ogbench.md](docs/anq_ogbench.md).

Critic-only drift variants without a learned value or auxiliary actor are also
available: `agents/anq_dfp.py` uses Gaussian drift latents, and
`agents/anq_stdfp.py` adds a learned latent-noise actor.  See
[docs/anq_drift_variants.md](docs/anq_drift_variants.md) for their objectives,
AntMaze commands, and refinement sweep.

```
@inproceedings{
  li2025reinforcement,
  title={Reinforcement Learning with Action Chunking},
  author={Qiyang Li and Zhiyuan Zhou and Sergey Levine},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=XUks1Y96NR}
}
```

## Acknowledgments
This codebase is built on top of [FQL](https://github.com/seohongpark/fql). The two rlpd_* folders are directly taken from [RLPD](https://github.com/ikostrikov/rlpd).
