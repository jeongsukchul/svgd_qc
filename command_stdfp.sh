#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/sukchul/miniconda3/envs/fql/bin/python}
ENV_NAME=${ENV_NAME:-cube-double-play-singletask-task2-v0}
SEED=${SEED:-123}
HORIZON_LENGTH=${HORIZON_LENGTH:-5}
OFFLINE_STEPS=${OFFLINE_STEPS:-600000}
ONLINE_STEPS=${ONLINE_STEPS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-50000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50000}
RUN_GROUP=${RUN_GROUP:-stdfp-fix8}
MUJOCO_GL=${MUJOCO_GL:-egl}

COMMON=(
  main.py
  --run_group="${RUN_GROUP}"
  --seed="${SEED}"
  --env_name="${ENV_NAME}"
  --sparse=False
  --horizon_length="${HORIZON_LENGTH}"
  --agent=agents/stdfp.py
  --offline_steps="${OFFLINE_STEPS}"
  --online_steps="${ONLINE_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --save_best_eval=True
  --agent.q_agg=pessimistic
  --agent.actor_q_agg=mean
  --agent.sample_q_agg=mean
  --agent.use_target_latent=True
  --agent.noise_actor_squash_tanh=False
  --agent.noise_log_prob_clip=100.0
)

run_variant() {
  local name=$1
  shift
  echo
  echo "===== ${name} ====="
  echo "MUJOCO_GL=${MUJOCO_GL} ${PYTHON} ${COMMON[*]} $*"
  MUJOCO_GL="${MUJOCO_GL}" "${PYTHON}" "${COMMON[@]}" "$@"
}

run_variant normal_statedep_auto_l2_0.3_lr1e-4 \
  --agent.lr=1e-4 \
  --agent.noise_state_dependent_std=True \
  --agent.noise_log_std_min=-5.0 \
  --agent.noise_log_std_max=-0.5 \
  --agent.noise_pre_tanh_l2=0.3 \
  --agent.noise_scale=1.0

run_variant normal_statedep_auto_l2_0.25_lr1e-4 \
  --agent.lr=1e-4 \
  --agent.noise_state_dependent_std=True \
  --agent.noise_log_std_min=-5.0 \
  --agent.noise_log_std_max=-0.5 \
  --agent.noise_pre_tanh_l2=0.25 \
  --agent.noise_scale=1.0

run_variant normal_statedep_auto_l2_0.35_lr1e-4 \
  --agent.lr=1e-4 \
  --agent.noise_state_dependent_std=True \
  --agent.noise_log_std_min=-5.0 \
  --agent.noise_log_std_max=-0.5 \
  --agent.noise_pre_tanh_l2=0.35 \
  --agent.noise_scale=1.0

run_variant normal_statedep_coeff_0.35_l2_0.3_lr1e-4 \
  --agent.lr=1e-4 \
  --agent.noise_state_dependent_std=True \
  --agent.noise_log_std_min=-5.0 \
  --agent.noise_log_std_max=-0.5 \
  --agent.noise_normal_target_entropy_multiplier=0.35 \
  --agent.noise_pre_tanh_l2=0.3 \
  --agent.noise_scale=1.0

run_variant normal_statedep_coeff_0.50_l2_0.3_lr1e-4 \
  --agent.lr=1e-4 \
  --agent.noise_state_dependent_std=True \
  --agent.noise_log_std_min=-5.0 \
  --agent.noise_log_std_max=-0.5 \
  --agent.noise_normal_target_entropy_multiplier=0.50 \
  --agent.noise_pre_tanh_l2=0.3 \
  --agent.noise_scale=1.0
