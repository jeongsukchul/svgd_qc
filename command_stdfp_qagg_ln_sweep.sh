#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ENV_NAME=${ENV_NAME:-cube-double-play-singletask-task4-v0}
HORIZON_LENGTH=${HORIZON_LENGTH:-5}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
ONLINE_STEPS=${ONLINE_STEPS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
RUN_GROUP=${RUN_GROUP:-stdfp-qagg-ln-sweep}
MUJOCO_GL=${MUJOCO_GL:-egl}
SEED=${SEED:-0}

COMMON=(
  main.py
  --run_group="${RUN_GROUP}"
  --env_name="${ENV_NAME}"
  --sparse=False
  --horizon_length="${HORIZON_LENGTH}"
  --agent=agents/stdfp.py
  --agent.actor_type=sac
  --agent.noise_regularizer=kl
  --offline_steps="${OFFLINE_STEPS}"
  --online_steps="${ONLINE_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --seed="${SEED}"
)

run_variant() {
  local actor_layer_norm=$1
  local sample_q_agg=$2
  local actor_q_agg=$3
  local name="ln${actor_layer_norm}_sample-${sample_q_agg}_actor-${actor_q_agg}"

  echo
  echo "===== ${name} ====="
  echo "MUJOCO_GL=${MUJOCO_GL} ${PYTHON} ${COMMON[*]} --agent.actor_layer_norm=${actor_layer_norm} --agent.sample_q_agg=${sample_q_agg} --agent.actor_q_agg=${actor_q_agg}"
  MUJOCO_GL="${MUJOCO_GL}" "${PYTHON}" "${COMMON[@]}" \
    --agent.actor_layer_norm="${actor_layer_norm}" \
    --agent.sample_q_agg="${sample_q_agg}" \
    --agent.actor_q_agg="${actor_q_agg}"
}

for actor_layer_norm in True False; do
  for sample_q_agg in pessimistic; do
    for actor_q_agg in mean; do
      run_variant "${actor_layer_norm}" "${sample_q_agg}" "${actor_q_agg}"
    done
  done
done
