#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ENV_NAME=${ENV_NAME:-cube-double-play-singletask-task4-v0}
HORIZON_LENGTH=${HORIZON_LENGTH:-5}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
ONLINE_STEPS=${ONLINE_STEPS:-0}
FREEZE_AFTER=${FREEZE_AFTER:-300000}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
RUN_GROUP=${RUN_GROUP:-dfp-drift-freeze}
MUJOCO_GL=${MUJOCO_GL:-egl}
SEED=${SEED:-0}

echo "Running DFP with actor drift frozen after ${FREEZE_AFTER} updates"
MUJOCO_GL="${MUJOCO_GL}" "${PYTHON}" main.py \
  --run_group="${RUN_GROUP}" \
  --env_name="${ENV_NAME}" \
  --sparse=False \
  --horizon_length="${HORIZON_LENGTH}" \
  --agent=agents/dfp.py \
  --agent.actor_drift_network=vector_field \
  --agent.actor_layer_norm=False \
  --agent.actor_type=best-of-n \
  --agent.actor_num_samples=16 \
  --agent.q_agg=mean \
  --agent.freeze_actor_drift_after="${FREEZE_AFTER}" \
  --offline_steps="${OFFLINE_STEPS}" \
  --online_steps="${ONLINE_STEPS}" \
  --log_interval="${LOG_INTERVAL}" \
  --eval_interval="${EVAL_INTERVAL}" \
  --save_interval=0 \
  --save_best_eval=True \
  --seed="${SEED}"
