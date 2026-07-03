#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ENV_NAME=${ENV_NAME:-cube-double-play-singletask-task2-v0}
SEED=${SEED:-123}
HORIZON_LENGTH=${HORIZON_LENGTH:-5}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
ONLINE_STEPS=${ONLINE_STEPS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
RUN_GROUP=${RUN_GROUP:-dsrl-reg-compare}
MUJOCO_GL=${MUJOCO_GL:-egl}

COMMON=(
  main.py
  --run_group="${RUN_GROUP}"
  --seed="${SEED}"
  --env_name="${ENV_NAME}"
  --sparse=False
  --horizon_length="${HORIZON_LENGTH}"
  --agent=agents/dsrl.py
  --offline_steps="${OFFLINE_STEPS}"
  --online_steps="${ONLINE_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --agent.lr=3e-4
  --agent.rho=0.5
  --agent.best_of_n=1
  --agent.noise_scale=1.0
)

run_variant() {
  local name=$1
  shift
  echo
  echo "===== ${name} ====="
  echo "MUJOCO_GL=${MUJOCO_GL} ${PYTHON} ${COMMON[*]} $*"
  MUJOCO_GL="${MUJOCO_GL}" "${PYTHON}" "${COMMON[@]}" "$@"
}
# run_variant "entropy_auto_mult_0.5_lr3e4" \
#   --agent.regularizer=entropy \
#   --agent.target_entropy_multiplier=.5
# run_variant "entropy_auto_mult_0.5_lr3e4" \
#   --agent.regularizer=entropy \
#   --agent.target_entropy_multiplier=1.

for target_kl in 15 20 25; do
  run_variant "kl_${target_kl}_lr3e4" \
    --agent.regularizer=kl \
    --agent.target_kl="${target_kl}"
done
