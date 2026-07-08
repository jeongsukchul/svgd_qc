#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ENV_NAME=${ENV_NAME:-cube-double-play-singletask-task2-v0}
HORIZON_LENGTH=${HORIZON_LENGTH:-5}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
ONLINE_STEPS=${ONLINE_STEPS:-0}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
RUN_GROUP=${RUN_GROUP:-stdfp-reg-sweep}
MUJOCO_GL=${MUJOCO_GL:-egl}

COMMON=(
  main.py
  --run_group="${RUN_GROUP}"
  --env_name="${ENV_NAME}"
  --sparse=False
  --horizon_length="${HORIZON_LENGTH}"
  --agent=agents/stdfp.py
  --offline_steps="${OFFLINE_STEPS}"
  --online_steps="${ONLINE_STEPS}"
  --log_interval="${LOG_INTERVAL}"
  --eval_interval="${EVAL_INTERVAL}"
  --save_interval="${SAVE_INTERVAL}"
  --agent.actor_type=sac
  # --agent.lr=3e-4
  --agent.actor_q_agg=mean
  --agent.sample_q_agg=mean
  --agent.use_target_latent=True
)

run_variant() {
  local name=$1
  shift
  echo
  echo "===== ${name} ====="
  echo "MUJOCO_GL=${MUJOCO_GL} ${PYTHON} ${COMMON[*]} $*"
  MUJOCO_GL="${MUJOCO_GL}" "${PYTHON}" "${COMMON[@]}" "$@"
}
# MUJOCO_GL="${MUJOCO_GL}" "${PYTHON}"  "$@" main.py --agent=agents/trqam.py --horizon_length=5

# for target_kl in 10 20 30 40 50; do
#   run_variant "kl_postsquash_${target_kl}_lr1e4" \
#     --agent.noise_regularizer=kl \
#     --agent.noise_target_kl="${target_kl}" \
#     --agent.noise_scale=1.0
# done
# for target_kl in 10 20 30 40 50; do
#   run_variant "kl_postsquash_${target_kl}_lr1e4" \
#     --agent.noise_regularizer=kl \
#     --agent.noise_target_kl="${target_kl}" \
#     --agent.noise_scale=2.0 \
#     --agent.noise_state_dependent_std=True
# done
for target_kl in 40 50; do
  run_variant "kl_postsquash_${target_kl}_lr1e4" \
    --agent.noise_regularizer=kl \
    --agent.noise_target_kl="${target_kl}" \
    --agent.noise_scale=2.0 \
    --agent.noise_state_dependent_std=False
done
for target_kl in 30 40 50; do
  run_variant "kl_postsquash_${target_kl}_lr1e4" \
    --agent.noise_regularizer=kl \
    --agent.noise_target_kl="${target_kl}" \
    --agent.noise_scale=1.0 \
    --agent.noise_state_dependent_std=True
done
for target_kl in 30 40 50; do
  run_variant "kl_postsquash_${target_kl}_lr1e4" \
    --agent.noise_regularizer=kl \
    --agent.noise_target_kl="${target_kl}" \
    --agent.noise_scale=1.0 \
    --agent.noise_state_dependent_std=False
done