#!/usr/bin/env bash
set -euo pipefail

# AntMaze-oriented search for the critic-only ANQ drift variants.
VARIANT=${VARIANT:-anq_dfp}
ENV_NAME=${ENV_NAME:-antmaze-large-navigate-singletask-v0}
STAGE=${STAGE:-radius}
SEEDS=${SEEDS:-"0 1 2"}
RADII=${RADII:-"0.05 0.1 0.2 0.4"}
EXPECTILES=${EXPECTILES:-"0.7 0.8 0.9"}
Q_AGG=${Q_AGG:-min}
REFINE_STEPS=${REFINE_STEPS:-3}
REFINE_STEP_SIZE=${REFINE_STEP_SIZE:-0.05}
DISCOUNT=${DISCOUNT:-0.995}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_GROUP=${RUN_GROUP:-${VARIANT}-antmaze-${STAGE}}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}

case "${VARIANT}" in
  anq_dfp|anq_stdfp)
    AGENT_PATH="agents/${VARIANT}.py"
    ;;
  *)
    echo "VARIANT must be 'anq_dfp' or 'anq_stdfp'" >&2
    exit 2
    ;;
esac

run_one() {
  local seed=$1
  local radius=$2
  local expectile=$3
  MUJOCO_GL=${MUJOCO_GL:-egl} "${PYTHON_BIN}" main.py \
    --run_group="${RUN_GROUP}" \
    --env_name="${ENV_NAME}" \
    --agent="${AGENT_PATH}" \
    --seed="${seed}" \
    --offline_steps="${OFFLINE_STEPS}" \
    --online_steps=0 \
    --eval_interval="${EVAL_INTERVAL}" \
    --discount="${DISCOUNT}" \
    --horizon_length=1 \
    --agent.action_chunking=False \
    --agent.q_agg="${Q_AGG}" \
    --agent.critic_expectile="${expectile}" \
    --agent.refine_radius="${radius}" \
    --agent.refine_steps="${REFINE_STEPS}" \
    --agent.refine_step_size="${REFINE_STEP_SIZE}"
}

case "${STAGE}" in
  radius)
    for radius in ${RADII}; do
      run_one "${SEEDS%% *}" "${radius}" 0.9
    done
    ;;
  expectile)
    # Set RADII to the best one or two values from the radius stage.
    for radius in ${RADII}; do
      for expectile in ${EXPECTILES}; do
        for seed in ${SEEDS}; do
          run_one "${seed}" "${radius}" "${expectile}"
        done
      done
    done
    ;;
  *)
    echo "STAGE must be 'radius' or 'expectile'" >&2
    exit 2
    ;;
esac
