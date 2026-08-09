#!/usr/bin/env bash
set -euo pipefail

# Staged ANQ search for an OGBench single-task environment.  Override any grid
# with an environment variable, e.g. LAMS="0.3 1 3" SEEDS="0 1 2".
ENV_NAME=${ENV_NAME:-antmaze-giant-navigate-singletask-task5-v0}
PROFILE=${PROFILE:-navigation}
STAGE=${STAGE:-lambda}
SEEDS=${SEEDS:-"0"}
LAMS=${LAMS:-"0.1 0.3 1 3 5 10"}
ALPHAS=${ALPHAS:-"0.5 1 2"}
Q_AGG=${Q_AGG:-mean}
DATA_Q_AGG=${DATA_Q_AGG:-mean}
REFINE_Q_AGG=${REFINE_Q_AGG:-min}
PYTHON_BIN=${PYTHON_BIN:-python}
RUN_GROUP=${RUN_GROUP:-anq-ogbench-${PROFILE}-${STAGE}}
OFFLINE_STEPS=${OFFLINE_STEPS:-1000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}

case "${PROFILE}" in
  manipulation)
    DISCOUNT=${DISCOUNT:-0.99}
    EXPECTILE=${EXPECTILE:-0.5}
    BETA=${BETA:-3}
    AUX_WEIGHT_MAX=${AUX_WEIGHT_MAX:-30}
    ACTOR_WEIGHT_MAX=${ACTOR_WEIGHT_MAX:-3}
    ;;
  navigation)
    DISCOUNT=${DISCOUNT:-0.995}
    EXPECTILE=${EXPECTILE:-0.5}
    BETA=${BETA:-10}
    AUX_WEIGHT_MAX=${AUX_WEIGHT_MAX:-10}
    ACTOR_WEIGHT_MAX=${ACTOR_WEIGHT_MAX:-100}
    ;;
  *)
    echo "PROFILE must be 'manipulation' or 'navigation'" >&2
    exit 2
    ;;
esac

run_one() {
  local seed=$1
  local lam=$2
  local alpha=$3
  MUJOCO_GL=${MUJOCO_GL:-egl} "${PYTHON_BIN}" main.py \
    --run_group="${RUN_GROUP}" \
    --env_name="${ENV_NAME}" \
    --agent=agents/anq.py \
    --seed="${seed}" \
    --offline_steps="${OFFLINE_STEPS}" \
    --online_steps=0 \
    --eval_interval="${EVAL_INTERVAL}" \
    --discount="${DISCOUNT}" \
    --horizon_length=1 \
    --agent.action_chunking=False \
    --agent.lam="${lam}" \
    --agent.alpha="${alpha}" \
    --agent.q_agg="${Q_AGG}" \
    --agent.data_q_agg="${DATA_Q_AGG}" \
    --agent.refine_q_agg="${REFINE_Q_AGG}" \
    --agent.expectile="${EXPECTILE}" \
    --agent.beta="${BETA}" \
    --agent.aux_weight_max="${AUX_WEIGHT_MAX}" \
    --agent.actor_weight_max="${ACTOR_WEIGHT_MAX}"
}

case "${STAGE}" in
  lambda)
    # First select neighborhood size using one seed; repeat finalists with 3+.
    for lam in ${LAMS}; do
      run_one "${SEEDS%% *}" "${lam}" 1
    done
    ;;
  alpha)
    # Set LAMS to the best one or two values from the lambda stage.
    for lam in ${LAMS}; do
      for alpha in ${ALPHAS}; do
        for seed in ${SEEDS}; do
          run_one "${seed}" "${lam}" "${alpha}"
        done
      done
    done
    ;;
  *)
    echo "STAGE must be 'lambda' or 'alpha'" >&2
    exit 2
    ;;
esac
