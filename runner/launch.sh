#!/usr/bin/env bash
# launch.sh <gpu_id> <mem_frac> <run_group> <agent_path> <task_num> <seed> [extra args...]
set -u
GPU=$1; MEMF=$2; GROUP=$3; AGENT=$4; TASK=$5; SEED=$6; shift 6
cd /workspace/svgd_qc
LOGDIR=/workspace/svgd_qc/exp_logs
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=$MEMF
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export WANDB_MODE=offline
export WANDB_SILENT=true
export JAX_PLATFORMS=cuda
nohup /venv/main/bin/python main.py \
  --agent="$AGENT" \
  --env_name=antmaze-giant-navigate-singletask-task${TASK}-v0 \
  --discount=0.995 \
  --offline_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=${EVAL_EPISODES:-50} \
  --save_interval=-1 \
  --seed=$SEED \
  --save_dir=exp/beat \
  --run_group="$GROUP" \
  --offline_scan_chunk=25 \
  "$@" > "$LOGDIR/${GROUP}.log" 2>&1 &
echo "launched $GROUP pid=$! gpu=$GPU"
