#!/usr/bin/env bash
# rm_trqam.sh <gpu_id> <task:lift|can|square> <seed> <kl_budget>
# TRQAM baseline pipeline on robomimic: BC pretrain 300k (cached per task) then
# 1M offline fine-tune.  Runs from /workspace/trqam.
set -u
GPU=$1; TASK=$2; SEED=$3; KL=$4
ENV="${TASK}-mh-low_dim"
cd /workspace/trqam
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10 XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline WANDB_SILENT=true JAX_PLATFORMS=cuda
LOGDIR=/workspace/svgd_qc/exp_logs; mkdir -p "$LOGDIR"

BC_CKPT=$(find exp/trqam/bc_pretrain -path "*/${ENV}/*/params_300000.pkl" 2>/dev/null | head -1)
if [ -z "$BC_CKPT" ]; then
  /venv/main/bin/python main.py --run_group=bc_pretrain --agent=agents/trqam.py --tags=BC --seed=10001 \
    --env_name=$ENV --sparse=False --horizon_length=5 \
    --agent.action_chunking=True --bc_only=True --offline_steps=300000 --online_steps=0 \
    --eval_interval=300000 --eval_episodes=10 --video_episodes=0 --save_interval=300000 --save_last_checkpoint=True \
    --save_dir=exp/trqam/bc_pretrain > "$LOGDIR/TRQAM_BC_${TASK}.log" 2>&1
  BC_CKPT=$(find exp/trqam/bc_pretrain -path "*/${ENV}/*/params_300000.pkl" 2>/dev/null | head -1)
fi
[ -z "$BC_CKPT" ] && { echo "BC pretrain failed for $ENV"; exit 1; }

/venv/main/bin/python main.py --run_group=RMTRQAM_${TASK}_s${SEED} --agent=agents/trqam.py --tags=TRQAM --seed=$SEED \
  --env_name=$ENV --sparse=False --horizon_length=5 \
  --agent.action_chunking=True --pretrained_actor_path=$BC_CKPT \
  --agent.kl_budget=$KL --offline_steps=1000000 --online_steps=0 \
  --eval_interval=100000 --eval_episodes=50 --video_episodes=0 --save_interval=-1 \
  --save_dir=/workspace/svgd_qc/exp/beat \
  > "$LOGDIR/RMTRQAM_${TASK}_s${SEED}.log" 2>&1
echo "TRQAM ${TASK} s${SEED} done"
