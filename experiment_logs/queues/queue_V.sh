#!/usr/bin/env bash
cd /home/sukchul/qc
SP=/tmp/claude-1000/-home-sukchul-qc/682f546f-7056-4a11-8268-2787858681d7/scratchpad
PY=/home/sukchul/miniconda3/envs/fql/bin/python
C="--discount=0.995 --offline_steps=1000000 --eval_interval=100000 --eval_episodes=50 --save_interval=-1 --seed=0 --save_dir=exp/beat --agent.drift_temps=3.0 --agent.lam=0.01 --agent.refine_anchor=data --agent.refine_residual_space=pretanh"
while pgrep -f "Q1_rebrac_t2_s1" > /dev/null; do sleep 60; done
XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 MUJOCO_GL=egl $PY main.py --agent=agents/anq_stdfp.py \
  --env_name=antmaze-giant-navigate-singletask-task1-v0 $C --agent.base_scale=0.5 \
  --run_group=V_bs05_t1 > $SP/V_bs05_t1.log 2>&1
