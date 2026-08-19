#!/usr/bin/env bash
cd /home/sukchul/qc
SP=/tmp/claude-1000/-home-sukchul-qc/682f546f-7056-4a11-8268-2787858681d7/scratchpad
PY=/home/sukchul/miniconda3/envs/fql/bin/python
while pgrep -f "E_rebrac_a5_t3" | grep -qv $$ && pgrep -f "main.py --agent=agents/rebrac.py" > /dev/null; do
  sleep 60
done
XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 MUJOCO_GL=egl $PY main.py --agent=agents/mani_stdfp.py \
  --env_name=antmaze-giant-navigate-singletask-task3-v0 --discount=0.995 \
  --offline_steps=1000000 --eval_interval=100000 --eval_episodes=50 --save_interval=-1 --seed=0 \
  --save_dir=exp/beat --agent.drift_temps=3.0 --agent.lam=0.00013 --agent.refine_anchor=data \
  --run_group=I_mani_weak_t3 > $SP/I_mani_weak_t3.log 2>&1
