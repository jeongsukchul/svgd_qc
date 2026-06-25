
#dfp tuning process

# MUJOCO_GL=egl python main.py -env_name=cube-double-play-singletask-task2-v0 \
# --sparse=False --horizon_length=5 --agent=agents/dfp.py --offline_steps=500000 \
# --agent.alpha_target=1. --agent.q_score_coeff=5000

# MUJOCO_GL=egl python main.py -env_name=cube-double-play-singletask-task2-v0 \
# --sparse=False --horizon_length=5 --agent=agents/dfp.py --offline_steps=500000 \
# --agent.alpha_target=2. --agent.q_score_coeff=1000

# MUJOCO_GL=egl python main.py -env_name=cube-double-play-singletask-task2-v0 \
# --sparse=False --horizon_length=5 --agent=agents/dfp.py --offline_steps=500000 \
# --agent.alpha_target=5. --agent.q_score_coeff=1000
#svgd tuning process
for epsilon in 0.1 0.01 1e-3 1e-4 
do
        for bandwidth in 0.1 0.01 1e-3 1-e4
        do
        MUJOCO_GL=egl python main.py -env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 \
                --agent=agents/svgd.py --agent.epsilon=$epsilon --agent.bandwidth=$bandwidth --offline_steps=400000
        done
done
# MUJOCO_GL=egl python main.py -env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 \
#         --agent=agents/svgd.py --agent.score_gain=10 --agent.epsilon=0.1 --agent.bandwidth=0.1 --offline_steps=400000