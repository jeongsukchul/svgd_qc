for lam in .01; do
        MUJOCO_GL=egl python main.py   --agent=agents/anq_stdfp2.py   --env_name=antmaze-giant-navigate-singletask-task1-v0 \
                --discount=0.995 --agent.drift_temps=0.2 --agent.target_multiplier=0.8 --agent.lam=$lam
        MUJOCO_GL=egl python main.py   --agent=agents/anq_stdfp2.py   --env_name=antmaze-giant-navigate-singletask-task2-v0 \
                --discount=0.995 --agent.drift_temps=0.2 --agent.target_multiplier=0.8 --agent.lam=$lam
        MUJOCO_GL=egl python main.py   --agent=agents/anq_stdfp2.py   --env_name=antmaze-giant-navigate-singletask-task3-v0 \
                --discount=0.995 --agent.drift_temps=0.2 --agent.target_multiplier=0.8 --agent.lam=$lam
        MUJOCO_GL=egl python main.py   --agent=agents/anq_stdfp2.py   --env_name=antmaze-giant-navigate-singletask-task4-v0 \
                --discount=0.995 --agent.drift_temps=0.2 --agent.target_multiplier=0.8 --agent.lam=$lam
        MUJOCO_GL=egl python main.py   --agent=agents/anq_stdfp2.py   --env_name=antmaze-giant-navigate-singletask-task5-v0 \
                --discount=0.995 --agent.drift_temps=0.2 --agent.target_multiplier=0.8 --agent.lam=$lam
done
