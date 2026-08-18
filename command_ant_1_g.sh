for drift_temps in 1.; do
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-giant-navigate-singletask-task1-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps --offline_steps=500000 --agent.target_multiplier=0.8
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-giant-navigate-singletask-task2-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps --offline_steps=500000 --agent.target_multiplier=0.8
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-giant-navigate-singletask-task3-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps --offline_steps=500000 --agent.target_multiplier=0.8
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-giant-navigate-singletask-task4-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps --offline_steps=500000 --agent.target_multiplier=0.8  
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-giant-navigate-singletask-task5-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps --offline_steps=500000 --agent.target_multiplier=0.8
done
