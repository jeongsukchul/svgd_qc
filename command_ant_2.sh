for drift_temps in .5; do
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-large-navigate-singletask-task1-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps 
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-large-navigate-singletask-task2-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps 
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-large-navigate-singletask-task3-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps 
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-large-navigate-singletask-task4-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps 
        MUJOCO_GL=egl python main.py   --agent=agents/stdfp.py   --env_name=antmaze-large-navigate-singletask-task5-v0 \
                --discount=0.995 --agent.drift_temps=$drift_temps 
done
