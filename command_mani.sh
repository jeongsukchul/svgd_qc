for lam in 50; do
        MUJOCO_GL=egl python main.py  --agent=agents/mani_stdfp.py   --env_name=cube-double-play-singletask-task1-v0 \
                --discount=0.99 --agent.lam=$lam --agent.gen_per_label=8  --agent.refine_base_source=latent --horizon_length=5
        MUJOCO_GL=egl python main.py  --agent=agents/mani_stdfp.py   --env_name=cube-double-play-singletask-task2-v0 \
                --discount=0.99 --agent.lam=$lam --agent.gen_per_label=8  --agent.refine_base_source=latent --horizon_length=5
        MUJOCO_GL=egl python main.py  --agent=agents/mani_stdfp.py   --env_name=cube-double-play-singletask-task3-v0 \
                --discount=0.99 --agent.lam=$lam --agent.gen_per_label=8  --agent.refine_base_source=latent --horizon_length=5
        MUJOCO_GL=egl python main.py  --agent=agents/mani_stdfp.py   --env_name=cube-double-play-singletask-task4-v0 \
                --discount=0.99 --agent.lam=$lam --agent.gen_per_label=8  --agent.refine_base_source=latent --horizon_length=5
        MUJOCO_GL=egl python main.py  --agent=agents/mani_stdfp.py   --env_name=cube-double-play-singletask-task5-v0 \
                --discount=0.99 --agent.lam=$lam --agent.gen_per_label=8  --agent.refine_base_source=latent --horizon_length=5
done
