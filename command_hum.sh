for lam in 0.1 1 10; do
        MUJOCO_GL=egl python main_pretrain_bc2.py   --agent=agents/mani_stdfp.py   --env_name=humanoidmaze-medium-navigation-singletask-task1-v0 \
                --discount=0.995 --agent.lam=$lam  --agent.refine_base_source=latent --horizon_length=1
        MUJOCO_GL=egl python main_pretrain_bc2.py   --agent=agents/mani_stdfp.py   --env_name=humanoidmaze-medium-navigation-singletask-task2-v0 \
                --discount=0.995 --agent.lam=$lam  --agent.refine_base_source=latent --horizon_length=1
        MUJOCO_GL=egl python main_pretrain_bc2.py   --agent=agents/mani_stdfp.py   --env_name=humanoidmaze-medium-navigation-singletask-task3-v0 \
                --discount=0.995 --agent.lam=$lam  --agent.refine_base_source=latent --horizon_length=1
        MUJOCO_GL=egl python main_pretrain_bc2.py   --agent=agents/mani_stdfp.py   --env_name=humanoidmaze-medium-navigation-singletask-task4-v0 \
                --discount=0.995 --agent.lam=$lam  --agent.refine_base_source=latent --horizon_length=1
        MUJOCO_GL=egl python main_pretrain_bc2.py   --agent=agents/mani_stdfp.py   --env_name=humanoidmaze-medium-navigation-singletask-task5-v0 \
                --discount=0.995 --agent.lam=$lam  --agent.refine_base_source=latent --horizon_length=1
done
