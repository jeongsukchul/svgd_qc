for lam in 0 5 10 20; do
    for noise_target_kl in 8 12; do
        for alpha in 0 1 2 4; do
            MUJOCO_GL=egl python main.py   --agent=agents/anq_stdfp.py   --env_name=antmaze-giant-navigate-singletask-task5-v0 \
                 --discount=0.995 --agent.lam=$lam --agent.noise_target_kl=$noise_target_kl --agent.alpha=$alpha --offline_steps=400000
        done
    done
done
