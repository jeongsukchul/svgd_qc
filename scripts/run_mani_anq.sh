for lam in 10 20 30; do
    for noise_target_kl in 8; do
        for alpha in 0 1; do
            MUJOCO_GL=egl python main.py   --agent=agents/mani_stdfp.py   --env_name=antmaze-giant-navigate-singletask-task5-v0 \
                 --discount=0.995 --agent.lam=$lam  --agent.noise_target_kl=$noise_target_kl --agent.alpha=$alpha --agent.refine_base_source=latent
        done
    done
done
