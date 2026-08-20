import glob, tqdm, wandb, os, json, random, shutil, sys, time, jax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_wandb_video

from envs.env_utils import is_robomimic_env_name, make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from utils.flax_utils import save_agent, save_critic
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = '0'
    os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval',  -1, 'Save interval.')
flags.DEFINE_bool('save_best_eval', False, 'Save a checkpoint whenever eval success/return improves.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_bool('prune_batch_keys', False, 'Drop batch keys the agent never reads (full_observations/terminals/next_actions) before the host->device copy.')

flags.DEFINE_integer('offline_scan_chunk', 1, 'Fuse this many offline updates into one lax.scan dispatch (1 = per-step, as before). Mathematically identical; purely a host-dispatch optimization.')

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/anq_stdfp.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 1, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")
class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def get_eval_success_postfix(eval_info):
    """Return a tqdm postfix showing eval success rate when the env reports one."""
    preferred_keys = (
        "success",
        "success_rate",
        "episode.success",
        "task_success",
        "is_success",
    )
    success_key = None
    for key in preferred_keys:
        if key in eval_info:
            success_key = key
            break
    if success_key is None:
        for key in eval_info:
            if "success" in key.lower():
                success_key = key
                break
    if success_key is None:
        return None

    value = eval_info[success_key]
    try:
        value = f"{float(np.asarray(value)):.3f}"
    except (TypeError, ValueError):
        value = str(value)
    return {"eval_success": value}


def get_eval_score(eval_info):
    """Return a comparable eval score tuple, preferring success then return."""
    preferred_success_keys = (
        "success",
        "success_rate",
        "episode.success",
        "task_success",
        "is_success",
    )
    success_key = None
    for key in preferred_success_keys:
        if key in eval_info:
            success_key = key
            break
    if success_key is None:
        for key in eval_info:
            if "success" in key.lower():
                success_key = key
                break
    if success_key is None:
        return None

    return_key = None
    for key in ("episode.return", "return", "episode_return"):
        if key in eval_info:
            return_key = key
            break
    if return_key is None:
        for key in eval_info:
            if "return" in key.lower():
                return_key = key
                break

    try:
        success = float(np.asarray(eval_info[success_key]))
        episode_return = (
            float(np.asarray(eval_info[return_key])) if return_key is not None else 0.0
        )
    except (TypeError, ValueError):
        return None

    return success, episode_return


def add_eval_video(eval_info, renders, fps=15):
    """Attach rendered eval videos to the eval log payload when available."""
    if len(renders) == 0:
        return eval_info
    renders = [render for render in renders if len(render) > 0]
    if len(renders) == 0:
        return eval_info

    eval_info["video"] = get_wandb_video(renders=renders, fps=fps)
    return eval_info

def set_agent_online_learning(agent, online_learning):
    if "online_learning" not in agent.config:
        return agent
    return agent.replace(config=agent.config.copy({"online_learning": online_learning}))


def save_checkpoints(agent, save_dir, epoch):
    save_agent(agent, save_dir, epoch)
    # save_critic disabled: the standalone critic pkl (~13MB/run) is never
    # consumed anywhere and the critic params are already inside the agent pkl.
    # (2.7GB reclaimed across 213 runs when this was turned off.)


def _resolve_agent_config_path(argv, default_path):
    for idx, arg in enumerate(argv[1:], start=1):
        if arg.startswith("--agent="):
            return arg.split("=", 1)[1]
        if arg == "--agent" and idx + 1 < len(argv):
            return argv[idx + 1]
    return default_path


def save_agent_source_snapshot(save_dir, default_agent_path):
    agent_path = os.path.abspath(
        _resolve_agent_config_path(sys.argv, default_agent_path)
    )
    snapshot_dir = os.path.join(save_dir, "source_snapshot", "agents")
    os.makedirs(snapshot_dir, exist_ok=True)

    if os.path.isfile(agent_path):
        shutil.copy2(agent_path, os.path.join(snapshot_dir, os.path.basename(agent_path)))

    with open(os.path.join(save_dir, "agent_source_path.txt"), "w") as f:
        f.write(agent_path + "\n")

def main(_):
    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name)
    run = setup_wandb(project='ant2', group=FLAGS.run_group, name=exp_name, entity="tjrcjf410-seoul-national-university")
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    config = FLAGS.agent
    config["discount"] = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)
    save_agent_source_snapshot(FLAGS.save_dir, "agents/acfql.py")
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = 0
    
    discount = FLAGS.discount

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env_name(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    best_eval_score = None

    def maybe_save_best_eval(agent, eval_info, step):
        nonlocal best_eval_score
        if not FLAGS.save_best_eval:
            return
        eval_score = get_eval_score(eval_info)
        if eval_score is None:
            return
        if best_eval_score is None or eval_score > best_eval_score:
            best_eval_score = eval_score
            print(f"New best eval at step {step}: {eval_score}", flush=True)
            save_checkpoints(agent, FLAGS.save_dir, "best")

    offline_init_time = time.time()
    agent = set_agent_online_learning(agent, False)
    # Offline RL
    scan_chunk = max(1, FLAGS.offline_scan_chunk)
    if scan_chunk > 1:
        for _name, _iv in (("eval_interval", FLAGS.eval_interval), ("log_interval", FLAGS.log_interval),
                           ("offline_steps", FLAGS.offline_steps)):
            if _iv and _iv % scan_chunk != 0:
                raise ValueError(f"offline_scan_chunk={scan_chunk} must divide {_name}={_iv}")
    _DROPPABLE = ('full_observations', 'terminals', 'next_actions')
    _prune = ((lambda b: {k: v for k, v in b.items() if k not in _DROPPABLE})
              if FLAGS.prune_batch_keys else (lambda b: b))
    import inspect as _inspect
    _scan_takes_full_update = 'full_update' in _inspect.signature(
        type(agent).batch_update.__wrapped__ if hasattr(type(agent).batch_update, '__wrapped__')
        else type(agent).batch_update).parameters
    offline_pbar = tqdm.tqdm(total=FLAGS.offline_steps, desc="offline")
    i = 0
    while i < FLAGS.offline_steps:
        n_sub = min(scan_chunk, FLAGS.offline_steps - i)
        i += n_sub
        log_step += n_sub
        offline_pbar.update(n_sub)

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        if n_sub == 1:
            batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)
            batch = _prune(batch)
            agent, offline_info = agent.update(batch)
        else:
            _bs = [_prune(train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount))
                   for _ in range(n_sub)]
            batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *_bs)
            # Some agents (e.g. rebrac) default batch_update to full_update=False,
            # which skips the actor and target updates.  Force the full update so the
            # scanned path is identical to the per-step path for every agent.
            if _scan_takes_full_update:
                agent, offline_info = agent.batch_update(batch, full_update=True)
            else:
                agent, offline_info = agent.batch_update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_checkpoints(agent, FLAGS.save_dir, log_step)

        # eval
        if (i == FLAGS.offline_steps if scan_chunk > 1 else i == FLAGS.offline_steps - 1) or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            eval_info = add_eval_video(eval_info, renders)
            logger.log(eval_info, "eval", step=log_step)
            maybe_save_best_eval(agent, eval_info, log_step)
            eval_postfix = get_eval_success_postfix(eval_info)
            if eval_postfix is not None:
                offline_pbar.set_postfix(eval_postfix, refresh=True)

    if FLAGS.offline_steps > 0:
        save_checkpoints(agent, FLAGS.save_dir, "offline")

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    agent = set_agent_online_learning(agent, True)
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    online_pbar = tqdm.tqdm(range(1, FLAGS.online_steps + 1), desc="online")
    for i in online_pbar:
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env_name(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            eval_info = add_eval_video(eval_info, renders)
            logger.log(eval_info, "eval", step=log_step)
            maybe_save_best_eval(agent, eval_info, log_step)
            eval_postfix = get_eval_success_postfix(eval_info)
            if eval_postfix is not None:
                online_pbar.set_postfix(eval_postfix, refresh=True)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_checkpoints(agent, FLAGS.save_dir, log_step)

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url or '')

if __name__ == '__main__':
    app.run(main)
