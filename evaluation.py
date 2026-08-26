from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from functools import partial


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
            
        observation_history = []
        action_history = []
        
        done = False
        step = 0
        render = []
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        while not done:
            
            action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            # print(info)
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                # print(info["gripper_contact"])
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders

def evaluate_vectorized(
    agent,
    envs,
    num_eval_episodes=50,
    eval_gaussian=None,
    action_dim=None,
):
    """Parallel-episode evaluation over a list of identical envs.

    Statistically identical to ``evaluate`` for the table metrics: the same
    number of independent episodes, the same policy and chunk-execution
    semantics, stats taken from each episode's terminal ``info`` (where
    ``success`` lives).  Gripper-contact diagnostics are not collected here.
    MuJoCo releases the GIL inside ``mj_step``, so a thread pool genuinely
    parallelises the physics stepping across envs.
    """
    from concurrent.futures import ThreadPoolExecutor

    actor_fn = supply_rng(
        agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))
    )
    stats = defaultdict(list)
    pool = ThreadPoolExecutor(max_workers=len(envs))
    episodes_left = num_eval_episodes
    while episodes_left > 0:
        n = min(len(envs), episodes_left)
        resets = list(pool.map(lambda e: e.reset(), envs[:n]))
        obs = [r[0] for r in resets]
        queues = [[] for _ in range(n)]
        done = [False] * n
        while not all(done):
            idx = [i for i in range(n) if not done[i]]
            if len(queues[idx[0]]) == 0:
                batch = np.stack([obs[i] for i in idx])
                acts = np.asarray(actor_fn(observations=batch))
                acts = acts.reshape(len(idx), -1, action_dim)
                for j, i in enumerate(idx):
                    queues[i] = list(acts[j])
            acts_now = {}
            for i in idx:
                a = queues[i].pop(0)
                if eval_gaussian is not None:
                    a = np.random.normal(a, eval_gaussian)
                acts_now[i] = np.clip(a, -1, 1)
            results = list(pool.map(lambda i: envs[i].step(acts_now[i]), idx))
            for j, i in enumerate(idx):
                o2, r, term, trunc, info = results[j]
                obs[i] = o2
                if term or trunc:
                    done[i] = True
                    add_to(stats, flatten(info))
        episodes_left -= n
    pool.shutdown()
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, [], []
