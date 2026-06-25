import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance
import glob

class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def _config_get(config, key, default=None):
    """Read from ConfigDict/dict-like configs without assuming a concrete type."""
    if config is None:
        return default
    try:
        return config[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(config, key, default)


def get_agent_setting_name():
    """Return the paper-style method name implied by the current agent flags."""
    try:
        agent_config = getattr(flags.FLAGS, 'agent', None)
        horizon_length = getattr(flags.FLAGS, 'horizon_length', None)
    except Exception:
        return None

    agent_name = _config_get(agent_config, 'agent_name')
    action_chunking = _config_get(agent_config, 'action_chunking', True)
    actor_type = _config_get(agent_config, 'actor_type')
    if horizon_length is None:
        horizon_length = _config_get(agent_config, 'horizon_length', 1)
    horizon_length = 1 if horizon_length is None else int(horizon_length)

    if agent_name == 'acfql':
        if actor_type == 'best-of-n':
            if action_chunking and horizon_length > 1:
                return 'qc'
            if horizon_length > 1:
                return 'bfn-n'
            return 'bfn'
        if actor_type == 'distill-ddpg':
            if action_chunking and horizon_length > 1:
                return 'qc-fql'
            if horizon_length > 1:
                return 'fql-n'
            return 'fql'
        if actor_type is not None:
            return f'{agent_name}-{actor_type}'
        return agent_name

    if agent_name == 'acrlpd':
        bc_alpha = float(_config_get(agent_config, 'bc_alpha', 0.0))
        if bc_alpha > 0 and action_chunking and horizon_length > 1:
            return 'qc-rlpd'
        if action_chunking and horizon_length > 1:
            return 'rlpd-ac'
        return 'rlpd'

    if agent_name == 'mfp':
        if not action_chunking and horizon_length > 1:
            return 'mfp-n'
        return 'mfp'

    if agent_name == 'svgd':
        score_gain = _config_get(agent_config, 'score_gain', 0)
        eps = _config_get(agent_config, 'epsilon', 1e-5)
        bdw = _config_get(agent_config, 'bandwidth', 0.05)
        return f'svgd_sc={score_gain}_eps={eps}_bdw={bdw}'

    return agent_name


def get_action_chunking_setting_name():
    """Return an explicit action-chunking/horizon suffix for experiment names."""
    try:
        agent_config = getattr(flags.FLAGS, 'agent', None)
        horizon_length = getattr(flags.FLAGS, 'horizon_length', None)
    except Exception:
        return None

    action_chunking = bool(_config_get(agent_config, 'action_chunking', True))
    if horizon_length is None:
        horizon_length = _config_get(agent_config, 'horizon_length', 1)
    horizon_length = 1 if horizon_length is None else int(horizon_length)

    chunk_name = 'chunk' if action_chunking else 'nstep'
    return f'{chunk_name}-h{horizon_length}'


def get_bc_bandwidth_setting_name():
    """Return the behavior-cloning drift bandwidth suffix when configured."""
    try:
        agent_config = getattr(flags.FLAGS, 'agent', None)
    except Exception:
        return None

    bc_bandwidth = _config_get(agent_config, 'bc_drift_bandwidth')
    if bc_bandwidth is None:
        bc_bandwidth = _config_get(agent_config, 'bc_bandwidth')
    if bc_bandwidth is None:
        return None

    try:
        bc_bandwidth = f'{float(bc_bandwidth):g}'
    except (TypeError, ValueError):
        bc_bandwidth = str(bc_bandwidth)
    return f'bc-bw{bc_bandwidth}'


def get_exp_name(seed, env_name=None):
    """Return the experiment name."""
    exp_name = ''
    if env_name:
        exp_name += f'{env_name}_'
    exp_name += f'sd{seed:03d}'
    agent_setting = get_agent_setting_name()
    if agent_setting is not None:
        exp_name += f'_{agent_setting}'
    chunk_setting = get_action_chunking_setting_name()
    if chunk_setting is not None:
        exp_name += f'_{chunk_setting}'
    bc_bandwidth_setting = get_bc_bandwidth_setting_name()
    if bc_bandwidth_setting is not None:
        exp_name += f'_{bc_bandwidth_setting}'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f'_s{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    if 'SLURM_ARRAY_JOB_ID' in os.environ:
        exp_name += f'{os.environ["SLURM_ARRAY_JOB_ID"]}.'
    if 'SLURM_ARRAY_TASK_ID' in os.environ:
        exp_name += f'{os.environ["SLURM_ARRAY_TASK_ID"]}.'
    exp_name += f'_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
    )

    run = wandb.init(**init_kwargs)

    # assume a flat structure
    run.save('*.py')
    run.save('**/*.py')

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
