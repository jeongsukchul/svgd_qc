#!/usr/bin/env python3
"""Plot decoded OGBench cube trajectories split by dataset terminals."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


XYZ_CENTER = np.array([0.425, 0.0, 0.0], dtype=np.float32)
XYZ_SCALER = 10.0
DEFAULT_DATASETS = (
    "cube-double-play-singletask-v0",
    "cube-triple-play-singletask-task2-v0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode and plot OGBench cube state trajectories. Episodes are split "
            "with terminals, never masks."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="OGBench dataset names or local .npz paths.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("~/.ogbench/data").expanduser(),
        help="Directory containing OGBench .npz files when loading directly.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0],
        help="Episode ids to plot for each dataset.",
    )
    parser.add_argument(
        "--summary-episodes",
        type=int,
        default=5,
        help="Number of episodes per dataset to write into the summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/trajectory_plots"),
        help="Directory for PNG plots and summary CSV.",
    )
    parser.add_argument(
        "--use-ogbench",
        action="store_true",
        help="Try ogbench.make_env_and_datasets first, then fall back to .npz loading.",
    )
    parser.add_argument(
        "--no-final-next",
        action="store_true",
        help="Do not append the final next_observation to plotted state sequences.",
    )
    return parser.parse_args()


def play_dataset_name(name_or_path: str) -> str:
    """Return the underlying play dataset name for a single-task cube dataset."""
    name = Path(name_or_path).name
    if name.endswith(".npz"):
        name = name[:-4]
    if name.endswith("-val"):
        name = name[:-4]

    splits = name.split("-")
    if "singletask" not in splits:
        return name

    pos = splits.index("singletask")
    return "-".join(splits[:pos] + splits[-1:])


def load_raw_npz_as_transitions(name_or_path: str, data_dir: Path) -> dict[str, np.ndarray]:
    """Load an OGBench raw .npz and convert it to transition format."""
    path = Path(name_or_path).expanduser()
    if not path.exists():
        path = data_dir / f"{play_dataset_name(name_or_path)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    raw = np.load(path)
    required = {"observations", "actions", "terminals"}
    missing = sorted(required.difference(raw.files))
    if missing:
        raise KeyError(f"{path} is missing required keys: {missing}")

    terminals = raw["terminals"].astype(bool)
    ob_mask = ~terminals
    next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
    new_terminals = np.concatenate([terminals[1:], [True]])

    dataset = {
        "observations": raw["observations"][ob_mask],
        "actions": raw["actions"][ob_mask],
        "next_observations": raw["observations"][next_ob_mask],
        "terminals": new_terminals[ob_mask].astype(bool),
    }
    for key in ("rewards", "masks"):
        if key in raw.files:
            dataset[key] = raw[key][ob_mask]

    return dataset


def load_with_ogbench(dataset_name: str) -> dict[str, np.ndarray] | None:
    try:
        import ogbench
    except ImportError:
        return None

    _, train_dataset, _ = ogbench.make_env_and_datasets(
        dataset_name,
        compact_dataset=False,
    )
    return train_dataset


def load_dataset(
    dataset_name: str,
    data_dir: Path,
    use_ogbench: bool,
) -> tuple[str, dict[str, np.ndarray], str]:
    if use_ogbench and not dataset_name.endswith(".npz"):
        dataset = load_with_ogbench(dataset_name)
        if dataset is not None:
            return dataset_name, dataset, "ogbench"

    return play_dataset_name(dataset_name), load_raw_npz_as_transitions(dataset_name, data_dir), "npz"


def infer_num_blocks(dataset_name: str, obs_dim: int) -> int:
    if "cube-single" in dataset_name:
        return 1
    if "cube-double" in dataset_name:
        return 2
    if "cube-triple" in dataset_name:
        return 3
    if "cube-quadruple" in dataset_name:
        return 4

    maybe = (obs_dim - 19) // 9
    if obs_dim >= 28 and 19 + 9 * maybe == obs_dim:
        return maybe
    raise ValueError(f"Cannot infer number of cube blocks from obs dim {obs_dim}.")


def decode_cube_obs(
    observations: np.ndarray,
    num_blocks: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    obs = np.asarray(observations)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.shape[-1] < 19 + 9 * num_blocks:
        raise ValueError(
            f"Observation dim {obs.shape[-1]} is too small for {num_blocks} blocks."
        )

    eff_pos = obs[:, 12:15] / XYZ_SCALER + XYZ_CENTER
    block_positions = []
    for block_id in range(num_blocks):
        base = 19 + 9 * block_id
        block_positions.append(obs[:, base : base + 3] / XYZ_SCALER + XYZ_CENTER)
    return eff_pos, block_positions


def split_episodes(dataset: dict[str, np.ndarray]) -> list[dict[str, np.ndarray]]:
    terminals = np.asarray(dataset["terminals"]).astype(bool)
    terminal_idxs = np.where(terminals)[0]

    episodes = []
    start = 0
    for end_idx in terminal_idxs:
        end = end_idx + 1
        episodes.append({key: value[start:end] for key, value in dataset.items()})
        start = end

    if start < len(terminals):
        episodes.append({key: value[start:] for key, value in dataset.items()})

    return episodes


def episode_state_sequence(
    episode: dict[str, np.ndarray],
    include_final_next: bool,
) -> np.ndarray:
    obs = episode["observations"]
    if include_final_next and "next_observations" in episode:
        return np.concatenate([obs, episode["next_observations"][-1:]], axis=0)
    return obs


def displacement(path: np.ndarray) -> float:
    return float(np.linalg.norm(path[-1] - path[0]))


def valid_episode_ids(
    episodes: list[dict[str, np.ndarray]],
    episode_ids: list[int],
    dataset_label: str,
) -> list[int]:
    valid_ids = [episode_id for episode_id in episode_ids if 0 <= episode_id < len(episodes)]
    if not valid_ids:
        raise ValueError(f"No requested episode ids exist for {dataset_label}.")
    return valid_ids


def summarize_episode(
    dataset_label: str,
    episode_id: int,
    episode: dict[str, np.ndarray],
    num_blocks: int,
    include_final_next: bool,
) -> dict[str, object]:
    obs_seq = episode_state_sequence(episode, include_final_next)
    eff_pos, block_positions = decode_cube_obs(obs_seq, num_blocks)

    row: dict[str, object] = {
        "dataset": dataset_label,
        "episode": episode_id,
        "length_transitions": len(episode["actions"]),
        "length_states": len(obs_seq),
        "eef_displacement": displacement(eff_pos),
    }
    if "rewards" in episode:
        row["reward_sum"] = float(np.asarray(episode["rewards"]).sum())

    for block_id, block_pos in enumerate(block_positions):
        row[f"block{block_id}_start_x"] = float(block_pos[0, 0])
        row[f"block{block_id}_start_y"] = float(block_pos[0, 1])
        row[f"block{block_id}_start_z"] = float(block_pos[0, 2])
        row[f"block{block_id}_end_x"] = float(block_pos[-1, 0])
        row[f"block{block_id}_end_y"] = float(block_pos[-1, 1])
        row[f"block{block_id}_end_z"] = float(block_pos[-1, 2])
        row[f"block{block_id}_displacement"] = displacement(block_pos)
    return row


def plot_episodes(
    dataset_label: str,
    episodes: list[dict[str, np.ndarray]],
    episode_ids: list[int],
    num_blocks: int,
    include_final_next: bool,
    output_dir: Path,
) -> Path:
    valid_ids = valid_episode_ids(episodes, episode_ids, dataset_label)

    ncols = min(3, len(valid_ids))
    nrows = math.ceil(len(valid_ids) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 5.2 * nrows),
        squeeze=False,
    )

    block_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for axis in axes.ravel():
        axis.set_visible(False)

    for axis, episode_id in zip(axes.ravel(), valid_ids):
        axis.set_visible(True)
        episode = episodes[episode_id]
        obs_seq = episode_state_sequence(episode, include_final_next)
        eff_pos, block_positions = decode_cube_obs(obs_seq, num_blocks)

        axis.plot(eff_pos[:, 0], eff_pos[:, 1], color="black", alpha=0.6, label="end-effector")
        axis.scatter(eff_pos[0, 0], eff_pos[0, 1], color="black", marker="o", s=24)
        axis.scatter(eff_pos[-1, 0], eff_pos[-1, 1], color="black", marker="x", s=36)

        for block_id, block_pos in enumerate(block_positions):
            color = block_colors[block_id % len(block_colors)]
            axis.plot(
                block_pos[:, 0],
                block_pos[:, 1],
                color=color,
                linewidth=2,
                label=f"block {block_id}",
            )
            axis.scatter(block_pos[0, 0], block_pos[0, 1], color=color, marker="o", s=32)
            axis.scatter(block_pos[-1, 0], block_pos[-1, 1], color=color, marker="x", s=48)

        axis.set_title(f"episode {episode_id} ({len(episode['actions'])} transitions)")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.axis("equal")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)

    fig.suptitle(f"{dataset_label} decoded XY trajectories", fontsize=14)
    fig.tight_layout()

    suffix = "_".join(str(episode_id) for episode_id in valid_ids)
    output_path = output_dir / f"{dataset_label}_episodes_{suffix}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def set_3d_limits(axis, paths: list[np.ndarray]) -> None:
    points = np.concatenate(paths, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = 0.5 * (mins + maxs)
    ranges = np.maximum(maxs - mins, np.array([0.12, 0.12, 0.08], dtype=np.float32))

    for setter, center, radius in zip(
        (axis.set_xlim, axis.set_ylim, axis.set_zlim),
        centers,
        0.55 * ranges,
    ):
        setter(float(center - radius), float(center + radius))
    axis.set_box_aspect(tuple(float(value) for value in ranges))


def plot_episodes_3d(
    dataset_label: str,
    episodes: list[dict[str, np.ndarray]],
    episode_ids: list[int],
    num_blocks: int,
    include_final_next: bool,
    output_dir: Path,
) -> Path:
    valid_ids = valid_episode_ids(episodes, episode_ids, dataset_label)

    ncols = min(3, len(valid_ids))
    nrows = math.ceil(len(valid_ids) / ncols)
    fig = plt.figure(figsize=(6.2 * ncols, 5.4 * nrows))
    axes = [
        fig.add_subplot(nrows, ncols, index + 1, projection="3d")
        for index in range(nrows * ncols)
    ]

    block_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for axis in axes:
        axis.set_visible(False)

    for axis, episode_id in zip(axes, valid_ids):
        axis.set_visible(True)
        episode = episodes[episode_id]
        obs_seq = episode_state_sequence(episode, include_final_next)
        eff_pos, block_positions = decode_cube_obs(obs_seq, num_blocks)

        axis.plot(
            eff_pos[:, 0],
            eff_pos[:, 1],
            eff_pos[:, 2],
            color="black",
            alpha=0.6,
            label="end-effector",
        )
        axis.scatter(eff_pos[0, 0], eff_pos[0, 1], eff_pos[0, 2], color="black", marker="o", s=24)
        axis.scatter(eff_pos[-1, 0], eff_pos[-1, 1], eff_pos[-1, 2], color="black", marker="x", s=42)

        for block_id, block_pos in enumerate(block_positions):
            color = block_colors[block_id % len(block_colors)]
            axis.plot(
                block_pos[:, 0],
                block_pos[:, 1],
                block_pos[:, 2],
                color=color,
                linewidth=2,
                label=f"block {block_id}",
            )
            axis.scatter(block_pos[0, 0], block_pos[0, 1], block_pos[0, 2], color=color, marker="o", s=32)
            axis.scatter(block_pos[-1, 0], block_pos[-1, 1], block_pos[-1, 2], color=color, marker="x", s=52)

        set_3d_limits(axis, [eff_pos, *block_positions])
        axis.view_init(elev=24, azim=-58)
        axis.set_title(f"episode {episode_id} ({len(episode['actions'])} transitions)")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)

    fig.suptitle(f"{dataset_label} decoded XYZ trajectories", fontsize=14)
    fig.tight_layout()

    suffix = "_".join(str(episode_id) for episode_id in valid_ids)
    output_path = output_dir / f"{dataset_label}_episodes_{suffix}_3d.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_summary(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for requested_name in args.datasets:
        dataset_label, dataset, source = load_dataset(
            requested_name,
            data_dir=args.data_dir,
            use_ogbench=args.use_ogbench,
        )
        episodes = split_episodes(dataset)
        num_blocks = infer_num_blocks(dataset_label, dataset["observations"].shape[-1])

        print(f"\n{requested_name}")
        print(f"  source: {source}")
        print(f"  trajectory dataset: {dataset_label}")
        print(f"  observations: {dataset['observations'].shape}")
        print(f"  actions: {dataset['actions'].shape}")
        print(f"  num episodes: {len(episodes)}")
        print(f"  first episode length: {len(episodes[0]['actions'])}")
        print(f"  terminal count: {int(np.asarray(dataset['terminals']).astype(bool).sum())}")

        plot_path = plot_episodes(
            dataset_label=dataset_label,
            episodes=episodes,
            episode_ids=args.episodes,
            num_blocks=num_blocks,
            include_final_next=not args.no_final_next,
            output_dir=args.output_dir,
        )
        plot_3d_path = plot_episodes_3d(
            dataset_label=dataset_label,
            episodes=episodes,
            episode_ids=args.episodes,
            num_blocks=num_blocks,
            include_final_next=not args.no_final_next,
            output_dir=args.output_dir,
        )
        print(f"  xy plot: {plot_path}")
        print(f"  3d plot: {plot_3d_path}")

        for episode_id in range(min(args.summary_episodes, len(episodes))):
            summary_rows.append(
                summarize_episode(
                    dataset_label=dataset_label,
                    episode_id=episode_id,
                    episode=episodes[episode_id],
                    num_blocks=num_blocks,
                    include_final_next=not args.no_final_next,
                )
            )

    summary_path = args.output_dir / "trajectory_summary.csv"
    write_summary(summary_rows, summary_path)
    print(f"\nsummary: {summary_path}")


if __name__ == "__main__":
    main()
