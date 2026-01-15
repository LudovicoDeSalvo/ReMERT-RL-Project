from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    moving_average,
    plot_convergence_comparison,
    plot_convergence_time,
    plot_early_training,
    plot_reward_distribution,
    plot_rewards,
)


def load_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def load_runs(prefix: str) -> list[np.ndarray]:
    run_paths = sorted(Path(".").glob(f"{prefix}_run*.npy"))
    return [np.load(path) for path in run_paths]


def plot_rewards_split(
    rewards_series: Sequence[Sequence[float]],
    labels: Sequence[str],
    save_path_base: str = "rewards_split",
    window: int = 10,
    std_series: Optional[Sequence[Sequence[float]]] = None,
) -> None:
    """Generate two separate plots: one for DQN, one for ReMERT, to improve readability."""
    if len(rewards_series) != len(labels):
        raise ValueError("Mismatch between rewards and labels")

    half = len(rewards_series) // 2
    std_list = list(std_series) if std_series is not None else None

    groups = [
        (rewards_series[:half], labels[:half], std_list[:half] if std_list is not None else None, "DQN"),
        (rewards_series[half:], labels[half:], std_list[half:] if std_list is not None else None, "DQN + ReMERT"),
    ]

    for idx, (group_rewards, group_labels, group_stds, title) in enumerate(groups):
        if not group_rewards:
            continue
        plt.figure(figsize=(10, 5))
        for i, (rewards, label) in enumerate(zip(group_rewards, group_labels)):
            rewards_arr = np.array(rewards, dtype=np.float32)
            smoothed = moving_average(rewards_arr, window)
            plt.plot(rewards_arr, alpha=0.3, label=f"{label} (raw)")
            plt.plot(smoothed, linewidth=2, label=f"{label} (avg {window})")
            if group_stds is not None and i < len(group_stds):
                std_arr = np.array(group_stds[i], dtype=np.float32)
                if std_arr.shape == rewards_arr.shape:
                    x = np.arange(len(rewards_arr))
                    plt.fill_between(
                        x,
                        rewards_arr - std_arr,
                        rewards_arr + std_arr,
                        alpha=0.15,
                        label=f"{label} (+-1sigma)",
                    )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"CartPole - {title}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{save_path_base}_{idx + 1}.png")
        plt.close()


def main() -> None:
    window = 10
    threshold = 475.0

    mean_dqn = load_required(Path("rewards_dqn_mean.npy"))
    mean_rmmert = load_required(Path("rewards_rmmert_mean.npy"))
    std_dqn = load_required(Path("rewards_dqn_std.npy"))
    std_rmmert = load_required(Path("rewards_rmmert_std.npy"))

    mean_labels = ["DQN (mean of runs)", "DQN + ReMERT (mean of runs)"]

    plot_rewards(
        rewards_series=[mean_dqn, mean_rmmert],
        labels=mean_labels,
        save_path="rewards.png",
        window=window,
        std_series=[std_dqn, std_rmmert],
    )

    plot_rewards_split(
        rewards_series=[mean_dqn, mean_rmmert],
        labels=mean_labels,
        save_path_base="rewards_split",
        window=window,
        std_series=[std_dqn, std_rmmert],
    )

    dqn_runs = load_runs("rewards_dqn")
    rmmert_runs = load_runs("rewards_rmmert")

    if dqn_runs and rmmert_runs:
        dqn_dist = np.concatenate(dqn_runs)
        rmmert_dist = np.concatenate(rmmert_runs)
        dqn_early = np.mean(np.stack(dqn_runs, axis=0), axis=0)
        rmmert_early = np.mean(np.stack(rmmert_runs, axis=0), axis=0)
        dist_labels = ["DQN (all runs)", "DQN + ReMERT (all runs)"]
        early_labels = mean_labels
    else:
        dqn_dist = mean_dqn
        rmmert_dist = mean_rmmert
        dqn_early = mean_dqn
        rmmert_early = mean_rmmert
        dist_labels = mean_labels
        early_labels = mean_labels

    plot_reward_distribution(
        rewards_series=[dqn_dist, rmmert_dist],
        labels=dist_labels,
        save_path="reward_distribution.png",
        tail=100,
    )

    plot_early_training(
        rewards_series=[dqn_early, rmmert_early],
        labels=early_labels,
        save_path="early_training_rewards.png",
        cutoff=200,
        window=window,
    )

    dqn_conv = plot_convergence_time(dqn_early, threshold=threshold, window=window)
    rmmert_conv = plot_convergence_time(rmmert_early, threshold=threshold, window=window)
    plot_convergence_comparison(
        convergence_steps=[dqn_conv, rmmert_conv],
        labels=["DQN", "DQN + ReMERT"],
        save_path="convergence_time.png",
        threshold=threshold,
        window=window,
    )
    print(
        "Grafici salvati: rewards.png, rewards_split_1.png, rewards_split_2.png, "
        "reward_distribution.png, early_training_rewards.png, convergence_time.png"
    )
    print(f"Convergenza stimata (episodi): DQN={dqn_conv}, DQN+ReMERT={rmmert_conv}")


if __name__ == "__main__":
    main()
