import random
from typing import Iterable, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    if window <= 1:
        return np.array(values, dtype=np.float32)
    values_array = np.array(values, dtype=np.float32)
    result: List[float] = []
    running = 0.0
    for idx, value in enumerate(values_array):
        running += value
        if idx + 1 > window:
            running -= values_array[idx - window]
            result.append(running / window)
        else:
            result.append(running / (idx + 1))
    return np.array(result, dtype=np.float32)


def plot_rewards(
    rewards_series: List[Sequence[float]],
    labels: Sequence[str],
    save_path: str = "rewards.png",
    window: int = 10,
    title: str = "CartPole - DQN vs ReMERT",
    std_series: Optional[List[Sequence[float]]] = None,
) -> None:
    plt.figure(figsize=(10, 5))
    for idx, (rewards, label) in enumerate(zip(rewards_series, labels)):
        rewards_arr = np.array(rewards, dtype=np.float32)
        smoothed = moving_average(rewards_arr, window)
        plt.plot(rewards_arr, alpha=0.3, label=f"{label} (raw)")
        plt.plot(smoothed, linewidth=2, label=f"{label} (avg {window})")
        if std_series is not None and idx < len(std_series):
            std_arr = np.array(std_series[idx], dtype=np.float32)
            if std_arr.shape == rewards_arr.shape:
                x = np.arange(len(rewards_arr))
                plt.fill_between(
                    x,
                    rewards_arr - std_arr,
                    rewards_arr + std_arr,
                    alpha=0.15,
                    label=f"{label} (±1σ)",
                )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_convergence_time(
    rewards: Sequence[float],
    threshold: float = 475.0,
    window: int = 10
) -> int:
    avg_rewards = moving_average(rewards, window)
    for idx in range(len(avg_rewards) - window):
        if all(avg_rewards[idx + i] >= threshold for i in range(window)):
            return idx + window
    return len(rewards)


def plot_convergence_comparison(
    convergence_steps: Sequence[int],
    labels: Sequence[str],
    save_path: str = "convergence_time.png",
    threshold: float = 475.0,
    window: int = 10
) -> None:
    plt.figure(figsize=(7, 4))
    x = np.arange(len(labels))
    plt.bar(x, convergence_steps, color="#4C78A8")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Episode")
    plt.title(f"Convergence episode (avg >= {threshold} for {window} eps)")
    for idx, value in enumerate(convergence_steps):
        plt.text(idx, value, str(value), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_reward_distribution(
    rewards_series: List[Sequence[float]],
    labels: Sequence[str],
    save_path: str = "reward_distribution.png",
    tail: int = 100
) -> None:
    plt.figure(figsize=(8, 5))
    for rewards, label in zip(rewards_series, labels):
        tail_rewards = rewards[-tail:]
        plt.hist(tail_rewards, bins=20, alpha=0.6, label=label)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of rewards (last {tail} episodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_early_training(
    rewards_series: List[Sequence[float]],
    labels: Sequence[str],
    save_path: str = "early_training_rewards.png",
    cutoff: int = 200,
    window: int = 10
) -> None:
    plt.figure(figsize=(10, 5))
    for rewards, label in zip(rewards_series, labels):
        rewards = rewards[:cutoff]
        smoothed = moving_average(rewards, window)
        plt.plot(smoothed, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Early Training Performance (first {cutoff} episodes)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

