"""
Experiment runner for comparative analysis.

Runs multiple configurations with multiple seeds to produce
statistically valid comparisons. This is critical for the
Results & Analysis rubric section (30 points).

Configurations tested:
  1. Random baseline — random agent + random tool every step
  2. Heuristic — fixed round-robin agent selection
  3. PPO only — PPO picks everything, no bandits
  4. Bandit only — random agent, bandit picks tools
  5. Full system — PPO + bandits together
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import ResearchEnv
from src.controller.ppo import PPOController
from src.bandits.contextual_bandit import MultiAgentBanditManager
from src.utils.config import MadisonConfig, PPOConfig, BanditConfig
from src.utils.data_structures import EpisodeMetrics
from loguru import logger


AGENT_TOOLS = {
    "search": ["web_scraper", "api_client", "academic_search"],
    "evaluator": ["credibility_scorer", "web_scraper"],
    "synthesis": ["relevance_extractor", "pdf_parser"],
    "deep_dive": ["web_scraper", "api_client", "relevance_extractor"],
}


def run_random_baseline(env: ResearchEnv, n_episodes: int) -> List[float]:
    """Baseline: random agent + random tool every step."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            done = term or trunc
        rewards.append(episode_reward)
    return rewards


def run_heuristic(env: ResearchEnv, n_episodes: int) -> List[float]:
    """Heuristic: cycle through agents in order, use first compatible tool."""
    # Pre-compute best actions for each agent (first compatible tool)
    agent_actions = [0, 7, 14, 18]  # search+scraper, eval+scorer, synth+extractor, deep+scraper
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        while not done:
            action = agent_actions[step % len(agent_actions)]
            obs, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            done = term or trunc
            step += 1
        rewards.append(episode_reward)
    return rewards


def run_ppo_only(
    env: ResearchEnv, n_episodes: int, config: PPOConfig
) -> List[float]:
    """PPO picks everything, no bandits."""
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    ppo = PPOController(obs_dim, n_actions, config)

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, log_prob, value = ppo.select_action(obs)
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ppo.store_transition(obs, action, reward, done, log_prob, value)
            episode_reward += reward
            obs = next_obs

        if len(ppo.buffer) >= config.batch_size:
            ppo.update()

        rewards.append(episode_reward)
    return rewards


def run_bandit_only(
    env: ResearchEnv, n_episodes: int, config: BanditConfig
) -> List[float]:
    """Random agent, bandit picks tools."""
    bandits = MultiAgentBanditManager(config, AGENT_TOOLS)
    agents = ["search", "evaluator", "synthesis", "deep_dive"]
    n_tools = env.n_tools

    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            # Random agent
            agent_idx = np.random.randint(len(agents))
            agent_name = agents[agent_idx]

            # Bandit picks tool
            context = obs[:config.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)

            # Map to action
            tool_names_list = [t.value for t in env.TOOLS]
            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = 0
            action = agent_idx * n_tools + tool_idx

            next_obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            bandits.update(agent_name, tool_arm, context, reward)
            episode_reward += reward
            obs = next_obs

        rewards.append(episode_reward)
    return rewards


def run_full_system(
    env: ResearchEnv, n_episodes: int, ppo_config: PPOConfig, bandit_config: BanditConfig
) -> List[float]:
    """Full system: PPO selects agent (4 actions), bandit selects tool (6 per agent).

    Separation is key: PPO owns WHO works, the bandit owns WHAT tool they use.
    Their optimisation problems are orthogonal, so they cooperate rather than
    compete. PPO's importance ratio is computed over the 4-agent space, which
    is what its network actually sampled — the stored action is agent_idx (0-3).
    """
    obs_dim = env.observation_space.shape[0]
    ppo = PPOController(obs_dim, env.n_agents, ppo_config)  # 4-action agent selector
    bandits = MultiAgentBanditManager(bandit_config, AGENT_TOOLS)
    tool_names_list = [t.value for t in env.TOOLS]

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            agent_idx, log_prob, value = ppo.select_action(obs)
            agent_name = env.AGENTS[agent_idx].value

            context = obs[:bandit_config.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)

            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = 0
            final_action = agent_idx * env.n_tools + tool_idx

            next_obs, reward, term, trunc, _ = env.step(final_action)
            done = term or trunc

            ppo.store_transition(obs, agent_idx, reward, done, log_prob, value)
            bandits.update(agent_name, tool_arm, context, reward)
            episode_reward += reward
            obs = next_obs

        if len(ppo.buffer) >= ppo_config.batch_size:
            ppo.update()

        rewards.append(episode_reward)
    return rewards


def smooth(data: List[float], window: int = 20) -> np.ndarray:
    """Apply rolling average smoothing."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def run_all_experiments(
    n_episodes: int = 500,
    n_seeds: int = 3,
    output_dir: str = "experiments/logs",
    env_config=None,
    reward_config=None,
) -> None:
    """
    Run all five configurations with multiple seeds.

    This produces the comparative analysis needed for the report.
    Pass env_config / reward_config to run with a non-default environment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = MadisonConfig()
    # Allow callers (e.g. CLI with --env) to override the env/reward configs
    if env_config is not None:
        config.environment = env_config
    if reward_config is not None:
        config.reward = reward_config

    all_results = {}

    experiments = {
        "Random Baseline": lambda env: run_random_baseline(env, n_episodes),
        "Heuristic": lambda env: run_heuristic(env, n_episodes),
        "PPO Only": lambda env: run_ppo_only(env, n_episodes, config.ppo),
        "Bandit Only": lambda env: run_bandit_only(env, n_episodes, config.bandit),
        "Full System (PPO+Bandit)": lambda env: run_full_system(
            env, n_episodes, config.ppo, config.bandit
        ),
    }

    for exp_name, run_fn in experiments.items():
        logger.info("Running experiment: {} ({} seeds)", exp_name, n_seeds)
        seed_results = []

        for seed in range(n_seeds):
            logger.info("  Seed {}/{}...", seed + 1, n_seeds)
            env = ResearchEnv(config.environment, config.reward)
            env.reset(seed=seed * 100)

            start = time.time()
            rewards = run_fn(env)
            elapsed = time.time() - start

            seed_results.append(rewards)
            logger.info(
                "  Seed {} done in {:.1f}s | Mean reward: {:.2f}",
                seed + 1, elapsed, np.mean(rewards[-50:]),
            )

        all_results[exp_name] = seed_results

    # Save raw results
    serializable = {
        name: [list(map(float, seed)) for seed in seeds]
        for name, seeds in all_results.items()
    }
    with open(output_path / "experiment_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Generate comparison plots
    _plot_comparison(all_results, output_path)
    _plot_final_performance(all_results, output_path)

    logger.info("All experiments complete! Results in {}", output_path)


def _plot_comparison(
    results: Dict[str, List[List[float]]], output_path: Path
) -> None:
    """Plot learning curves for all configurations."""
    colors = {
        "Random Baseline": "#888888",
        "Heuristic": "#E8593C",
        "PPO Only": "#534AB7",
        "Bandit Only": "#1D9E75",
        "Full System (PPO+Bandit)": "#3B8BD4",
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for name, seed_results in results.items():
        # Average across seeds
        min_len = min(len(s) for s in seed_results)
        trimmed = [s[:min_len] for s in seed_results]
        mean = np.mean(trimmed, axis=0)
        std = np.std(trimmed, axis=0)

        smoothed_mean = smooth(list(mean), window=20)
        x = range(19, 19 + len(smoothed_mean))

        color = colors.get(name, "#333333")
        ax.plot(x, smoothed_mean, label=name, color=color, linewidth=2)

        # Confidence band
        smoothed_upper = smooth(list(mean + std), window=20)
        smoothed_lower = smooth(list(mean - std), window=20)
        min_band_len = min(len(smoothed_mean), len(smoothed_upper), len(smoothed_lower))
        ax.fill_between(
            range(19, 19 + min_band_len),
            smoothed_lower[:min_band_len],
            smoothed_upper[:min_band_len],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title("Comparative Learning Curves — All Configurations", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "comparison_learning_curves.png", dpi=150)
    plt.close()
    logger.info("Comparison plot saved")


def _plot_final_performance(
    results: Dict[str, List[List[float]]], output_path: Path
) -> None:
    """Bar chart of final performance (last 50 episodes) with error bars."""
    names = []
    means = []
    stds = []

    for name, seed_results in results.items():
        final_rewards = []
        for seed in seed_results:
            final_rewards.append(np.mean(seed[-50:]))
        names.append(name)
        means.append(np.mean(final_rewards))
        stds.append(np.std(final_rewards))

    colors = ["#888888", "#E8593C", "#534AB7", "#1D9E75", "#3B8BD4"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[:len(names)], alpha=0.85)

    ax.set_ylabel("Mean Reward (Last 50 Episodes)", fontsize=12)
    ax.set_title("Final Performance Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.5,
            f"{mean:.1f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "final_performance.png", dpi=150)
    plt.close()
    logger.info("Final performance plot saved")


ENV_PRESETS = {
    "default": {
        "description": "Standard environment (default config)",
        "overrides": {},
    },
    "tight_budget": {
        "description": "Higher latency → budget exhausted faster; fewer steps allowed",
        "overrides": {
            "latency_mean": 1.0,   # double default (0.5) — each step costs more
            "latency_std": 0.3,    # more variance in cost
            "max_steps_per_episode": 12,  # 60% of default 20
        },
    },
    "complex_queries": {
        "description": "Higher relevance noise + smaller source pool → harder retrieval",
        "overrides": {
            "relevance_noise_std": 0.25,  # 2.5× default (0.1)
            "n_source_pool": 20,          # 40% of default 50
        },
    },
}


if __name__ == "__main__":
    from src.utils.logging import setup_logging

    parser = argparse.ArgumentParser(
        description="Run Madison RL comparative experiments."
    )
    parser.add_argument(
        "--env",
        choices=list(ENV_PRESETS.keys()),
        default="default",
        help="Environment variant to run (default: default)",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    setup_logging()

    preset = ENV_PRESETS[args.env]
    logger.info("Environment preset: {} — {}", args.env, preset["description"])

    # Build config and apply env overrides
    config = MadisonConfig()
    for key, val in preset["overrides"].items():
        setattr(config.environment, key, val)
        logger.info("  env.{} = {}", key, val)

    # Output to a subdirectory named after the preset
    if args.env == "default":
        output_dir = "experiments/logs"
    else:
        output_dir = f"experiments/logs/{args.env}"

    run_all_experiments(
        n_episodes=args.episodes,
        n_seeds=args.seeds,
        output_dir=output_dir,
        env_config=config.environment,
        reward_config=config.reward,
    )