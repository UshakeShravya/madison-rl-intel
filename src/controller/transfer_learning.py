"""
Transfer Learning module (RL Method #3).

Demonstrates that a PPO policy trained on one query domain
can quickly adapt to a new domain with fewer episodes.

This is the top-25% differentiator. The experiment:
  1. Train PPO on "source domain" (e.g. technical queries) for N episodes
  2. Save the trained weights
  3. Fine-tune on "target domain" (e.g. opinion queries) for M << N episodes
  4. Compare against training from scratch on target domain for M episodes
  5. Show the transfer-learned agent adapts faster

The key insight: the shared backbone has learned general research
strategies (when to use search vs synthesis, budget management)
that transfer across query types. Only the surface-level preferences
(which tools work best) need to adapt.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.research_env import ResearchEnv
from src.controller.ppo import PPOController
from src.bandits.contextual_bandit import MultiAgentBanditManager
from src.utils.config import MadisonConfig, PPOConfig, BanditConfig, EnvironmentConfig, RewardConfig
from src.utils.data_structures import QueryType
from loguru import logger


AGENT_TOOLS = {
    "search": ["web_scraper", "api_client", "academic_search"],
    "evaluator": ["credibility_scorer", "web_scraper"],
    "synthesis": ["relevance_extractor", "pdf_parser"],
    "deep_dive": ["web_scraper", "api_client", "relevance_extractor"],
}


class DomainSpecificEnv(ResearchEnv):
    """
    A ResearchEnv that only generates queries of a specific type.

    This lets us train on one domain and test transfer to another.
    """

    def __init__(
        self,
        query_type: QueryType,
        env_config: Optional[EnvironmentConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__(env_config, reward_config)
        self.fixed_query_type = query_type

    def reset(self, *, seed=None, options=None):
        """Override reset to force a specific query type."""
        obs, info = super().reset(seed=seed, options=options)
        # Override the query type
        self.current_query.query_type = self.fixed_query_type
        info["query_type"] = self.fixed_query_type.value
        return obs, info


def train_on_domain(
    query_type: QueryType,
    n_episodes: int,
    config: MadisonConfig,
    existing_ppo: Optional[PPOController] = None,
) -> Tuple[PPOController, List[float]]:
    """
    Train PPO on a specific query domain.

    Args:
        query_type: The domain to train on
        n_episodes: Number of training episodes
        config: Configuration
        existing_ppo: If provided, fine-tune this instead of training fresh

    Returns:
        Trained PPO controller and list of episode rewards
    """
    env = DomainSpecificEnv(query_type, config.environment, config.reward)
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    if existing_ppo is not None:
        ppo = existing_ppo
    else:
        ppo = PPOController(obs_dim, n_actions, config.ppo)

    bandits = MultiAgentBanditManager(config.bandit, AGENT_TOOLS)

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = ppo.select_action(obs)

            agent_idx = action // env.n_tools
            agent_name = env.AGENTS[agent_idx].value

            context = obs[:config.bandit.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)

            tool_names_list = [t.value for t in env.TOOLS]
            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = action % env.n_tools
            final_action = agent_idx * env.n_tools + tool_idx

            next_obs, reward, term, trunc, _ = env.step(final_action)
            done = term or trunc

            ppo.store_transition(obs, final_action, reward, done, log_prob, value)
            bandits.update(agent_name, tool_arm, context, reward)
            episode_reward += reward
            obs = next_obs

        if len(ppo.buffer) >= config.ppo.batch_size:
            ppo.update()

        rewards.append(episode_reward)

    return ppo, rewards


def run_transfer_experiment(
    source_domain: QueryType = QueryType.TECHNICAL,
    target_domain: QueryType = QueryType.OPINION,
    source_episodes: int = 300,
    target_episodes: int = 100,
    n_seeds: int = 3,
    output_dir: str = "experiments/logs",
) -> Dict:
    """
    Run the full transfer learning experiment.

    Compares three conditions:
      1. From scratch: train on target domain only
      2. Transfer: pre-train on source, fine-tune on target
      3. Source only: use source-trained policy directly on target (no fine-tuning)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = MadisonConfig()

    results = {
        "from_scratch": [],
        "transfer": [],
        "no_finetune": [],
    }

    for seed in range(n_seeds):
        logger.info("Transfer experiment — Seed {}/{}", seed + 1, n_seeds)
        np.random.seed(seed * 42)

        # ── Condition 1: Train from scratch on target domain ──
        logger.info("  Training from scratch on {} ({} episodes)...",
                     target_domain.value, target_episodes)
        _, scratch_rewards = train_on_domain(
            target_domain, target_episodes, config
        )
        results["from_scratch"].append(scratch_rewards)

        # ── Pre-train on source domain ──
        logger.info("  Pre-training on {} ({} episodes)...",
                     source_domain.value, source_episodes)
        source_ppo, source_rewards = train_on_domain(
            source_domain, source_episodes, config
        )

        # ── Condition 2: Transfer — fine-tune source model on target ──
        logger.info("  Fine-tuning on {} ({} episodes)...",
                     target_domain.value, target_episodes)
        # Deep copy the trained PPO so we don't modify the original
        import torch
        transfer_ppo = PPOController(
            obs_dim=396, n_actions=24, config=config.ppo
        )
        transfer_ppo.network.load_state_dict(
            copy.deepcopy(source_ppo.network.state_dict())
        )
        # Lower learning rate for fine-tuning
        for param_group in transfer_ppo.optimizer.param_groups:
            param_group['lr'] = config.ppo.learning_rate * 0.3

        _, transfer_rewards = train_on_domain(
            target_domain, target_episodes, config,
            existing_ppo=transfer_ppo,
        )
        results["transfer"].append(transfer_rewards)

        # ── Condition 3: No fine-tuning — use source model directly ──
        logger.info("  Evaluating source model on {} (no fine-tuning)...",
                     target_domain.value)
        env = DomainSpecificEnv(target_domain, config.environment, config.reward)
        bandits = MultiAgentBanditManager(config.bandit, AGENT_TOOLS)
        no_ft_rewards = []

        for _ in range(target_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action, _, _ = source_ppo.select_action(obs)
                agent_idx = action // env.n_tools
                agent_name = env.AGENTS[agent_idx].value
                context = obs[:config.bandit.context_dim]
                tool_arm, tool_name = bandits.select_tool(agent_name, context)
                tool_names_list = [t.value for t in env.TOOLS]
                if tool_name in tool_names_list:
                    tool_idx = tool_names_list.index(tool_name)
                else:
                    tool_idx = action % env.n_tools
                final_action = agent_idx * env.n_tools + tool_idx
                obs, reward, term, trunc, _ = env.step(final_action)
                done = term or trunc
                ep_reward += reward
            no_ft_rewards.append(ep_reward)

        results["no_finetune"].append(no_ft_rewards)

    # Save results
    serializable = {
        k: [list(map(float, seed)) for seed in v]
        for k, v in results.items()
    }
    serializable["source_domain"] = source_domain.value
    serializable["target_domain"] = target_domain.value
    serializable["source_episodes"] = source_episodes
    serializable["target_episodes"] = target_episodes

    with open(output_path / "transfer_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Plot
    _plot_transfer_results(results, source_domain, target_domain, output_path)

    logger.info("Transfer experiment complete!")
    return results


def _plot_transfer_results(
    results: Dict,
    source: QueryType,
    target: QueryType,
    output_path: Path,
) -> None:
    """Plot transfer learning comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "from_scratch": "#E8593C",
        "transfer": "#3B8BD4",
        "no_finetune": "#888888",
    }
    labels = {
        "from_scratch": f"From scratch ({target.value})",
        "transfer": f"Transfer ({source.value} → {target.value})",
        "no_finetune": f"Source model only (no fine-tuning)",
    }

    # Left plot: learning curves
    ax = axes[0]
    for key in ["from_scratch", "transfer", "no_finetune"]:
        seeds = results[key]
        min_len = min(len(s) for s in seeds)
        trimmed = [s[:min_len] for s in seeds]
        mean = np.mean(trimmed, axis=0)
        std = np.std(trimmed, axis=0)

        window = min(15, len(mean) // 3)
        if window > 1:
            smoothed = np.convolve(mean, np.ones(window) / window, mode="valid")
            x = range(window - 1, window - 1 + len(smoothed))
            ax.plot(x, smoothed, label=labels[key], color=colors[key], linewidth=2)

            sm_upper = np.convolve(mean + std, np.ones(window) / window, mode="valid")
            sm_lower = np.convolve(mean - std, np.ones(window) / window, mode="valid")
            band_len = min(len(smoothed), len(sm_upper), len(sm_lower))
            ax.fill_between(
                range(window - 1, window - 1 + band_len),
                sm_lower[:band_len], sm_upper[:band_len],
                alpha=0.15, color=colors[key],
            )

    ax.set_xlabel("Episode (on target domain)", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(f"Transfer Learning: {source.value} → {target.value}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right plot: bar chart of final performance
    ax = axes[1]
    names = []
    means = []
    stds = []
    bar_colors = []

    for key in ["from_scratch", "transfer", "no_finetune"]:
        seeds = results[key]
        final = [np.mean(s[-20:]) for s in seeds]
        names.append(labels[key].split("(")[0].strip())
        means.append(np.mean(final))
        stds.append(np.std(final))
        bar_colors.append(colors[key])

    bars = ax.bar(names, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Mean Reward (Last 20 Episodes)", fontsize=12)
    ax.set_title("Final Performance on Target Domain", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + 0.3,
            f"{m:.1f}", ha="center", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path / "transfer_learning.png", dpi=150)
    plt.close()
    logger.info("Transfer learning plot saved: {}", output_path / "transfer_learning.png")


if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()
    run_transfer_experiment(
        source_episodes=300,
        target_episodes=100,
        n_seeds=2,
    )