"""
Main training loop for Madison RL Intelligence Agent.

Ties together:
  - ResearchEnv (simulation)
  - PPOController (RL Method 1 — strategic decisions)
  - MultiAgentBanditManager (RL Method 2 — tactical decisions)
  - RewardEngine (multi-component reward signal)

Produces learning curves, checkpoints, and metrics for analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from src.environment import ResearchEnv
from src.controller.ppo import PPOController
from src.bandits.contextual_bandit import MultiAgentBanditManager
from src.utils.config import MadisonConfig
from src.utils.data_structures import EpisodeMetrics
from src.utils.logging import setup_logging, log_metric
from loguru import logger


class Trainer:
    """Orchestrates the full training process."""

    def __init__(self, config: MadisonConfig) -> None:
        self.config = config

        # Environment
        self.env = ResearchEnv(config.environment, config.reward)

        # PPO controller (RL Method 1) — selects WHICH AGENT to activate (4 choices).
        # The bandit independently selects WHICH TOOL to use (up to 6 per agent).
        # Keeping the two decisions separate lets each optimizer focus on its own
        # problem without their gradients interfering with each other.
        obs_dim = self.env.observation_space.shape[0]
        self.ppo = PPOController(obs_dim, self.env.n_agents, config.ppo)

        # Bandit manager (RL Method 2)
        agent_tools = {
            "search": ["web_scraper", "api_client", "academic_search"],
            "evaluator": ["credibility_scorer", "web_scraper"],
            "synthesis": ["relevance_extractor", "pdf_parser"],
            "deep_dive": ["web_scraper", "api_client", "relevance_extractor"],
        }
        self.bandits = MultiAgentBanditManager(config.bandit, agent_tools)

        # Tracking
        self.episode_rewards: List[float] = []
        self.eval_rewards: List[float] = []
        self.eval_episodes_x: List[int] = []
        self.metrics_history: List[Dict] = []

        # Directories
        self.log_dir = Path(config.training.log_dir)
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        """Run the full training loop."""
        logger.info(
            "Starting training: {} episodes", self.config.training.n_episodes
        )
        start_time = time.time()

        for episode in range(1, self.config.training.n_episodes + 1):
            metrics = self._run_episode(training=True)
            self.episode_rewards.append(metrics.total_reward)

            # Log progress every 10 episodes
            if episode % 10 == 0:
                recent_avg = np.mean(self.episode_rewards[-10:])
                logger.info(
                    "Episode {}/{} | Avg Reward: {:.3f} | Coverage: {:.3f} | Sources: {}",
                    episode,
                    self.config.training.n_episodes,
                    recent_avg,
                    metrics.coverage_score,
                    metrics.n_sources,
                )

            # PPO update when buffer is full enough
            if len(self.ppo.buffer) >= self.config.ppo.batch_size:
                update_metrics = self.ppo.update()
                if episode % 50 == 0:
                    logger.info(
                        "PPO Update | Policy Loss: {:.4f} | Value Loss: {:.4f} | Entropy: {:.4f}",
                        update_metrics["policy_loss"],
                        update_metrics["value_loss"],
                        update_metrics["entropy"],
                    )

            # Evaluation
            if episode % self.config.training.eval_interval == 0:
                eval_reward = self._evaluate()
                self.eval_rewards.append(eval_reward)
                self.eval_episodes_x.append(episode)
                logger.info(
                    ">>> EVAL at episode {} | Mean Reward: {:.3f}",
                    episode, eval_reward,
                )

            # Checkpoint
            if episode % self.config.training.checkpoint_interval == 0:
                self._save_checkpoint(episode)

            # Store metrics
            self.metrics_history.append({
                "episode": episode,
                "reward": metrics.total_reward,
                "coverage": metrics.coverage_score,
                "diversity": metrics.diversity_score,
                "n_sources": metrics.n_sources,
                "budget_used": metrics.budget_used,
            })

        elapsed = time.time() - start_time
        logger.info("Training complete in {:.1f} minutes", elapsed / 60)

        # Save final results
        self._save_results()
        self._plot_learning_curves()

    def _run_episode(self, training: bool = True) -> EpisodeMetrics:
        """Run a single episode."""
        obs, info = self.env.reset()
        episode_reward = 0.0
        agent_counts = {}
        tool_counts = {}

        done = False
        while not done:
            # PPO selects WHICH AGENT to activate (action ∈ {0,1,2,3})
            agent_idx, log_prob, value = self.ppo.select_action(obs)
            agent_name = self.env.AGENTS[agent_idx].value

            # Bandit selects WHICH TOOL for that agent (separate 6-action space)
            bandit_context = obs[:self.config.bandit.context_dim]
            tool_arm, tool_name = self.bandits.select_tool(
                agent_name, bandit_context
            )

            # Compose the full environment action from the two cooperative decisions
            tool_names_list = [t.value for t in self.env.TOOLS]
            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = 0
            final_action = agent_idx * self.env.n_tools + tool_idx

            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(
                final_action
            )
            done = terminated or truncated

            # Store agent_idx (0-3) — NOT final_action (0-23).
            # PPO's importance ratio is over the 4-agent space, so the stored
            # action must match what the network actually sampled.
            if training:
                self.ppo.store_transition(
                    obs, agent_idx, reward, done, log_prob, value
                )

                # Update bandit with reward
                self.bandits.update(
                    agent_name, tool_arm, bandit_context, reward
                )

            episode_reward += reward
            obs = next_obs

            # Track agent/tool usage
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return EpisodeMetrics(
            total_reward=episode_reward,
            coverage_score=self.env.memory.get_coverage_score(),
            diversity_score=self.env.memory.get_source_diversity(),
            n_sources=len(self.env.memory.sources),
            n_findings=len(self.env.memory.findings),
            n_steps=self.env.memory.step_count,
            budget_used=self.env.memory.budget_used,
            agent_activations=agent_counts,
            tool_selections=tool_counts,
        )

    def _evaluate(self) -> float:
        """Run evaluation episodes (no training, no exploration)."""
        rewards = []
        for _ in range(self.config.training.eval_episodes):
            metrics = self._run_episode(training=False)
            rewards.append(metrics.total_reward)
        return float(np.mean(rewards))

    def _save_checkpoint(self, episode: int) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / f"ppo_episode_{episode}.pt"
        self.ppo.save(str(path))
        logger.info("Checkpoint saved: {}", path)

    def _save_results(self) -> None:
        """Save training metrics to JSON."""
        results = {
            "episode_rewards": self.episode_rewards,
            "eval_rewards": self.eval_rewards,
            "eval_episodes": self.eval_episodes_x,
            "bandit_stats": self.bandits.get_all_stats(),
            "metrics_history": self.metrics_history,
        }
        path = self.log_dir / "training_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved: {}", path)

    def _plot_learning_curves(self) -> None:
        """Generate and save learning curve plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Madison RL Intelligence Agent — Training Results", fontsize=14)

        # 1. Episode rewards with smoothing
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color="blue", label="Raw")
        # Smoothed curve (rolling average)
        if len(self.episode_rewards) > 50:
            smoothed = np.convolve(
                self.episode_rewards, np.ones(50) / 50, mode="valid"
            )
            ax.plot(range(49, 49 + len(smoothed)), smoothed, color="blue", linewidth=2, label="Smoothed (50)")
        if self.eval_rewards:
            ax.plot(self.eval_episodes_x, self.eval_rewards, "ro-", label="Eval", markersize=4)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Learning Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Coverage over time
        ax = axes[0, 1]
        coverages = [m["coverage"] for m in self.metrics_history]
        ax.plot(coverages, alpha=0.3, color="green")
        if len(coverages) > 50:
            smoothed = np.convolve(coverages, np.ones(50) / 50, mode="valid")
            ax.plot(range(49, 49 + len(smoothed)), smoothed, color="green", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Coverage Score")
        ax.set_title("Topic Coverage Over Training")
        ax.grid(True, alpha=0.3)

        # 3. Source diversity over time
        ax = axes[1, 0]
        diversities = [m["diversity"] for m in self.metrics_history]
        ax.plot(diversities, alpha=0.3, color="orange")
        if len(diversities) > 50:
            smoothed = np.convolve(diversities, np.ones(50) / 50, mode="valid")
            ax.plot(range(49, 49 + len(smoothed)), smoothed, color="orange", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Diversity Score")
        ax.set_title("Source Diversity Over Training")
        ax.grid(True, alpha=0.3)

        # 4. Budget efficiency over time
        ax = axes[1, 1]
        efficiencies = [
            m["n_sources"] / max(m["budget_used"], 0.1)
            for m in self.metrics_history
        ]
        ax.plot(efficiencies, alpha=0.3, color="purple")
        if len(efficiencies) > 50:
            smoothed = np.convolve(efficiencies, np.ones(50) / 50, mode="valid")
            ax.plot(range(49, 49 + len(smoothed)), smoothed, color="purple", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sources per Budget Unit")
        ax.set_title("Budget Efficiency Over Training")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.log_dir / "learning_curves.png"
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info("Learning curves saved: {}", path)


def main():
    """Entry point for training."""
    setup_logging()

    # Load config (use defaults for now)
    config = MadisonConfig()
    logger.info("Config loaded: {} episodes, PPO lr={}, Bandit alpha={}",
                config.training.n_episodes, config.ppo.learning_rate, config.bandit.alpha)

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()