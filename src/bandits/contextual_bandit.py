"""
Contextual Bandit (LinUCB) for tool selection.

This is RL Method #2. Each agent has its own bandit that learns
which tool works best given the current context (query type,
agent state, time remaining, etc.).

LinUCB maintains a confidence bound for each arm (tool).
It picks the arm with the highest upper confidence bound,
naturally balancing exploration vs exploitation.

Reference: Li et al., "A Contextual-Bandit Approach to
Personalized News Article Recommendation" (2010)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import BanditConfig


class LinUCBArm:
    """
    A single arm in the LinUCB bandit.

    Each arm maintains:
      - A: design matrix (d x d) — tracks how much we know
      - b: reward vector (d,) — tracks what we've learned
      - theta: weight vector = A^{-1} b — our best guess at reward

    The UCB for this arm given context x is:
      UCB = theta^T x + alpha * sqrt(x^T A^{-1} x)
              ^predicted     ^exploration bonus
              reward         (high when uncertain)
    """

    def __init__(self, context_dim: int, regularization: float = 1.0) -> None:
        self.d = context_dim
        self.A = np.eye(self.d) * regularization  # Regularized design matrix
        self.b = np.zeros(self.d)                  # Reward vector
        self.A_inv = np.eye(self.d) / regularization  # Cache inverse

    def get_ucb(self, context: np.ndarray, alpha: float) -> float:
        """
        Compute upper confidence bound for this arm.

        Args:
            context: Feature vector describing current situation
            alpha: Exploration parameter (higher = more exploration)

        Returns:
            UCB value — higher means "pick this arm"
        """
        theta = self.A_inv @ self.b
        predicted_reward = theta @ context
        uncertainty = np.sqrt(context @ self.A_inv @ context)
        return predicted_reward + alpha * uncertainty

    def update(self, context: np.ndarray, reward: float) -> None:
        """
        Update arm after observing reward.

        Uses Sherman-Morrison formula to incrementally update A_inv
        instead of recomputing the full inverse each time (O(d^2) vs O(d^3)).
        """
        self.A += np.outer(context, context)
        self.b += reward * context

        # Sherman-Morrison incremental inverse update
        x = context.reshape(-1, 1)
        self.A_inv -= (self.A_inv @ x @ x.T @ self.A_inv) / (
            1.0 + (x.T @ self.A_inv @ x).item()
        )


class ContextualBandit:
    """
    LinUCB contextual bandit for tool selection.

    One bandit per agent — each learns which tools work best
    for that agent's role given the current research context.
    """

    def __init__(self, config: BanditConfig, tool_names: List[str]) -> None:
        self.config = config
        self.tool_names = tool_names
        self.n_arms = len(tool_names)

        # Create one arm per tool
        self.arms = [
            LinUCBArm(config.context_dim, config.regularization)
            for _ in range(self.n_arms)
        ]

        # Tracking
        self.total_pulls = 0
        self.arm_pulls = np.zeros(self.n_arms, dtype=int)
        self.arm_rewards = np.zeros(self.n_arms, dtype=float)

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select the best tool given current context.

        Computes UCB for each arm and picks the highest.
        Ties are broken randomly.
        """
        # Ensure context has right dimension
        context = self._prepare_context(context)

        ucb_values = np.array([
            arm.get_ucb(context, self.config.alpha) for arm in self.arms
        ])

        # Break ties randomly
        max_ucb = ucb_values.max()
        max_arms = np.where(np.abs(ucb_values - max_ucb) < 1e-10)[0]
        chosen = np.random.choice(max_arms)

        return int(chosen)

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update the chosen arm with observed reward."""
        context = self._prepare_context(context)
        self.arms[arm].update(context, reward)

        self.total_pulls += 1
        self.arm_pulls[arm] += 1
        self.arm_rewards[arm] += reward

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about arm usage and performance."""
        stats = {}
        for i, name in enumerate(self.tool_names):
            pulls = self.arm_pulls[i]
            avg_reward = self.arm_rewards[i] / max(pulls, 1)
            stats[f"{name}_pulls"] = int(pulls)
            stats[f"{name}_avg_reward"] = float(avg_reward)
        stats["total_pulls"] = int(self.total_pulls)
        return stats

    def _prepare_context(self, context: np.ndarray) -> np.ndarray:
        """Ensure context vector has the right dimension."""
        context = context.astype(np.float64).flatten()

        if len(context) > self.config.context_dim:
            # Truncate if too long
            context = context[: self.config.context_dim]
        elif len(context) < self.config.context_dim:
            # Pad with zeros if too short
            padded = np.zeros(self.config.context_dim, dtype=np.float64)
            padded[: len(context)] = context
            context = padded

        return context


class MultiAgentBanditManager:
    """
    Manages one bandit per agent.

    This sits between the PPO controller (which picks the agent)
    and the environment (which needs a specific tool). Once PPO
    picks an agent, this manager asks that agent's bandit which
    tool to use.
    """

    def __init__(self, config: BanditConfig, agent_tools: Dict[str, List[str]]) -> None:
        """
        Args:
            config: Bandit hyperparameters
            agent_tools: Mapping of agent_name -> list of tool names
        """
        self.bandits: Dict[str, ContextualBandit] = {}

        for agent_name, tools in agent_tools.items():
            self.bandits[agent_name] = ContextualBandit(config, tools)

    def select_tool(self, agent_name: str, context: np.ndarray) -> Tuple[int, str]:
        """
        Select tool for a specific agent.

        Returns:
            tool_index: Index of selected tool
            tool_name: Name of selected tool
        """
        bandit = self.bandits[agent_name]
        arm = bandit.select_arm(context)
        tool_name = bandit.tool_names[arm]
        return arm, tool_name

    def update(
        self, agent_name: str, tool_index: int, context: np.ndarray, reward: float
    ) -> None:
        """Update the bandit for a specific agent."""
        self.bandits[agent_name].update(tool_index, context, reward)

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all agent bandits."""
        return {
            agent: bandit.get_stats()
            for agent, bandit in self.bandits.items()
        }