"""
Contextual Bandit (LinUCB) for tool selection.

This is RL Method #2. Each agent has its own bandit that learns
which tool works best given the current context (query type,
agent state, time remaining, etc.).

LinUCB maintains a confidence bound for each arm (tool).
It picks the arm with the highest upper confidence bound,
naturally balancing exploration vs exploitation.

Intrinsic motivation (novelty bonus) is layered on top of LinUCB:
  novelty(arm, ctx) = beta / sqrt(visit_count(arm, ctx_bucket) + 1)

This encourages the agent to explore (arm, context) pairs it has
rarely visited, decaying naturally as experience accumulates.
The context is bucketed via random projection so similar contexts
map to the same bucket without an exact-match requirement.

References:
  - Li et al., "A Contextual-Bandit Approach to Personalized News
    Article Recommendation" (2010)
  - Bellemare et al., "Unifying Count-Based Exploration and Intrinsic
    Motivation" (2016)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.config import BanditConfig


class NoveltyTracker:
    """
    Count-based intrinsic motivation via random-projection hashing.

    How it works:
      1. Project the d-dimensional context down to n_bits binary codes
         using a fixed random matrix (each bit = sign of a dot product).
      2. Hash (arm_index, binary_code) to get a bucket.
      3. The novelty bonus for that bucket is  beta / sqrt(count + 1).
         - High when the bucket is rarely visited (count small).
         - Decays toward zero as the bucket accumulates visits.

    The random projection preserves approximate cosine similarity
    (Johnson-Lindenstrauss), so similar contexts land in the same bucket
    even though we never compute exact nearest neighbours.

    Args:
        context_dim: Dimensionality of the context vectors.
        n_bits: Number of projection bits (bucket resolution).
                More bits → finer buckets, slower generalisation.
        beta: Novelty bonus scale. Larger values encourage more exploration.
        seed: RNG seed for the projection matrix (fixed for reproducibility).
    """

    def __init__(
        self,
        context_dim: int,
        n_bits: int = 8,
        beta: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.beta = beta
        rng = np.random.default_rng(seed)
        # Fixed projection matrix: (n_bits, context_dim)
        self._proj = rng.standard_normal((n_bits, context_dim))
        # Visit counts: (arm_idx, bucket_int) -> int
        self._counts: Dict[Tuple[int, int], int] = defaultdict(int)

    def _bucket(self, arm: int, context: np.ndarray) -> Tuple[int, int]:
        """Map (arm, context) to a discrete bucket key."""
        bits = (self._proj @ context) >= 0  # shape: (n_bits,)
        # Pack bits into a single integer
        bucket_int = int(np.packbits(bits, bitorder="big")[0])
        return (arm, bucket_int)

    def bonus(self, arm: int, context: np.ndarray) -> float:
        """Return the novelty bonus for this (arm, context) pair."""
        key = self._bucket(arm, context)
        count = self._counts[key]
        return self.beta / np.sqrt(count + 1)

    def update(self, arm: int, context: np.ndarray) -> None:
        """Increment the visit counter after selecting this arm."""
        key = self._bucket(arm, context)
        self._counts[key] += 1

    def get_total_visits(self) -> int:
        """Total number of (arm, context) visits recorded."""
        return sum(self._counts.values())

    def get_unique_buckets(self) -> int:
        """Number of distinct (arm, context-bucket) pairs seen."""
        return len(self._counts)


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
    LinUCB contextual bandit for tool selection with intrinsic motivation.

    One bandit per agent — each learns which tools work best
    for that agent's role given the current research context.

    Selection score for arm i:
        score_i = UCB_i(context) + novelty_bonus_i(context)

    The novelty bonus (via NoveltyTracker) decays as the arm-context
    combination is visited more often, so exploration is driven toward
    genuinely under-explored regions of the context space.
    Set config.novelty_beta = 0 to disable novelty and fall back to
    plain LinUCB.
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

        # Intrinsic motivation — shared across all arms so visit counts
        # are comparable (arm index is part of the bucket key)
        self.novelty = NoveltyTracker(
            context_dim=config.context_dim,
            n_bits=config.novelty_n_bits,
            beta=config.novelty_beta,
            seed=config.novelty_seed,
        )

        # Tracking
        self.total_pulls = 0
        self.arm_pulls = np.zeros(self.n_arms, dtype=int)
        self.arm_rewards = np.zeros(self.n_arms, dtype=float)
        self.novelty_bonuses = np.zeros(self.n_arms, dtype=float)  # running sum for stats

    def select_arm(self, context: np.ndarray) -> int:
        """
        Select the best tool given current context.

        Computes UCB + novelty bonus for each arm and picks the highest.
        Ties are broken randomly.

        Returns:
            Index of the chosen arm.
        """
        context = self._prepare_context(context)

        ucb_values = np.array([
            arm.get_ucb(context, self.config.alpha) for arm in self.arms
        ])
        bonus_values = np.array([
            self.novelty.bonus(i, context) for i in range(self.n_arms)
        ])
        scores = ucb_values + bonus_values

        # Break ties randomly
        max_score = scores.max()
        max_arms = np.where(np.abs(scores - max_score) < 1e-10)[0]
        chosen = int(np.random.choice(max_arms))

        return chosen

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update the chosen arm with observed reward and novelty counter."""
        context = self._prepare_context(context)

        # Update LinUCB model
        self.arms[arm].update(context, reward)

        # Record novelty bonus *before* incrementing (what the agent saw)
        bonus = self.novelty.bonus(arm, context)
        self.novelty_bonuses[arm] += bonus

        # Increment novelty visit counter
        self.novelty.update(arm, context)

        self.total_pulls += 1
        self.arm_pulls[arm] += 1
        self.arm_rewards[arm] += reward

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about arm usage, performance, and novelty."""
        stats: Dict[str, float] = {}
        for i, name in enumerate(self.tool_names):
            pulls = self.arm_pulls[i]
            avg_reward = self.arm_rewards[i] / max(pulls, 1)
            avg_novelty = self.novelty_bonuses[i] / max(pulls, 1)
            stats[f"{name}_pulls"] = int(pulls)
            stats[f"{name}_avg_reward"] = float(avg_reward)
            stats[f"{name}_avg_novelty_bonus"] = float(avg_novelty)
        stats["total_pulls"] = int(self.total_pulls)
        stats["novelty_unique_buckets"] = int(self.novelty.get_unique_buckets())
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