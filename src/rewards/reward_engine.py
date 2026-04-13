"""
Multi-component reward engine for the Madison RL system.

The reward combines five signals:
  R = α·relevance + β·coverage - γ·latency + δ·diversity + ε·quality

Each component is normalized to [0, 1] before weighting.
"""

from __future__ import annotations

import numpy as np

from src.utils.config import RewardConfig
from src.utils.data_structures import SharedMemory


class RewardEngine:
    """Computes multi-component rewards for research episodes."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute_step_reward(
        self,
        memory: SharedMemory,
        new_relevance: float,
        latency: float,
    ) -> dict:
        """
        Compute reward after a single agent step.

        Args:
            memory: Current shared memory state
            new_relevance: Relevance score of newly gathered info (0-1)
            latency: Time/cost of this step

        Returns:
            Dictionary with total reward and all components
        """
        # 1. Relevance — how useful was the new information?
        relevance = np.clip(new_relevance, 0.0, 1.0)

        # 2. Coverage — how well are we covering all subtopics?
        coverage = memory.get_coverage_score()

        # 3. Latency penalty — faster is better
        latency_penalty = np.clip(latency / 2.0, 0.0, 1.0)

        # 4. Diversity — are we using varied sources?
        diversity = memory.get_source_diversity()

        # 5. Quality — combined signal (relevance * coverage gives us
        #    a proxy for "useful and complete" research)
        quality = relevance * coverage

        # Weighted sum
        total = (
            self.config.relevance_weight * relevance
            + self.config.coverage_weight * coverage
            - self.config.latency_penalty * latency_penalty
            + self.config.diversity_weight * diversity
            + self.config.quality_weight * quality
        )

        return {
            "total": float(total),
            "relevance": float(relevance),
            "coverage": float(coverage),
            "latency_penalty": float(latency_penalty),
            "diversity": float(diversity),
            "quality": float(quality),
        }

    def compute_episode_reward(self, memory: SharedMemory) -> dict:
        """
        Compute a final bonus/penalty at the end of an episode.

        This gives a big-picture signal about overall research quality,
        on top of the per-step rewards the agent already received.
        """
        coverage = memory.get_coverage_score()
        diversity = memory.get_source_diversity()
        n_findings = len(memory.findings)

        # Bonus for high coverage at the end
        coverage_bonus = 2.0 * coverage if coverage > 0.7 else 0.0

        # Penalty for wasting budget without findings
        efficiency = n_findings / max(memory.budget_used, 0.1)
        efficiency_bonus = np.clip(efficiency, 0.0, 2.0)

        # Penalty for very low diversity (over-relying on one source type)
        diversity_penalty = -1.0 if diversity < 0.2 else 0.0

        total = coverage_bonus + efficiency_bonus + diversity_penalty

        return {
            "total": float(total),
            "coverage_bonus": float(coverage_bonus),
            "efficiency_bonus": float(efficiency_bonus),
            "diversity_penalty": float(diversity_penalty),
        }