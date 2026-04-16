"""
Configuration management for Madison RL Intelligence Agent.
Uses Pydantic for validation and YAML for persistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Environment config
# ──────────────────────────────────────────────
class EnvironmentConfig(BaseModel):
    """Configuration for the research simulation environment."""
    max_steps_per_episode: int = Field(20, description="Max agent actions per episode")
    n_source_pool: int = Field(50, description="Number of simulated sources available")
    query_embedding_dim: int = Field(384, description="Dimension of query embeddings")
    n_query_types: int = Field(8, description="Number of distinct query categories")
    relevance_noise_std: float = Field(0.1, description="Noise in relevance scoring")
    latency_mean: float = Field(0.5, description="Mean simulated latency per tool call")
    latency_std: float = Field(0.2, description="Std of simulated latency")


# ──────────────────────────────────────────────
# Agent configs
# ──────────────────────────────────────────────
class AgentConfig(BaseModel):
    """Configuration for a single specialized agent."""
    name: str
    role: str
    tools: List[str] = Field(default_factory=list)
    memory_size: int = Field(100, description="Max findings this agent can store")


class AgentsConfig(BaseModel):
    """Configuration for all agents in the system."""
    agents: List[AgentConfig] = Field(default_factory=lambda: [
        AgentConfig(
            name="search",
            role="Information gathering via web and API queries",
            tools=["web_scraper", "api_client", "academic_search"],
        ),
        AgentConfig(
            name="evaluator",
            role="Source credibility assessment and quality scoring",
            tools=["credibility_scorer", "fact_checker"],
        ),
        AgentConfig(
            name="synthesis",
            role="Combining findings into coherent summaries",
            tools=["relevance_extractor", "summarizer"],
        ),
        AgentConfig(
            name="deep_dive",
            role="Follow-up investigation on under-covered subtopics",
            tools=["web_scraper", "api_client", "relevance_extractor"],
        ),
    ])


# ──────────────────────────────────────────────
# PPO controller config
# ──────────────────────────────────────────────
class PPOConfig(BaseModel):
    """Hyperparameters for the PPO meta-controller."""
    hidden_dim: int = 256
    n_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 2048


# ──────────────────────────────────────────────
# Bandit config
# ──────────────────────────────────────────────
class BanditConfig(BaseModel):
    """Hyperparameters for the contextual bandit tool selector."""
    alpha: float = Field(1.0, description="UCB exploration parameter")
    context_dim: int = Field(128, description="Dimension of context vector for bandits")
    n_tools: int = Field(6, description="Number of available tools (arms)")
    regularization: float = Field(1.0, description="Ridge regression regularization")
    # Intrinsic motivation (novelty bonus)
    novelty_beta: float = Field(0.1, description="Novelty bonus scale (0 to disable)")
    novelty_n_bits: int = Field(8, description="Random-projection bits for context bucketing")
    novelty_seed: int = Field(0, description="Seed for the novelty projection matrix")


# ──────────────────────────────────────────────
# Reward config
# ──────────────────────────────────────────────
class RewardConfig(BaseModel):
    """Weights for multi-component reward function."""
    relevance_weight: float = Field(1.0, description="Weight for relevance score")
    coverage_weight: float = Field(0.5, description="Weight for topic coverage")
    latency_penalty: float = Field(0.3, description="Penalty for slow responses")
    diversity_weight: float = Field(0.4, description="Weight for source diversity")
    quality_weight: float = Field(0.6, description="Weight for overall quality")


# ──────────────────────────────────────────────
# Training config
# ──────────────────────────────────────────────
class TrainingConfig(BaseModel):
    """Top-level training configuration."""
    n_episodes: int = 5000
    eval_interval: int = 100
    eval_episodes: int = 20
    checkpoint_interval: int = 500
    seed: int = 42
    device: str = "auto"
    log_dir: str = "experiments/logs"
    checkpoint_dir: str = "experiments/checkpoints"


# ──────────────────────────────────────────────
# Master config — combines everything
# ──────────────────────────────────────────────
class MadisonConfig(BaseModel):
    """Master configuration combining all sub-configs."""
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    ppo: PPOConfig = Field(default_factory=PPOConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MadisonConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(), f,
                default_flow_style=False, sort_keys=False
            )