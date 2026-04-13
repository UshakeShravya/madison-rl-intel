"""
PPO (Proximal Policy Optimization) meta-controller.

This is RL Method #1. It decides:
  - Which agent to activate (search, evaluator, synthesis, deep_dive)
  - Which tool that agent should use

Uses GAE (Generalized Advantage Estimation) for stable training
and clipped surrogate objective to prevent destructive updates.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from src.utils.config import PPOConfig


class PolicyNetwork(nn.Module):
    """
    Actor-Critic network for PPO.

    Shared backbone → policy head (actor) + value head (critic)
    """

    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig) -> None:
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(config.n_layers):
            layers.extend([
                nn.Linear(in_dim, config.hidden_dim),
                nn.ReLU(),
            ])
            in_dim = config.hidden_dim
        self.shared = nn.Sequential(*layers)

        # Policy head — outputs action probabilities
        self.policy_head = nn.Linear(config.hidden_dim, n_actions)

        # Value head — outputs state value estimate
        self.value_head = nn.Linear(config.hidden_dim, 1)

        # Initialize weights with small values for stable start
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization — standard for PPO."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        # Policy head gets smaller init for more uniform initial probabilities
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # Value head gets unit init
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """
        Forward pass.

        Returns:
            dist: Categorical distribution over actions
            value: State value estimate
        """
        features = self.shared(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value

    def get_action(self, obs: torch.Tensor) -> Tuple[int, float, float]:
        """
        Sample an action for environment interaction.

        Returns:
            action: sampled action index
            log_prob: log probability of the action
            value: state value estimate
        """
        with torch.no_grad():
            dist, value = self.forward(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()


class RolloutBuffer:
    """
    Stores experience from environment interaction.

    Collects (obs, action, reward, done, log_prob, value) tuples
    then computes GAE advantages when requested.
    """

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a single transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.

        GAE formula: A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        This balances bias vs variance in advantage estimation.
        High lambda (close to 1) = low bias, high variance
        Low lambda (close to 0) = high bias, low variance
        We use 0.95 which is the standard sweet spot.
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # Work backwards from the last step
        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            # If episode ended, next value is 0
            mask = 1.0 - float(dones[t])

            # TD error: delta = r + gamma * V(next) - V(current)
            delta = rewards[t] + gamma * next_value * mask - values[t]

            # GAE accumulation
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae

            next_value = values[t]

        # Returns = advantages + values (used for value loss)
        returns = advantages + values

        return advantages, returns

    def get_batches(
        self, advantages: np.ndarray, returns: np.ndarray, batch_size: int
    ):
        """Yield random mini-batches for PPO update."""
        n = len(self.observations)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield {
                "observations": torch.FloatTensor(
                    np.array([self.observations[i] for i in batch_idx])
                ),
                "actions": torch.LongTensor(
                    [self.actions[i] for i in batch_idx]
                ),
                "log_probs": torch.FloatTensor(
                    [self.log_probs[i] for i in batch_idx]
                ),
                "advantages": torch.FloatTensor(advantages[batch_idx]),
                "returns": torch.FloatTensor(returns[batch_idx]),
            }

    def clear(self) -> None:
        """Clear buffer for next rollout."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.observations)


class PPOController:
    """
    PPO training controller.

    Handles the full training loop:
    1. Collect rollout (interact with environment)
    2. Compute GAE advantages
    3. Update policy with clipped objective
    4. Repeat
    """

    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig) -> None:
        self.config = config

        # Set device — Mac M-chip uses MPS, otherwise CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Network and optimizer
        self.network = PolicyNetwork(obs_dim, n_actions, config).to(self.device)
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config.learning_rate, eps=1e-5
        )

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.total_updates = 0

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        """Select action given observation."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, value = self.network.get_action(obs_tensor)
        return action, log_prob, value

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Store a transition in the buffer."""
        self.buffer.store(obs, action, reward, done, log_prob, value)

    def update(self) -> Dict[str, float]:
        """
        Run PPO update using collected rollout.

        Returns dict of training metrics.
        """
        # Get last value for GAE computation
        last_obs = self.buffer.observations[-1]
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            _, last_value = self.network(obs_tensor)
            last_value = last_value.item()

        # Compute advantages using GAE
        advantages, returns = self.buffer.compute_gae(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        # Normalize advantages (standard practice for PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_batches = 0

        for _ in range(self.config.n_epochs):
            for batch in self.buffer.get_batches(
                advantages, returns, self.config.batch_size
            ):
                obs = batch["observations"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["log_probs"].to(self.device)
                advs = batch["advantages"].to(self.device)
                rets = batch["returns"].to(self.device)

                # Forward pass
                dist, values = self.network(obs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                ) * advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, rets)

                # Combined loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coeff * value_loss
                    - self.config.entropy_coeff * entropy
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    kl = (old_log_probs - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl
                n_batches += 1

        # Clear buffer
        self.buffer.clear()
        self.total_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_batches, 1),
            "value_loss": total_value_loss / max(n_batches, 1),
            "entropy": total_entropy / max(n_batches, 1),
            "approx_kl": total_kl / max(n_batches, 1),
            "n_updates": self.total_updates,
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_updates = checkpoint["total_updates"]