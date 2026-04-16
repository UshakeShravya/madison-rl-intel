"""
Few-shot learning evaluation for the Madison RL system.

Measures how quickly a pre-trained PPO policy adapts to a new query
domain compared to training from scratch with the same budget.

Protocol
--------
1. Pre-train a PPO on the *source domain* (TECHNICAL) for `source_episodes`.
2. For each K in `k_shots` = [1, 5, 10, 20]:
   a. **Transfer**: deep-copy pre-trained weights → fine-tune K episodes on
      target domain (OPINION) at a reduced learning rate → evaluate for
      `eval_episodes` episodes (no gradient updates during eval).
   b. **Scratch**: initialize a fresh policy → train K episodes on target
      domain at the full learning rate → evaluate for `eval_episodes` episodes.
3. Repeat for `n_seeds` random seeds.
4. Aggregate mean ± std across seeds for each (condition, K) pair.
5. Save:
     - experiments/logs/few_shot_results.json
     - experiments/logs/few_shot_learning_curve.png

Design rationale
----------------
- `evaluate_policy` never touches the optimizer or buffer — it is a pure
  rollout function so we measure the policy's current capability honestly.
- `finetune_policy` deep-copies the source policy to keep the shared
  pre-trained checkpoint intact across seeds.
- The buffer is always flushed at the end of fine-tuning even when K < 64
  (the default batch size), so even K=1 produces a gradient step.
- A log-scale x-axis reveals the diminishing returns of extra K shots.

Run directly:
    python -m src.controller.few_shot_eval
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
from matplotlib.lines import Line2D
import torch

from src.controller.ppo import PPOController
from src.controller.transfer_learning import DomainSpecificEnv, AGENT_TOOLS, train_on_domain
from src.bandits.contextual_bandit import MultiAgentBanditManager
from src.utils.config import MadisonConfig
from src.utils.data_structures import QueryType
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation / training helpers
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    ppo: PPOController,
    query_type: QueryType,
    n_episodes: int,
    config: MadisonConfig,
) -> List[float]:
    """
    Evaluate a policy with no gradient updates.

    The policy is *not* modified.  Bandit is freshly initialized each
    evaluation so its visit counts don't bleed across calls.

    Returns:
        List of per-episode total rewards.
    """
    env = DomainSpecificEnv(query_type, config.environment, config.reward)
    bandits = MultiAgentBanditManager(config.bandit, AGENT_TOOLS)
    tool_names_list = [t.value for t in env.TOOLS]
    rewards: List[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # ppo has 4 actions; the returned value IS the agent index directly
            agent_idx, _, _ = ppo.select_action(obs)
            agent_name = env.AGENTS[agent_idx].value
            context = obs[:config.bandit.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)
            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = 0
            final_action = agent_idx * env.n_tools + tool_idx
            obs, reward, term, trunc, _ = env.step(final_action)
            done = term or trunc
            episode_reward += reward

        rewards.append(episode_reward)

    return rewards


def finetune_policy(
    source_ppo: PPOController,
    query_type: QueryType,
    n_episodes: int,
    config: MadisonConfig,
    lr_scale: float = 0.3,
) -> Tuple[PPOController, List[float]]:
    """
    Fine-tune a deep copy of `source_ppo` on `query_type` for `n_episodes`.

    Using a lower learning rate (lr_scale < 1) prevents the fine-tuning
    from overwriting the representations learned during pre-training too
    aggressively — the standard practice for transfer in deep RL.

    The buffer is always flushed at the end so even K=1 yields an update.

    Args:
        source_ppo: Pre-trained policy; never modified.
        query_type:  Target domain.
        n_episodes:  Number of fine-tuning episodes (K).
        config:      Global config.
        lr_scale:    Learning-rate multiplier for fine-tuning.

    Returns:
        (fine-tuned PPO controller, list of per-episode training rewards)
    """
    env = DomainSpecificEnv(query_type, config.environment, config.reward)
    bandits = MultiAgentBanditManager(config.bandit, AGENT_TOOLS)
    tool_names_list = [t.value for t in env.TOOLS]

    # Deep-copy so the caller's checkpoint is untouched.
    # Use env.n_agents (4) — architecture must match source_ppo.
    ft_ppo = PPOController(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.n_agents,
        config=config.ppo,
    )
    ft_ppo.network.load_state_dict(copy.deepcopy(source_ppo.network.state_dict()))
    for param_group in ft_ppo.optimizer.param_groups:
        param_group["lr"] = config.ppo.learning_rate * lr_scale

    train_rewards: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # 4-action PPO: returned value is agent_idx directly
            agent_idx, log_prob, value = ft_ppo.select_action(obs)
            agent_name = env.AGENTS[agent_idx].value
            context = obs[:config.bandit.context_dim]
            tool_arm, tool_name = bandits.select_tool(agent_name, context)
            if tool_name in tool_names_list:
                tool_idx = tool_names_list.index(tool_name)
            else:
                tool_idx = 0
            final_action = agent_idx * env.n_tools + tool_idx
            next_obs, reward, term, trunc, _ = env.step(final_action)
            done = term or trunc
            # Store agent_idx (0-3) — keeps importance ratio correct
            ft_ppo.store_transition(obs, agent_idx, reward, done, log_prob, value)
            bandits.update(agent_name, tool_arm, context, reward)
            episode_reward += reward
            obs = next_obs

        train_rewards.append(episode_reward)

    # Always flush buffer — handles K < batch_size gracefully
    if len(ft_ppo.buffer) > 0:
        ft_ppo.update()

    return ft_ppo, train_rewards


def train_from_scratch(
    query_type: QueryType,
    n_episodes: int,
    config: MadisonConfig,
) -> Tuple[PPOController, List[float]]:
    """
    Train a fresh policy on `query_type` for `n_episodes`.

    Identical training loop to `finetune_policy` but with a randomly
    initialized network and the full learning rate.  Buffer is always
    flushed at the end so K=1 still yields a gradient step.

    Returns:
        (trained PPO controller, list of per-episode training rewards)
    """
    env = DomainSpecificEnv(query_type, config.environment, config.reward)
    bandits = MultiAgentBanditManager(config.bandit, AGENT_TOOLS)
    tool_names_list = [t.value for t in env.TOOLS]

    ppo = PPOController(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.n_agents,  # 4-action agent selector
        config=config.ppo,
    )

    train_rewards: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            agent_idx, log_prob, value = ppo.select_action(obs)
            agent_name = env.AGENTS[agent_idx].value
            context = obs[:config.bandit.context_dim]
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

        train_rewards.append(episode_reward)

    if len(ppo.buffer) > 0:
        ppo.update()

    return ppo, train_rewards


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment driver
# ──────────────────────────────────────────────────────────────────────────────

def run_few_shot_experiment(
    source_domain: QueryType = QueryType.TECHNICAL,
    target_domain: QueryType = QueryType.OPINION,
    source_episodes: int = 300,
    k_shots: Optional[List[int]] = None,
    eval_episodes: int = 30,
    n_seeds: int = 3,
    output_dir: str = "experiments/logs",
) -> Dict:
    """
    Run the full few-shot learning evaluation experiment.

    For each seed:
      1. Pre-train one policy on `source_domain` for `source_episodes`.
      2. For every K in `k_shots`:
           - Transfer: fine-tune the pre-trained policy for K episodes, then
             evaluate on `target_domain` for `eval_episodes` episodes.
           - Scratch:  train a fresh policy for K episodes, then evaluate.
      3. Record the mean eval reward for each (condition, K, seed).

    Results are aggregated across seeds and saved as JSON + PNG.

    Args:
        source_domain:   Domain used for pre-training.
        target_domain:   Domain used for fine-tuning and evaluation.
        source_episodes: Number of pre-training episodes.
        k_shots:         List of K values to evaluate (default [1, 5, 10, 20]).
        eval_episodes:   Evaluation episodes per (condition, K, seed).
        n_seeds:         Number of independent random seeds.
        output_dir:      Directory for JSON + PNG output.

    Returns:
        Serializable results dict (same structure written to JSON).
    """
    if k_shots is None:
        k_shots = [1, 5, 10, 20]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = MadisonConfig()
    t_start = time.time()

    logger.info(
        "Few-shot experiment | source={} target={} src_eps={} K={} eval={} seeds={}",
        source_domain.value, target_domain.value,
        source_episodes, k_shots, eval_episodes, n_seeds,
    )

    # Structure: condition → K → [per-seed mean eval reward]
    results: Dict[str, Dict[int, List[float]]] = {
        "transfer": {k: [] for k in k_shots},
        "scratch":  {k: [] for k in k_shots},
    }

    for seed in range(n_seeds):
        logger.info("── Seed {}/{} ──────────────────────────────", seed + 1, n_seeds)
        np.random.seed(seed * 17 + 7)
        torch.manual_seed(seed * 17 + 7)

        # ── Step 1: Pre-train on source domain ──────────────────────────
        logger.info("  [1/3] Pre-training on '{}' ({} episodes)...",
                     source_domain.value, source_episodes)
        source_ppo, _ = train_on_domain(source_domain, source_episodes, config)
        logger.info("  [1/3] Pre-training done.")

        # ── Step 2: Transfer condition — fine-tune K eps + evaluate ─────
        logger.info("  [2/3] Transfer: fine-tune + eval for K in {}...", k_shots)
        for k in k_shots:
            ft_ppo, _ = finetune_policy(source_ppo, target_domain, k, config)
            eval_rewards = evaluate_policy(ft_ppo, target_domain, eval_episodes, config)
            mean_reward = float(np.mean(eval_rewards))
            results["transfer"][k].append(mean_reward)
            logger.info("        K={:3d} → transfer eval mean = {:.3f}", k, mean_reward)

        # ── Step 3: Scratch condition — train K eps + evaluate ───────────
        logger.info("  [3/3] Scratch: train + eval for K in {}...", k_shots)
        for k in k_shots:
            scratch_ppo, _ = train_from_scratch(target_domain, k, config)
            eval_rewards = evaluate_policy(scratch_ppo, target_domain, eval_episodes, config)
            mean_reward = float(np.mean(eval_rewards))
            results["scratch"][k].append(mean_reward)
            logger.info("        K={:3d} → scratch   eval mean = {:.3f}", k, mean_reward)

    elapsed = time.time() - t_start

    # ── Serialize ────────────────────────────────────────────────────────
    serializable: Dict = {
        "config": {
            "source_domain":   source_domain.value,
            "target_domain":   target_domain.value,
            "source_episodes": source_episodes,
            "k_shots":         k_shots,
            "eval_episodes":   eval_episodes,
            "n_seeds":         n_seeds,
            "elapsed_seconds": round(elapsed, 1),
        },
    }
    for condition in ("transfer", "scratch"):
        serializable[condition] = {}
        for k in k_shots:
            seed_means = results[condition][k]
            serializable[condition][str(k)] = {
                "seed_means": seed_means,
                "mean":       float(np.mean(seed_means)),
                "std":        float(np.std(seed_means)),
            }

    out_json = output_path / "few_shot_results.json"
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved → {}", out_json)

    _plot_few_shot_results(results, serializable["config"], k_shots, output_path)

    # ── Console summary ──────────────────────────────────────────────────
    logger.info("")
    logger.info("━━━ Few-Shot Results Summary ━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  K  │ Transfer (mean±std)  │ Scratch (mean±std)  │ Advantage")
    logger.info("  ───┼─────────────────────┼─────────────────────┼──────────")
    for k in k_shots:
        t_mean = serializable["transfer"][str(k)]["mean"]
        t_std  = serializable["transfer"][str(k)]["std"]
        s_mean = serializable["scratch"][str(k)]["mean"]
        s_std  = serializable["scratch"][str(k)]["std"]
        adv = 100.0 * (t_mean - s_mean) / (abs(s_mean) + 1e-6)
        logger.info(
            "  {:2d} │ {:7.3f} ± {:5.3f}      │ {:7.3f} ± {:5.3f}      │ {:+.1f}%",
            k, t_mean, t_std, s_mean, s_std, adv,
        )
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Elapsed: {:.1f}s", elapsed)

    return serializable


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot_few_shot_results(
    results: Dict[str, Dict[int, List[float]]],
    cfg: Dict,
    k_shots: List[int],
    output_path: Path,
) -> None:
    """
    Save a three-panel few-shot learning curve.

    Panel A — Adaptation curve
        Mean eval reward vs K shots for Transfer and Scratch, with ±1-std bands.
        X-axis is log-scaled to reveal diminishing-returns behaviour.

    Panel B — Transfer advantage
        Bar chart of relative improvement: (transfer − scratch) / |scratch| × 100 %.
        Positive = transfer wins; negative = scratch wins.

    Panel C — Per-seed scatter
        Individual seed values as dots with means as horizontal bars, making
        variance and outliers immediately visible without hiding them in the band.
    """
    transfer_means = np.array([np.mean(results["transfer"][k]) for k in k_shots])
    transfer_stds  = np.array([np.std( results["transfer"][k]) for k in k_shots])
    scratch_means  = np.array([np.mean(results["scratch"][k])  for k in k_shots])
    scratch_stds   = np.array([np.std( results["scratch"][k])  for k in k_shots])

    transfer_color = "#3B8BD4"
    scratch_color  = "#E8593C"
    x = np.array(k_shots, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Few-Shot Adaptation: {cfg['source_domain']} → {cfg['target_domain']}  "
        f"(pre-trained {cfg['source_episodes']} eps · eval {cfg['eval_episodes']} eps/K · "
        f"{cfg['n_seeds']} seeds)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    # ── Panel A: Adaptation curve ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, transfer_means, "o-", color=transfer_color, lw=2.2,
            label="Transfer (pre-train → fine-tune)", zorder=3)
    ax.fill_between(
        x,
        transfer_means - transfer_stds,
        transfer_means + transfer_stds,
        alpha=0.15, color=transfer_color,
    )
    ax.plot(x, scratch_means, "s--", color=scratch_color, lw=2.2,
            label="From scratch", zorder=3)
    ax.fill_between(
        x,
        scratch_means - scratch_stds,
        scratch_means + scratch_stds,
        alpha=0.15, color=scratch_color,
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(k_shots)
    ax.set_xticklabels([str(k) for k in k_shots])
    ax.set_xlabel("Fine-tuning episodes (K)", fontsize=11)
    ax.set_ylabel(f"Mean eval reward ({cfg['eval_episodes']} eps)", fontsize=11)
    ax.set_title("Adaptation Curve", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate each point with its mean value
    for xi, (tm, sm) in zip(x, zip(transfer_means, scratch_means)):
        ax.annotate(f"{tm:.2f}", (xi, tm), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color=transfer_color)
        ax.annotate(f"{sm:.2f}", (xi, sm), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=scratch_color)

    # ── Panel B: Relative advantage ───────────────────────────────────────
    ax = axes[1]
    eps = 1e-6
    relative_gain = [
        100.0 * (tm - sm) / (abs(sm) + eps)
        for tm, sm in zip(transfer_means, scratch_means)
    ]
    bar_colors = [transfer_color if g >= 0 else scratch_color for g in relative_gain]
    bars = ax.bar([str(k) for k in k_shots], relative_gain,
                  color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.9, zorder=3)
    ax.set_xlabel("K (fine-tuning episodes)", fontsize=11)
    ax.set_ylabel("Transfer advantage (%)", fontsize=11)
    ax.set_title("Relative Gain: Transfer vs Scratch", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, g in zip(bars, relative_gain):
        offset = 1.5 if g >= 0 else -3.5
        ax.text(
            bar.get_x() + bar.get_width() / 2, g + offset,
            f"{g:+.1f}%",
            ha="center", fontsize=10, fontweight="bold",
            color=transfer_color if g >= 0 else scratch_color,
        )

    # ── Panel C: Per-seed scatter ──────────────────────────────────────────
    ax = axes[2]
    rng = np.random.default_rng(0)
    for ki, k in enumerate(k_shots):
        t_vals = np.array(results["transfer"][k])
        s_vals = np.array(results["scratch"][k])
        n = len(t_vals)
        jitter_t = (rng.random(n) - 0.5) * 0.18
        jitter_s = (rng.random(n) - 0.5) * 0.18

        ax.scatter(ki - 0.22 + jitter_t, t_vals,
                   color=transfer_color, s=55, alpha=0.8, zorder=4)
        ax.scatter(ki + 0.22 + jitter_s, s_vals,
                   color=scratch_color, marker="s", s=55, alpha=0.8, zorder=4)

        # Mean as a thick horizontal dash
        ax.hlines(t_vals.mean(), ki - 0.38, ki - 0.06,
                  colors=transfer_color, lw=2.5, zorder=5)
        ax.hlines(s_vals.mean(), ki + 0.06, ki + 0.38,
                  colors=scratch_color, lw=2.5, zorder=5)

    ax.set_xticks(range(len(k_shots)))
    ax.set_xticklabels([f"K={k}" for k in k_shots])
    ax.set_ylabel("Mean eval reward", fontsize=11)
    ax.set_title("Per-Seed Variance", fontsize=12)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=transfer_color,
                   markersize=9, label="Transfer"),
            Line2D([0], [0], marker="s", color="w", markerfacecolor=scratch_color,
                   markersize=9, label="Scratch"),
        ],
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = output_path / "few_shot_learning_curve.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved → {}", out_png)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils.logging import setup_logging
    setup_logging()
    run_few_shot_experiment(
        source_domain=QueryType.TECHNICAL,
        target_domain=QueryType.OPINION,
        source_episodes=300,
        k_shots=[1, 5, 10, 20],
        eval_episodes=30,
        n_seeds=3,
    )
