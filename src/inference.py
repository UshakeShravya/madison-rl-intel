"""
Live Inference — trained RL policy orchestrating real API tools.

The trained PPO meta-controller selects which agent to activate.
The contextual bandit for that agent selects which real tool to call.
Agents make actual API requests (Wikipedia, arXiv, OpenAlex, etc.)
and return real sources that are semantically ranked.

This demonstrates that the RL policy learned in simulation
transfers to directing real-world research workflows.

Usage (CLI):
    PYTHONPATH=. python -m src.inference "What are the latest advances in quantum computing?"
    PYTHONPATH=. python -m src.inference "climate change effects on biodiversity" --query_type causal

Usage (programmatic):
    from src.inference import LiveResearchSession
    session = LiveResearchSession()
    report  = session.research("reinforcement learning for robotics")
    print(report.summary())
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# ── Constants ─────────────────────────────────────────────────────────────────

_AGENTS = ["search", "evaluator", "synthesis", "deep_dive"]
_TOOLS  = [
    "web_scraper", "api_client", "academic_search",
    "pdf_parser", "relevance_extractor", "credibility_scorer",
]
_AGENT_TOOLS = {
    "search":    ["web_scraper", "api_client", "academic_search"],
    "evaluator": ["credibility_scorer", "web_scraper"],
    "synthesis": ["relevance_extractor", "pdf_parser"],
    "deep_dive": ["web_scraper", "api_client", "relevance_extractor"],
}
_MAX_STEPS   = 12   # fewer steps than training (faster demo)
_MAX_BUDGET  = 10.0
_CONTEXT_DIM = 128  # bandit context slice of observation vector
_N_TOOLS     = len(_TOOLS)
_N_AGENTS    = len(_AGENTS)


# ── Report data class ─────────────────────────────────────────────────────────

@dataclass
class ResearchReport:
    query:        str
    query_type:   str
    sources:      List[Dict]    = field(default_factory=list)
    key_passages: List[str]     = field(default_factory=list)
    agent_log:    List[Dict]    = field(default_factory=list)
    coverage:     Dict[str, float] = field(default_factory=dict)
    total_reward: float         = 0.0
    steps:        int           = 0
    elapsed_sec:  float         = 0.0

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"MADISON RL — Live Research Report",
            f"{'='*60}",
            f"Query      : {self.query}",
            f"Type       : {self.query_type}",
            f"Steps      : {self.steps}  |  Elapsed: {self.elapsed_sec:.1f}s",
            f"Est. Reward: {self.total_reward:.2f}",
            f"",
            f"Sources Found ({len(self.sources)}):",
        ]
        for i, s in enumerate(self.sources[:6], 1):
            cred = s.get('credibility', 0)
            lines.append(f"  {i}. [{s.get('source_type','?'):10s}] {s.get('title','?')[:55]}")
            lines.append(f"     url={s.get('url','')[:65]}  cred={cred:.2f}")
        if self.key_passages:
            lines += ["", "Key Passages:"]
            for i, p in enumerate(self.key_passages[:3], 1):
                lines.append(f"  {i}. {p[:180]}...")
        lines += ["", "Agent Activation Log:"]
        for step in self.agent_log:
            compat = "OK" if step.get("compatible") else "INCOMPAT"
            lines.append(
                f"  Step {step['step']:2d}  {step['agent']:10s}  "
                f"{step['tool']:22s}  [{compat}]  reward={step['reward']:+.2f}"
            )
        lines += [
            "",
            "Subtopic Coverage:",
            *[f"  {t:25s}: {v:.2f}" for t, v in self.coverage.items()],
            "=" * 60,
        ]
        return "\n".join(lines)


# ── Main session class ────────────────────────────────────────────────────────

class LiveResearchSession:
    """
    Orchestrates a live research session using the trained RL policy.

    The PPO meta-controller (trained on 5 000 simulation episodes) selects
    which specialist agent to activate at each step.  The per-agent contextual
    bandit selects which real API tool to call.  Real sources are retrieved,
    semantically ranked, and assembled into a ResearchReport.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to a PPO .pt checkpoint.  If None, the latest checkpoint in
        experiments/checkpoints/ is used.  Falls back to a random policy if no
        checkpoint exists.
    device : str
        "cpu" or "cuda".  Defaults to auto-detect.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self._ppo     = None
        self._bandits = None
        self._sem     = None
        self._device  = device
        self._ckpt    = checkpoint_path
        self._loaded  = False

    # ── Lazy load everything (heavy models only needed at .research() time) ───

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        from src.controller.ppo import PPOController
        from src.bandits.contextual_bandit import MultiAgentBanditManager
        from src.utils.config import MadisonConfig
        from src.tools.semantic_extractor import SemanticRelevanceExtractor

        cfg = MadisonConfig()
        obs_dim = 396  # 384 query_emb + 8 coverage + 4 scalars (matches training)

        # PPO only selects the agent (4 actions); the bandit selects the tool
        self._ppo = PPOController(obs_dim, _N_AGENTS, cfg.ppo)

        # Find checkpoint
        ckpt = self._ckpt
        if ckpt is None:
            ckpt_dir = Path("experiments/checkpoints")
            pts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
            ckpt = str(pts[-1]) if pts else None

        if ckpt and Path(ckpt).exists():
            self._ppo.load(ckpt)
            logger.info("Loaded PPO checkpoint: {}", ckpt)
        else:
            logger.warning("No PPO checkpoint found — using random policy (untrained).")

        self._bandits = MultiAgentBanditManager(cfg.bandit, _AGENT_TOOLS)
        self._sem     = SemanticRelevanceExtractor()
        self._loaded  = True

    # ── Public interface ──────────────────────────────────────────────────────

    def research(
        self,
        query: str,
        query_type: str = "exploratory",
        max_steps: int = _MAX_STEPS,
        verbose: bool = True,
    ) -> ResearchReport:
        """
        Run a live research session for the given query.

        The RL policy directs real API calls, builds a source collection,
        and returns a structured ResearchReport.

        Args:
            query:      Natural-language research question
            query_type: One of factual/comparative/exploratory/temporal/
                        causal/technical/opinion/quantitative
            max_steps:  Max agent activations (default 12)
            verbose:    Print progress to stdout

        Returns:
            ResearchReport with sources, passages, agent log, and coverage map
        """
        self._ensure_loaded()
        t_start = time.time()

        report = ResearchReport(query=query, query_type=query_type)

        # Build initial observation
        obs       = self._initial_obs(query)
        budget    = _MAX_BUDGET
        sources   = []       # List[ToolResult]
        all_text   = []       # collected content strings (for passage extraction)
        seen_urls  = set()    # deduplication — skip sources already retrieved
        subtopics  = self._extract_subtopics(query)
        coverage   = {t: 0.0 for t in subtopics}
        step       = 0
        done       = False

        if verbose:
            print(f"\n[Madison RL] Live research: '{query}'  ({query_type})")
            print(f"[Madison RL] Subtopics: {subtopics}")
            print(f"{'─'*70}")

        while not done and step < max_steps and budget > 0:
            # ── Agent selection: PPO (action is already agent_idx 0-3) ─────────
            action, _, _ = self._ppo.select_action(obs)
            agent_idx    = int(action) % _N_AGENTS
            agent_name   = _AGENTS[agent_idx]

            # ── Tool selection: contextual bandit ─────────────────────────────
            context              = obs[:_CONTEXT_DIM]
            tool_arm, tool_name  = self._bandits.select_tool(agent_name, context)

            # ── Real tool execution ───────────────────────────────────────────
            compatible = tool_name in _AGENT_TOOLS.get(agent_name, [])
            t0 = time.time()
            result = self._call_tool(tool_name, query, sources)
            latency = time.time() - t0
            budget -= latency * (1.0 if compatible else 1.5)

            # ── Update state (deduplicate by URL) ────────────────────────────
            url_key = result.url or f"{result.domain}:{result.title}"
            already_seen = url_key in seen_urls and result.source_type not in ("extracted", "assessment")
            if compatible and result.content and not already_seen:
                seen_urls.add(url_key)
                sources.append(result)
                all_text.append(result.content)
                coverage = self._update_coverage(
                    query, subtopics, coverage, all_text
                )
                bandit_reward = result.credibility * (0.8 if compatible else 0.1)
                self._bandits.update(agent_name, tool_arm, context, bandit_reward)

            # ── Reward estimate ───────────────────────────────────────────────
            cov_score = float(np.mean(list(coverage.values()))) if coverage else 0.0
            diversity = self._source_diversity(sources)
            reward    = (1.0 * result.credibility
                         + 0.5 * cov_score
                         - 0.3 * min(latency, 3.0)
                         + 0.4 * diversity)
            report.total_reward += reward

            # ── Logging ───────────────────────────────────────────────────────
            report.agent_log.append({
                "step":       step + 1,
                "agent":      agent_name,
                "tool":       tool_name,
                "compatible": compatible,
                "domain":     result.domain,
                "credibility": round(result.credibility, 3),
                "latency":    round(latency, 2),
                "reward":     round(reward, 3),
                "coverage":   round(cov_score, 3),
            })

            if verbose:
                compat_icon = "✓" if compatible else "✗"
                print(
                    f"  Step {step+1:2d}  [{compat_icon}]  {agent_name:10s}  "
                    f"{tool_name:22s}  cred={result.credibility:.2f}  "
                    f"cov={cov_score:.2f}  r={reward:+.2f}"
                )

            # ── Build next observation ─────────────────────────────────────────
            obs = self._make_obs(query, coverage, budget, step + 1, len(sources), diversity)
            step += 1

            # Early stopping: good coverage or budget exhausted
            if cov_score > 0.85 or budget <= 0:
                done = True

        # ── Final passage extraction ──────────────────────────────────────────
        if all_text:
            combined = " ".join(all_text)
            passages = self._sem.extract(combined, query, top_k=5)
            report.key_passages = [p.text for p in passages if p.combined_score > 0.1]

        report.sources  = [
            {"title": s.title, "url": s.url, "domain": s.domain,
             "credibility": round(s.credibility, 3),
             "source_type": s.source_type, "content_len": len(s.content)}
            for s in sources
        ]
        report.coverage     = {k: round(v, 3) for k, v in coverage.items()}
        report.steps        = step
        report.elapsed_sec  = round(time.time() - t_start, 1)

        return report

    # ── Observation construction ──────────────────────────────────────────────

    def _initial_obs(self, query: str) -> np.ndarray:
        q_emb    = self._sem.embed_query(query)      # 384-dim
        coverage = np.zeros(8, dtype=np.float32)     # 8-dim
        scalars  = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # budget, step, sources, diversity
        return np.concatenate([q_emb, coverage, scalars])

    def _make_obs(
        self, query: str, coverage: Dict[str, float],
        budget: float, step: int, n_sources: int, diversity: float,
    ) -> np.ndarray:
        q_emb    = self._sem.embed_query(query)
        # Pack up to 8 coverage values (padded with 0)
        cov_vals = list(coverage.values())[:8]
        cov_arr  = np.zeros(8, dtype=np.float32)
        cov_arr[:len(cov_vals)] = cov_vals
        scalars = np.array([
            max(0.0, budget / _MAX_BUDGET),   # budget remaining
            min(1.0, step / _MAX_STEPS),      # step fraction
            min(1.0, n_sources / 20.0),       # sources normalised
            diversity,                         # source diversity
        ], dtype=np.float32)
        return np.concatenate([q_emb, cov_arr, scalars])

    # ── Helper methods ────────────────────────────────────────────────────────

    def _call_tool(self, tool_name: str, query: str, sources: List) -> "ToolResult":
        from src.tools.real_tools import execute_tool, ToolResult
        # For credibility_scorer: score the most recent source's URL
        if tool_name == "credibility_scorer" and sources:
            last = sources[-1]
            return execute_tool(tool_name, query, url=last.url,
                                content=last.content[:200])
        # For relevance_extractor: run over most recent content
        if tool_name == "relevance_extractor" and sources:
            last = sources[-1]
            return execute_tool(tool_name, query, document=last.content,
                                source_url=last.url)
        return execute_tool(tool_name, query)

    def _extract_subtopics(self, query: str) -> List[str]:
        """Derive subtopics from the query for coverage tracking."""
        import re
        # Remove stop words, take top 6 content words as subtopics
        stop = {"what", "are", "the", "is", "a", "an", "of", "in", "on",
                "for", "how", "why", "does", "do", "can", "will", "about",
                "and", "or", "with", "to", "by", "at", "from", "this",
                "that", "these", "those", "latest", "recent", "current"}
        tokens = re.findall(r'\b[a-z]{3,}\b', query.lower())
        content_words = [t for t in tokens if t not in stop]
        subtopics = list(dict.fromkeys(content_words))[:6]  # unique, order-preserving
        if not subtopics:
            subtopics = ["general"]
        return subtopics

    def _update_coverage(
        self,
        query: str,
        subtopics: List[str],
        current: Dict[str, float],
        texts: List[str],
    ) -> Dict[str, float]:
        """Compute semantic coverage of each subtopic given retrieved texts."""
        if not texts:
            return current
        try:
            new_cov = self._sem.score_query_coverage(query, subtopics, texts[-3:])
            updated = {}
            for t in subtopics:
                old = current.get(t, 0.0)
                new = new_cov.get(t, 0.0)
                # Diminishing-returns accumulation (mirrors simulation env)
                updated[t] = old + (1.0 - old) * new * 0.4
            return updated
        except Exception as e:
            logger.debug("Coverage update failed: {}", e)
            return current

    def _source_diversity(self, sources: List) -> float:
        """Fraction of unique domains (capped at 1.0)."""
        if not sources:
            return 0.0
        unique_domains = len(set(s.domain for s in sources))
        return min(1.0, unique_domains / max(len(sources), 1))


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Madison RL — Live Research")
    parser.add_argument("query", nargs="?",
                        default="What are the latest advances in reinforcement learning?",
                        help="Research query")
    parser.add_argument("--query_type", default="exploratory",
                        choices=["factual","comparative","exploratory","temporal",
                                 "causal","technical","opinion","quantitative"])
    parser.add_argument("--steps", type=int, default=_MAX_STEPS)
    parser.add_argument("--checkpoint", default=None, help="Path to PPO .pt file")
    args = parser.parse_args()

    session = LiveResearchSession(checkpoint_path=args.checkpoint)
    report  = session.research(args.query, query_type=args.query_type,
                                max_steps=args.steps)
    print(report.summary())


if __name__ == "__main__":
    main()
