"""
Gym-compatible simulation environment for training the RL agents.

The environment simulates the research process:
1. A query arrives with subtopics
2. The agent decides which agent to activate and which tool to use
3. The environment simulates source retrieval with noise
4. Reward is computed based on relevance, coverage, diversity, latency
5. Episode ends when budget runs out or max steps reached
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.config import EnvironmentConfig, RewardConfig
from src.utils.data_structures import (
    AgentRole,
    Finding,
    QueryType,
    ResearchQuery,
    SharedMemory,
    Source,
    ToolName,
)
from src.rewards.reward_engine import RewardEngine
from src.agents.base_agent import SearchAgent, EvaluatorAgent, SynthesisAgent, DeepDiveAgent


class ResearchEnv(gym.Env):
    """
    Research simulation environment.

    Observation space:
        - query_embedding (384-dim): embedding of the current query
        - coverage_per_subtopic (8-dim): how well each subtopic is covered
        - budget_remaining (1-dim): fraction of budget left
        - step_fraction (1-dim): fraction of max steps used
        - n_sources_norm (1-dim): normalized count of sources found
        - diversity (1-dim): current source diversity score
        Total: 396 dimensions

    Action space:
        - agent_choice (4): which agent to activate (search/eval/synthesis/deep_dive)
        - tool_choice (6): which tool the agent should use
        Combined as single discrete: 4 * 6 = 24 actions
    """

    metadata = {"render_modes": ["human"]}

    # Agent and tool mappings
    AGENTS = [AgentRole.SEARCH, AgentRole.EVALUATOR, AgentRole.SYNTHESIS, AgentRole.DEEP_DIVE]
    TOOLS = [
        ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.ACADEMIC_SEARCH,
        ToolName.PDF_PARSER, ToolName.RELEVANCE_EXTRACTOR, ToolName.CREDIBILITY_SCORER,
    ]

    # Which tools each agent can actually use effectively
    AGENT_TOOL_COMPATIBILITY = {
        AgentRole.SEARCH: [ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.ACADEMIC_SEARCH],
        AgentRole.EVALUATOR: [ToolName.CREDIBILITY_SCORER, ToolName.WEB_SCRAPER],
        AgentRole.SYNTHESIS: [ToolName.RELEVANCE_EXTRACTOR, ToolName.PDF_PARSER],
        AgentRole.DEEP_DIVE: [ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.RELEVANCE_EXTRACTOR],
    }

    # Simulated tool effectiveness per query type (higher = better)
    TOOL_QUERY_AFFINITY = {
        QueryType.FACTUAL: {ToolName.WEB_SCRAPER: 0.8, ToolName.API_CLIENT: 0.9, ToolName.ACADEMIC_SEARCH: 0.5},
        QueryType.COMPARATIVE: {ToolName.WEB_SCRAPER: 0.7, ToolName.API_CLIENT: 0.6, ToolName.ACADEMIC_SEARCH: 0.8},
        QueryType.EXPLORATORY: {ToolName.WEB_SCRAPER: 0.9, ToolName.API_CLIENT: 0.5, ToolName.ACADEMIC_SEARCH: 0.7},
        QueryType.TEMPORAL: {ToolName.WEB_SCRAPER: 0.8, ToolName.API_CLIENT: 0.8, ToolName.ACADEMIC_SEARCH: 0.6},
        QueryType.CAUSAL: {ToolName.WEB_SCRAPER: 0.5, ToolName.API_CLIENT: 0.6, ToolName.ACADEMIC_SEARCH: 0.9},
        QueryType.TECHNICAL: {ToolName.WEB_SCRAPER: 0.6, ToolName.API_CLIENT: 0.7, ToolName.ACADEMIC_SEARCH: 0.9},
        QueryType.OPINION: {ToolName.WEB_SCRAPER: 0.9, ToolName.API_CLIENT: 0.4, ToolName.ACADEMIC_SEARCH: 0.5},
        QueryType.QUANTITATIVE: {ToolName.WEB_SCRAPER: 0.5, ToolName.API_CLIENT: 0.9, ToolName.ACADEMIC_SEARCH: 0.7},
    }

    def __init__(
        self,
        env_config: Optional[EnvironmentConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        super().__init__()

        self.env_config = env_config or EnvironmentConfig()
        self.reward_engine = RewardEngine(reward_config or RewardConfig())

        # Dimensions
        self.query_embed_dim = self.env_config.query_embedding_dim
        self.n_subtopics = self.env_config.n_query_types  # max subtopics per query
        self.obs_dim = self.query_embed_dim + self.n_subtopics + 4  # +4 for budget, step, sources, diversity

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.n_agents = len(self.AGENTS)
        self.n_tools = len(self.TOOLS)
        self.action_space = spaces.Discrete(self.n_agents * self.n_tools)

        # Internal state
        self.memory = SharedMemory()
        self.current_query: Optional[ResearchQuery] = None
        self.rng = np.random.default_rng(42)

        # Source pool — simulated sources with pre-assigned quality
        self.source_pool: list = []

        # Agent instances — one per role, share the same SharedMemory
        self._agents = {
            AgentRole.SEARCH: SearchAgent(),
            AgentRole.EVALUATOR: EvaluatorAgent(),
            AgentRole.SYNTHESIS: SynthesisAgent(),
            AgentRole.DEEP_DIVE: DeepDiveAgent(),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with a new random research query."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Generate a random query
        query_type = self.rng.choice(list(QueryType))
        n_subtopics = self.rng.integers(2, 6)
        subtopics = [f"subtopic_{i}" for i in range(n_subtopics)]

        self.current_query = ResearchQuery(
            text=f"Research query about {query_type.value}",
            query_type=query_type,
            embedding=self.rng.standard_normal(self.query_embed_dim).astype(np.float32),
            subtopics=subtopics,
            max_budget=10.0,
        )

        # Reset shared memory
        self.memory.reset(self.current_query)

        # Initialize coverage map with zeros for each subtopic
        for st in subtopics:
            self.memory.coverage_map[st] = 0.0

        # Generate source pool for this episode
        self._generate_source_pool()

        # Reset agent state (step counters, avoided domains, etc.)
        for agent in self._agents.values():
            agent.reset()

        obs = self._get_observation()
        info = {"query_type": query_type.value, "n_subtopics": n_subtopics}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step: activate an agent with a tool.

        Args:
            action: Integer encoding agent_choice * n_tools + tool_choice

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Decode action
        agent_idx = action // self.n_tools
        tool_idx = action % self.n_tools
        agent_role = self.AGENTS[agent_idx]
        tool = self.TOOLS[tool_idx]
        agent_instance = self._agents[agent_role]

        # Check compatibility — incompatible pairs get a penalty
        compatible = tool in self.AGENT_TOOL_COMPATIBILITY.get(agent_role, [])

        # Simulate latency
        latency = max(0.1, self.rng.normal(
            self.env_config.latency_mean,
            self.env_config.latency_std,
        ))

        # Call real agent logic — the agent reads shared memory, performs its
        # role-specific analysis (coverage gaps, credibility scoring, synthesis,
        # contradiction detection) and broadcasts messages to other agents.
        agent_result = agent_instance.process(self.memory, tool)

        if compatible:
            # Base relevance from tool-query affinity (the simulation's ground truth
            # for how effective this tool class is on this query type)
            base_affinity = self.TOOL_QUERY_AFFINITY.get(
                self.current_query.query_type, {}
            ).get(tool, 0.3)

            # Agent confidence reflects execution quality given current memory state.
            # e.g. SynthesisAgent has low confidence early (nothing to synthesize yet),
            # EvaluatorAgent confidence tracks actual avg source quality in memory.
            agent_confidence = agent_result.get("confidence", 0.5)

            # Blend tool capability with agent's state-aware execution quality
            relevance = np.clip(
                0.6 * base_affinity
                + 0.4 * agent_confidence
                + self.rng.normal(0, self.env_config.relevance_noise_std),
                0.0, 1.0,
            )

            # Use agent's analysis to target the right subtopic rather than
            # picking randomly (SearchAgent / DeepDiveAgent identify coverage gaps)
            subtopic = self._get_target_subtopic(agent_role, agent_result)

            source = self._retrieve_source(agent_role, tool, relevance)
            finding = Finding(
                content=f"{agent_result.get('action', 'research')} by {agent_role.value} using {tool.value}",
                sources=[source],
                confidence=relevance,
                subtopic=subtopic,
                agent=agent_role,
            )
            self.memory.add_finding(finding)
            self.memory.add_source(source)
        else:
            # Incompatible agent-tool pair — agent still reasons and broadcasts
            # messages (coordination still happens), but the wrong tool means
            # the retrieval quality is degraded
            relevance = 0.05
            latency *= 1.5

        # Update budget
        cost = latency * (1.0 if compatible else 1.5)
        self.memory.budget_used += cost
        self.memory.step_count += 1

        # Compute reward — same reward engine, now fed with real agent outputs
        reward_components = self.reward_engine.compute_step_reward(
            self.memory, relevance, latency
        )
        reward = reward_components["total"]

        # Check termination conditions
        budget_exceeded = self.memory.budget_used >= self.current_query.max_budget
        max_steps_reached = self.memory.step_count >= self.env_config.max_steps_per_episode
        high_coverage = self.memory.get_coverage_score() > 0.9

        terminated = budget_exceeded or high_coverage
        truncated = max_steps_reached

        # End-of-episode bonus
        if terminated or truncated:
            episode_reward = self.reward_engine.compute_episode_reward(self.memory)
            reward += episode_reward["total"]

        obs = self._get_observation()
        info = {
            "agent": agent_role.value,
            "tool": tool.value,
            "compatible": compatible,
            "relevance": relevance,
            "latency": latency,
            "reward_components": reward_components,
            "coverage": self.memory.get_coverage_score(),
            "diversity": self.memory.get_source_diversity(),
            "budget_used": self.memory.budget_used,
            "agent_action": agent_result.get("action", "unknown"),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current state."""
        # Query embedding
        query_emb = self.current_query.embedding

        # Coverage per subtopic (padded to n_subtopics)
        coverage = np.zeros(self.n_subtopics, dtype=np.float32)
        for i, st in enumerate(self.current_query.subtopics):
            if i < self.n_subtopics:
                coverage[i] = self.memory.coverage_map.get(st, 0.0)

        # Scalar features
        budget_remaining = 1.0 - (self.memory.budget_used / self.current_query.max_budget)
        step_fraction = self.memory.step_count / self.env_config.max_steps_per_episode
        n_sources_norm = min(len(self.memory.sources) / 20.0, 1.0)
        diversity = self.memory.get_source_diversity()

        scalars = np.array(
            [budget_remaining, step_fraction, n_sources_norm, diversity],
            dtype=np.float32,
        )

        return np.concatenate([query_emb, coverage, scalars])

    def _generate_source_pool(self) -> None:
        """Generate a pool of simulated sources for this episode."""
        domains = ["arxiv.org", "wikipedia.org", "reuters.com", "nature.com",
                    "github.com", "medium.com", "nytimes.com", "bbc.com",
                    "techcrunch.com", "scholar.google.com"]

        self.source_pool = []
        for i in range(self.env_config.n_source_pool):
            domain = self.rng.choice(domains)
            self.source_pool.append(Source(
                url=f"https://{domain}/article_{i}",
                title=f"Article {i} from {domain}",
                content=f"Simulated content for article {i}",
                domain=domain,
                credibility_score=self.rng.uniform(0.3, 1.0),
                recency_score=self.rng.uniform(0.1, 1.0),
            ))

    def _get_target_subtopic(self, agent_role: AgentRole, agent_result: Dict) -> str:
        """Use the agent's analysis to pick which subtopic the finding addresses.

        SearchAgent and DeepDiveAgent explicitly identify coverage gaps.
        EvaluatorAgent and SynthesisAgent don't have a subtopic target, so
        we fall back to the least-covered subtopic.
        """
        target = agent_result.get("target_subtopic")
        if target and target != "general" and target in self.current_query.subtopics:
            return target
        # Fall back to least-covered subtopic
        if self.memory.coverage_map:
            return min(self.memory.coverage_map.items(), key=lambda x: x[1])[0]
        return str(self.rng.choice(self.current_query.subtopics))

    def _retrieve_source(
        self, agent: AgentRole, tool: ToolName, relevance: float
    ) -> Source:
        """Simulate retrieving a source — better tools find better sources."""
        # Weight source selection by credibility * relevance
        weights = np.array([s.credibility_score for s in self.source_pool])
        weights = weights / weights.sum()

        idx = self.rng.choice(len(self.source_pool), p=weights)
        source = self.source_pool[idx]

        # credibility_score starts at 0.0 so EvaluatorAgent can assess it.
        # The evaluator uses source.domain and source.recency_score (both copied
        # from the pool) to compute a real credibility score when it next runs.
        return Source(
            url=source.url,
            title=source.title,
            content=source.content,
            domain=source.domain,
            credibility_score=0.0,
            relevance_score=relevance,
            recency_score=source.recency_score,
            retrieved_by=agent,
            tool_used=tool,
        )