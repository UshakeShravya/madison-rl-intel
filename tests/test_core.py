"""
Unit tests for Madison RL Intelligence Agent.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from src.utils.config import MadisonConfig, PPOConfig, BanditConfig, RewardConfig
from src.utils.data_structures import (
    AgentRole, Finding, QueryType, ResearchQuery, SharedMemory, Source, ToolName,
)
from src.rewards.reward_engine import RewardEngine
from src.environment.research_env import ResearchEnv
from src.controller.ppo import PPOController, RolloutBuffer
from src.bandits.contextual_bandit import ContextualBandit, LinUCBArm, MultiAgentBanditManager
from src.tools.relevance_extractor import RelevanceExtractor
from src.agents.base_agent import SearchAgent, EvaluatorAgent, SynthesisAgent, DeepDiveAgent


# ──────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────
class TestConfig:
    def test_default_config(self):
        config = MadisonConfig()
        assert config.ppo.learning_rate == 3e-4
        assert config.bandit.alpha == 1.0
        assert config.training.n_episodes == 5000

    def test_config_modification(self):
        config = MadisonConfig()
        config.training.n_episodes = 100
        assert config.training.n_episodes == 100


# ──────────────────────────────────────────────
# Data structure tests
# ──────────────────────────────────────────────
class TestDataStructures:
    def test_shared_memory_reset(self):
        memory = SharedMemory()
        query = ResearchQuery(text="test", query_type=QueryType.FACTUAL)
        memory.reset(query)
        assert memory.query == query
        assert len(memory.findings) == 0

    def test_shared_memory_add_finding(self):
        memory = SharedMemory()
        query = ResearchQuery(text="test", query_type=QueryType.FACTUAL, subtopics=["a"])
        memory.reset(query)
        memory.coverage_map["a"] = 0.0

        source = Source(url="http://test.com", title="Test", content="content", domain="test.com")
        finding = Finding(content="test", sources=[source], confidence=0.8, subtopic="a", agent=AgentRole.SEARCH)
        memory.add_finding(finding)

        assert len(memory.findings) == 1
        assert memory.coverage_map["a"] > 0.0

    def test_source_diversity(self):
        memory = SharedMemory()
        query = ResearchQuery(text="test", query_type=QueryType.FACTUAL)
        memory.reset(query)

        memory.add_source(Source(url="1", title="1", content="1", domain="a.com"))
        memory.add_source(Source(url="2", title="2", content="2", domain="b.com"))
        memory.add_source(Source(url="3", title="3", content="3", domain="c.com"))

        assert memory.get_source_diversity() > 0.5


# ──────────────────────────────────────────────
# Environment tests
# ──────────────────────────────────────────────
class TestEnvironment:
    def test_env_reset(self):
        env = ResearchEnv()
        obs, info = env.reset(seed=42)
        assert obs.shape == (396,)
        assert "query_type" in info

    def test_env_step(self):
        env = ResearchEnv()
        obs, _ = env.reset(seed=42)
        obs, reward, term, trunc, info = env.step(0)
        assert isinstance(reward, float)
        assert "agent" in info
        assert "tool" in info

    def test_env_episode_completes(self):
        env = ResearchEnv()
        obs, _ = env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            done = term or trunc
            steps += 1
        assert steps > 0
        assert steps <= 20


# ──────────────────────────────────────────────
# PPO tests
# ──────────────────────────────────────────────
class TestPPO:
    def test_ppo_init(self):
        # PPO now controls only agent selection (4 actions), not joint (agent, tool)
        ppo = PPOController(obs_dim=396, n_actions=4, config=PPOConfig())
        assert ppo.total_updates == 0

    def test_ppo_select_action(self):
        # Returned action is an agent index in {0, 1, 2, 3}
        ppo = PPOController(obs_dim=396, n_actions=4, config=PPOConfig())
        obs = np.random.randn(396).astype(np.float32)
        action, log_prob, value = ppo.select_action(obs)
        assert 0 <= action < 4
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_rollout_buffer(self):
        buffer = RolloutBuffer()
        for _ in range(10):
            buffer.store(
                obs=np.zeros(5), action=0, reward=1.0,
                done=False, log_prob=-0.5, value=0.5,
            )
        assert len(buffer) == 10
        advantages, returns = buffer.compute_gae(0.0, gamma=0.99, gae_lambda=0.95)
        assert len(advantages) == 10


# ──────────────────────────────────────────────
# Bandit tests
# ──────────────────────────────────────────────
class TestBandit:
    def test_linucb_arm(self):
        arm = LinUCBArm(context_dim=10)
        context = np.random.randn(10)
        ucb = arm.get_ucb(context, alpha=1.0)
        assert isinstance(ucb, float)

    def test_contextual_bandit(self):
        config = BanditConfig(context_dim=10, n_tools=3)
        bandit = ContextualBandit(config, ["tool_a", "tool_b", "tool_c"])
        context = np.random.randn(10)
        arm = bandit.select_arm(context)
        assert 0 <= arm < 3

    def test_multi_agent_manager(self):
        config = BanditConfig(context_dim=10)
        manager = MultiAgentBanditManager(config, {"agent_a": ["t1", "t2"]})
        context = np.random.randn(10)
        idx, name = manager.select_tool("agent_a", context)
        assert name in ["t1", "t2"]


# ──────────────────────────────────────────────
# Reward engine tests
# ──────────────────────────────────────────────
class TestRewardEngine:
    def test_step_reward(self):
        engine = RewardEngine(RewardConfig())
        memory = SharedMemory()
        query = ResearchQuery(text="test", query_type=QueryType.FACTUAL)
        memory.reset(query)
        result = engine.compute_step_reward(memory, new_relevance=0.8, latency=0.5)
        assert "total" in result
        assert result["relevance"] == 0.8


# ──────────────────────────────────────────────
# Custom tool tests
# ──────────────────────────────────────────────
class TestRelevanceExtractor:
    def test_extract(self):
        extractor = RelevanceExtractor()
        doc = "Machine learning is great.\n\nDeep learning uses neural networks."
        results = extractor.extract(doc, "neural networks", top_k=2)
        assert len(results) > 0
        assert results[0].relevance_score > 0

    def test_keywords(self):
        extractor = RelevanceExtractor()
        extractor.extract("AI and machine learning advances", "machine learning")
        keywords = extractor.get_query_keywords("machine learning")
        assert len(keywords) > 0


# ──────────────────────────────────────────────
# Agent tests
# ──────────────────────────────────────────────
class TestAgents:
    def test_search_agent(self):
        agent = SearchAgent()
        assert agent.role == AgentRole.SEARCH
        memory = SharedMemory()
        query = ResearchQuery(text="test", query_type=QueryType.FACTUAL, subtopics=["a", "b"])
        memory.reset(query)
        memory.coverage_map = {"a": 0.1, "b": 0.8}
        result = agent.process(memory, ToolName.WEB_SCRAPER)
        assert result["target_subtopic"] == "a"

    def test_evaluator_agent(self):
        agent = EvaluatorAgent()
        assert agent.role == AgentRole.EVALUATOR

    def test_all_agents_reset(self):
        for AgentClass in [SearchAgent, EvaluatorAgent, SynthesisAgent, DeepDiveAgent]:
            agent = AgentClass()
            agent.reset()
            assert agent.steps_taken == 0