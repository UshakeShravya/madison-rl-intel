"""
Specialized research agents.

Each agent has a distinct role in the research process:
  - SearchAgent: finds new sources
  - EvaluatorAgent: assesses source credibility
  - SynthesisAgent: combines findings into insights
  - DeepDiveAgent: follows up on gaps in coverage

All agents share memory and communicate through AgentMessage objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np

from src.utils.data_structures import (
    AgentMessage,
    AgentRole,
    Finding,
    SharedMemory,
    Source,
    ToolName,
)


class BaseAgent(ABC):
    """Abstract base class for all research agents."""

    def __init__(self, role: AgentRole, tools: List[ToolName]) -> None:
        self.role = role
        self.tools = tools
        self.is_active = False
        self.steps_taken = 0

    @abstractmethod
    def process(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """
        Execute one step of the agent's role.

        Args:
            memory: Shared memory with current research state
            tool: Tool selected by the bandit

        Returns:
            Dict with results of this step
        """
        pass

    def get_status(self) -> Dict:
        """Return agent status for observation vector."""
        return {
            "role": self.role.value,
            "active": self.is_active,
            "steps": self.steps_taken,
        }

    def reset(self) -> None:
        """Reset agent for new episode."""
        self.is_active = False
        self.steps_taken = 0


class SearchAgent(BaseAgent):
    """
    Finds and retrieves new information sources.

    Specializes in discovering relevant sources using various
    search tools. Learns which tools work best for different
    query types through the bandit mechanism.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.SEARCH,
            tools=[ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.ACADEMIC_SEARCH],
        )

    def process(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Search for sources and add to shared memory."""
        self.is_active = True
        self.steps_taken += 1

        # Identify which subtopics need more coverage
        gaps = self._find_coverage_gaps(memory)
        target_subtopic = gaps[0] if gaps else "general"

        return {
            "action": "search",
            "target_subtopic": target_subtopic,
            "tool_used": tool.value,
            "coverage_gaps": gaps,
        }

    def _find_coverage_gaps(self, memory: SharedMemory) -> List[str]:
        """Find subtopics with lowest coverage."""
        if not memory.coverage_map:
            return memory.query.subtopics if memory.query else []

        sorted_topics = sorted(
            memory.coverage_map.items(), key=lambda x: x[1]
        )
        return [topic for topic, score in sorted_topics if score < 0.7]


class EvaluatorAgent(BaseAgent):
    """
    Assesses credibility and quality of gathered sources.

    Analyzes sources for reliability, recency, and relevance.
    Helps the system avoid low-quality information.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.EVALUATOR,
            tools=[ToolName.CREDIBILITY_SCORER, ToolName.WEB_SCRAPER],
        )

    def process(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Evaluate sources in shared memory."""
        self.is_active = True
        self.steps_taken += 1

        # Score unscored sources
        unscored = [s for s in memory.sources if s.credibility_score == 0.0]
        scored_count = 0

        for source in unscored[:5]:  # Evaluate up to 5 per step
            source.credibility_score = self._assess_credibility(source)
            scored_count += 1

        # Calculate average quality
        if memory.sources:
            avg_quality = np.mean([s.credibility_score for s in memory.sources])
        else:
            avg_quality = 0.0

        return {
            "action": "evaluate",
            "sources_scored": scored_count,
            "avg_quality": float(avg_quality),
            "tool_used": tool.value,
        }

    def _assess_credibility(self, source: Source) -> float:
        """Simple credibility heuristic based on domain and recency."""
        domain_scores = {
            "arxiv.org": 0.9,
            "nature.com": 0.95,
            "scholar.google.com": 0.85,
            "wikipedia.org": 0.7,
            "reuters.com": 0.85,
            "bbc.com": 0.8,
            "nytimes.com": 0.8,
            "github.com": 0.6,
            "medium.com": 0.4,
            "techcrunch.com": 0.65,
        }
        base = domain_scores.get(source.domain, 0.5)
        # Weight by recency
        return float(np.clip(base * (0.7 + 0.3 * source.recency_score), 0, 1))


class SynthesisAgent(BaseAgent):
    """
    Combines findings into coherent insights.

    Takes individual findings from search and evaluation,
    identifies patterns, resolves contradictions, and produces
    structured summaries.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.SYNTHESIS,
            tools=[ToolName.RELEVANCE_EXTRACTOR, ToolName.PDF_PARSER],
        )

    def process(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Synthesize findings in shared memory."""
        self.is_active = True
        self.steps_taken += 1

        # Group findings by subtopic
        by_topic = {}
        for f in memory.findings:
            if f.subtopic not in by_topic:
                by_topic[f.subtopic] = []
            by_topic[f.subtopic].append(f)

        # Identify topics with enough findings to synthesize
        synthesizable = {
            topic: findings
            for topic, findings in by_topic.items()
            if len(findings) >= 2
        }

        # Check for contradictions (simplified — high variance in confidence)
        contradictions = []
        for topic, findings in synthesizable.items():
            confidences = [f.confidence for f in findings]
            if np.std(confidences) > 0.3:
                contradictions.append(topic)

        return {
            "action": "synthesize",
            "topics_covered": len(by_topic),
            "synthesizable_topics": len(synthesizable),
            "contradictions_found": len(contradictions),
            "tool_used": tool.value,
        }


class DeepDiveAgent(BaseAgent):
    """
    Performs follow-up investigation on under-covered topics.

    Activated when coverage gaps are detected. Uses more
    targeted search strategies to fill specific knowledge gaps.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.DEEP_DIVE,
            tools=[ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.RELEVANCE_EXTRACTOR],
        )

    def process(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Deep dive into under-covered subtopics."""
        self.is_active = True
        self.steps_taken += 1

        # Find the weakest subtopic
        if memory.coverage_map:
            weakest = min(memory.coverage_map.items(), key=lambda x: x[1])
            target = weakest[0]
            current_coverage = weakest[1]
        else:
            target = "general"
            current_coverage = 0.0

        return {
            "action": "deep_dive",
            "target_subtopic": target,
            "current_coverage": float(current_coverage),
            "tool_used": tool.value,
        }