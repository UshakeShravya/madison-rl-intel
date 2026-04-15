"""
Specialized research agents.

Each agent has a distinct role in the research process:
  - SearchAgent: finds new sources
  - EvaluatorAgent: assesses source credibility
  - SynthesisAgent: combines findings into insights
  - DeepDiveAgent: follows up on gaps in coverage

All agents share memory and communicate through AgentMessage objects.
Agents actively read each other's messages and coordinate via SharedMemory.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

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
        self.errors: List[str] = []

    @abstractmethod
    def _execute(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Core agent logic — implemented by each subclass."""
        pass

    def process(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """
        Execute one step with full error handling and message logging.
        Wraps _execute with try/except so a failing agent never
        crashes the episode — it degrades gracefully instead.
        """
        self.is_active = True
        self.steps_taken += 1

        # Read messages from other agents before acting
        relevant_messages = self._read_messages(memory)

        try:
            result = self._execute(memory, tool)
            result["agent"] = self.role.value
            result["tool"] = tool.value
            result["success"] = True
            result["relevant_messages"] = len(relevant_messages)

            # Broadcast result to shared memory so other agents can read it
            self._broadcast(memory, result)

        except Exception as e:
            error_msg = f"{self.role.value} failed with {tool.value}: {str(e)}"
            logger.warning(error_msg)
            self.errors.append(error_msg)

            # Graceful degradation — return minimal result instead of crashing
            result = {
                "agent": self.role.value,
                "tool": tool.value,
                "success": False,
                "error": error_msg,
                "action": "fallback",
                "relevant_messages": 0,
            }

        return result

    def _read_messages(self, memory: SharedMemory) -> List[AgentMessage]:
        """
        Read messages from other agents that are relevant to this agent's role.
        This enables real inter-agent coordination — e.g. SearchAgent reads
        EvaluatorAgent's credibility warnings before choosing sources.
        """
        relevant = []
        for msg in memory.messages:
            if msg.sender != self.role:  # Don't read own messages
                relevant.append(msg)
        return relevant

    def _broadcast(self, memory: SharedMemory, result: Dict) -> None:
        """Post a message to shared memory for other agents to read."""
        msg = AgentMessage(
            sender=self.role,
            content=result,
            confidence=result.get("confidence", 0.5),
            timestamp=time.time(),
            message_type=result.get("action", "update"),
        )
        memory.messages.append(msg)

    def get_status(self) -> Dict:
        """Return agent status for observation vector."""
        return {
            "role": self.role.value,
            "active": self.is_active,
            "steps": self.steps_taken,
            "errors": len(self.errors),
        }

    def reset(self) -> None:
        """Reset agent for new episode."""
        self.is_active = False
        self.steps_taken = 0
        self.errors.clear()


class SearchAgent(BaseAgent):
    """
    Finds and retrieves new information sources.
    Reads EvaluatorAgent messages to avoid low-credibility domains.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.SEARCH,
            tools=[ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.ACADEMIC_SEARCH],
        )
        self.avoided_domains: List[str] = []

    def _execute(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Search for sources, incorporating evaluator feedback."""
        # Read evaluator warnings about low-quality domains
        for msg in memory.messages:
            if (msg.sender == AgentRole.EVALUATOR and
                    msg.message_type == "credibility_warning"):
                bad_domain = msg.content.get("domain")
                if bad_domain and bad_domain not in self.avoided_domains:
                    self.avoided_domains.append(bad_domain)

        gaps = self._find_coverage_gaps(memory)
        target_subtopic = gaps[0] if gaps else "general"

        return {
            "action": "search",
            "target_subtopic": target_subtopic,
            "coverage_gaps": gaps,
            "avoided_domains": self.avoided_domains,
            "confidence": 0.7,
        }

    def _find_coverage_gaps(self, memory: SharedMemory) -> List[str]:
        """Find subtopics with lowest coverage."""
        if not memory.coverage_map:
            return memory.query.subtopics if memory.query else []
        sorted_topics = sorted(memory.coverage_map.items(), key=lambda x: x[1])
        return [t for t, s in sorted_topics if s < 0.7]


class EvaluatorAgent(BaseAgent):
    """
    Assesses credibility and quality of gathered sources.
    Broadcasts warnings to SearchAgent about low-quality domains.
    """

    DOMAIN_SCORES = {
        "arxiv.org": 0.9, "nature.com": 0.95, "scholar.google.com": 0.85,
        "wikipedia.org": 0.7, "reuters.com": 0.85, "bbc.com": 0.8,
        "nytimes.com": 0.8, "github.com": 0.6, "medium.com": 0.4,
        "techcrunch.com": 0.65,
    }

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.EVALUATOR,
            tools=[ToolName.CREDIBILITY_SCORER, ToolName.WEB_SCRAPER],
        )

    def _execute(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Evaluate sources and broadcast credibility warnings."""
        unscored = [s for s in memory.sources if s.credibility_score == 0.0]
        scored_count = 0
        low_quality_domains = []

        for source in unscored[:5]:
            try:
                score = self._assess_credibility(source)
                source.credibility_score = score
                scored_count += 1

                # Flag low quality domains for SearchAgent
                if score < 0.5:
                    low_quality_domains.append(source.domain)
                    # Broadcast warning
                    warning_msg = AgentMessage(
                        sender=self.role,
                        content={"domain": source.domain, "score": score},
                        confidence=score,
                        timestamp=time.time(),
                        message_type="credibility_warning",
                    )
                    memory.messages.append(warning_msg)

            except Exception as e:
                logger.warning("Credibility scoring failed for {}: {}", source.url, e)
                source.credibility_score = 0.5  # Default fallback

        avg_quality = float(np.mean([s.credibility_score for s in memory.sources])) \
            if memory.sources else 0.0

        return {
            "action": "evaluate",
            "sources_scored": scored_count,
            "avg_quality": avg_quality,
            "low_quality_domains": low_quality_domains,
            "confidence": avg_quality,
        }

    def _assess_credibility(self, source: Source) -> float:
        """Credibility heuristic based on domain and recency."""
        base = self.DOMAIN_SCORES.get(source.domain, 0.5)
        return float(np.clip(base * (0.7 + 0.3 * source.recency_score), 0, 1))


class SynthesisAgent(BaseAgent):
    """
    Combines findings into coherent insights.
    Reads SearchAgent and EvaluatorAgent outputs to prioritize synthesis.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.SYNTHESIS,
            tools=[ToolName.RELEVANCE_EXTRACTOR, ToolName.PDF_PARSER],
        )

    def _execute(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Synthesize findings, prioritizing high-confidence sources."""
        # Only synthesize from high-credibility sources (reads evaluator output)
        quality_threshold = 0.6
        good_sources = [
            s for s in memory.sources
            if s.credibility_score >= quality_threshold or s.credibility_score == 0.0
        ]

        by_topic: Dict[str, List[Finding]] = {}
        for f in memory.findings:
            by_topic.setdefault(f.subtopic, []).append(f)

        synthesizable = {t: fs for t, fs in by_topic.items() if len(fs) >= 2}

        contradictions = []
        for topic, findings in synthesizable.items():
            confidences = [f.confidence for f in findings]
            if len(confidences) >= 2 and np.std(confidences) > 0.3:
                contradictions.append(topic)

        return {
            "action": "synthesize",
            "topics_covered": len(by_topic),
            "synthesizable_topics": len(synthesizable),
            "contradictions_found": len(contradictions),
            "quality_sources_used": len(good_sources),
            "confidence": min(1.0, len(synthesizable) * 0.2),
        }


class DeepDiveAgent(BaseAgent):
    """
    Follows up on under-covered topics.
    Reads SynthesisAgent messages to know which topics need more depth.
    """

    def __init__(self) -> None:
        super().__init__(
            role=AgentRole.DEEP_DIVE,
            tools=[ToolName.WEB_SCRAPER, ToolName.API_CLIENT, ToolName.RELEVANCE_EXTRACTOR],
        )

    def _execute(self, memory: SharedMemory, tool: ToolName) -> Dict:
        """Deep dive into weakest subtopic, informed by synthesis messages."""
        # Check if SynthesisAgent flagged any contradictions to resolve
        contradiction_topics = []
        for msg in memory.messages:
            if msg.sender == AgentRole.SYNTHESIS:
                contradiction_topics = msg.content.get("contradictions_found", [])

        # Prefer resolving contradictions over general gap-filling
        if contradiction_topics and isinstance(contradiction_topics, list) and len(contradiction_topics) > 0:
            target = contradiction_topics[0] if isinstance(contradiction_topics[0], str) else None
        else:
            target = None

        if not target and memory.coverage_map:
            weakest = min(memory.coverage_map.items(), key=lambda x: x[1])
            target = weakest[0]
            current_coverage = weakest[1]
        else:
            current_coverage = memory.coverage_map.get(target, 0.0) if target else 0.0

        return {
            "action": "deep_dive",
            "target_subtopic": target or "general",
            "current_coverage": float(current_coverage),
            "resolving_contradiction": len(contradiction_topics) > 0,
            "confidence": 0.6,
        }