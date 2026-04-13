"""
Core data structures shared across agents, controller, and environment.
These define the communication protocol between all system components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ──────────────────────────────────────────────
# Enums — categories the system understands
# ──────────────────────────────────────────────
class QueryType(Enum):
    """Categories of research queries the system handles."""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    TECHNICAL = "technical"
    OPINION = "opinion"
    QUANTITATIVE = "quantitative"


class AgentRole(Enum):
    """Roles for specialized agents."""
    SEARCH = "search"
    EVALUATOR = "evaluator"
    SYNTHESIS = "synthesis"
    DEEP_DIVE = "deep_dive"


class ToolName(Enum):
    """Available tools in the system."""
    WEB_SCRAPER = "web_scraper"
    API_CLIENT = "api_client"
    ACADEMIC_SEARCH = "academic_search"
    PDF_PARSER = "pdf_parser"
    RELEVANCE_EXTRACTOR = "relevance_extractor"
    CREDIBILITY_SCORER = "credibility_scorer"


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class ResearchQuery:
    """A user's research query with metadata."""
    text: str
    query_type: QueryType
    embedding: Optional[np.ndarray] = None
    subtopics: List[str] = field(default_factory=list)
    max_budget: float = 10.0


@dataclass
class Source:
    """A single information source discovered during research."""
    url: str
    title: str
    content: str
    domain: str
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    recency_score: float = 0.0
    retrieved_by: Optional[AgentRole] = None
    tool_used: Optional[ToolName] = None


@dataclass
class Finding:
    """A processed piece of information extracted from sources."""
    content: str
    sources: List[Source]
    confidence: float
    subtopic: str
    agent: AgentRole


@dataclass
class AgentMessage:
    """Message passed between agents via shared memory."""
    sender: AgentRole
    content: Dict[str, Any]
    confidence: float
    timestamp: float
    message_type: str = "finding"


@dataclass
class SharedMemory:
    """Shared state accessible by all agents during a research episode."""
    query: Optional[ResearchQuery] = None
    findings: List[Finding] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    coverage_map: Dict[str, float] = field(default_factory=dict)
    messages: List[AgentMessage] = field(default_factory=list)
    budget_used: float = 0.0
    step_count: int = 0

    def add_finding(self, finding: Finding) -> None:
        """Add a finding and update coverage map."""
        self.findings.append(finding)
        current = self.coverage_map.get(finding.subtopic, 0.0)
        # Diminishing returns — each new finding on same subtopic helps less
        self.coverage_map[finding.subtopic] = min(
            1.0, current + (1 - current) * finding.confidence * 0.3
        )

    def add_source(self, source: Source) -> None:
        """Register a new source."""
        self.sources.append(source)

    def get_coverage_score(self) -> float:
        """Overall coverage across all subtopics (0-1)."""
        if not self.coverage_map:
            return 0.0
        return float(np.mean(list(self.coverage_map.values())))

    def get_source_diversity(self) -> float:
        """Diversity of source domains (0-1)."""
        if not self.sources:
            return 0.0
        domains = set(s.domain for s in self.sources)
        return min(1.0, len(domains) / max(len(self.sources), 1))

    def reset(self, query: ResearchQuery) -> None:
        """Reset for a new episode."""
        self.query = query
        self.findings.clear()
        self.sources.clear()
        self.coverage_map.clear()
        self.messages.clear()
        self.budget_used = 0.0
        self.step_count = 0


@dataclass
class StepResult:
    """Result of a single environment step."""
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeMetrics:
    """Metrics collected at the end of an episode."""
    total_reward: float = 0.0
    relevance_score: float = 0.0
    coverage_score: float = 0.0
    diversity_score: float = 0.0
    efficiency: float = 0.0
    n_sources: int = 0
    n_findings: int = 0
    n_steps: int = 0
    budget_used: float = 0.0
    agent_activations: Dict[str, int] = field(default_factory=dict)
    tool_selections: Dict[str, int] = field(default_factory=dict)