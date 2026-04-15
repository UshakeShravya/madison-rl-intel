"""
Real API integrations for live research queries.

No API keys required:
  - Wikipedia REST API (free, open)
  - Semantic Scholar API (free, open)
  - DuckDuckGo Instant Answer API (free, open)

These replace the simulation during demo mode, allowing the
trained RL policy to perform real research on actual queries.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from dataclasses import dataclass

import urllib.request
import urllib.parse
import json

from loguru import logger


@dataclass
class RealSource:
    """A real source retrieved from an actual API."""
    title: str
    url: str
    snippet: str
    domain: str
    source_type: str  # wikipedia, semantic_scholar, duckduckgo
    relevance_score: float = 0.0
    credibility_score: float = 0.0


class WikipediaAPI:
    """
    Wikipedia REST API client.
    Free, no key needed, rate limit: be polite (1 req/sec).
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, n_results: int = 3) -> List[RealSource]:
        """Search Wikipedia and return top articles."""
        results = []
        try:
            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": n_results,
                "format": "json",
                "utf8": 1,
            })
            url = f"{self.SEARCH_URL}?{params}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "MadisonRL/1.0 (research project)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                results.append(RealSource(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    snippet=snippet,
                    domain="wikipedia.org",
                    source_type="wikipedia",
                    credibility_score=0.75,
                ))
            time.sleep(0.5)  # Be polite

        except Exception as e:
            logger.warning("Wikipedia search failed for '{}': {}", query, e)

        return results

    def get_summary(self, title: str) -> Optional[str]:
        """Get the summary section of a Wikipedia article."""
        try:
            encoded = urllib.parse.quote(title.replace(" ", "_"))
            url = f"{self.BASE_URL}/page/summary/{encoded}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "MadisonRL/1.0 (research project)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return data.get("extract", "")
        except Exception as e:
            logger.warning("Wikipedia summary failed for '{}': {}", title, e)
            return None


class SemanticScholarAPI:
    """
    Semantic Scholar Academic Graph API.
    Free, no key needed for basic search.
    Returns real academic papers with citations.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def search_papers(self, query: str, n_results: int = 3) -> List[RealSource]:
        """Search for academic papers."""
        results = []
        try:
            params = urllib.parse.urlencode({
                "query": query,
                "limit": n_results,
                "fields": "title,abstract,year,citationCount,url,authors",
            })
            url = f"{self.BASE_URL}/paper/search?{params}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "MadisonRL/1.0 (research project)"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            for paper in data.get("data", []):
                title = paper.get("title", "")
                abstract = paper.get("abstract", "") or ""
                snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                citations = paper.get("citationCount", 0) or 0
                year = paper.get("year", 2020) or 2020
                paper_url = paper.get("url", "") or f"https://semanticscholar.org/paper/{paper.get('paperId','')}"

                # Credibility based on citation count
                credibility = min(0.95, 0.6 + (min(citations, 1000) / 1000) * 0.35)
                # Recency bonus
                recency = min(1.0, max(0.3, (year - 2015) / 10))

                results.append(RealSource(
                    title=title,
                    url=paper_url,
                    snippet=snippet,
                    domain="semanticscholar.org",
                    source_type="academic",
                    credibility_score=float(credibility * recency),
                ))
            time.sleep(1.0)  # Respect rate limits

        except Exception as e:
            logger.warning("Semantic Scholar search failed for '{}': {}", query, e)

        return results


class DuckDuckGoAPI:
    """
    DuckDuckGo Instant Answer API.
    Free, no key, gives quick factual answers.
    """

    BASE_URL = "https://api.duckduckgo.com/"

    def search(self, query: str) -> List[RealSource]:
        """Get instant answers from DuckDuckGo."""
        results = []
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1,
            })
            url = f"{self.BASE_URL}?{params}"
            req = urllib.request.Request(
                url, headers={"User-Agent": "MadisonRL/1.0 (research project)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            # Abstract answer
            abstract = data.get("Abstract", "")
            abstract_url = data.get("AbstractURL", "")
            if abstract and len(abstract) > 20:
                results.append(RealSource(
                    title=data.get("Heading", query),
                    url=abstract_url or "https://duckduckgo.com",
                    snippet=abstract[:300],
                    domain="duckduckgo.com",
                    source_type="web",
                    credibility_score=0.6,
                ))

            # Related topics
            for topic in data.get("RelatedTopics", [])[:2]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(RealSource(
                        title=topic.get("Text", "")[:60],
                        url=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", "")[:200],
                        domain="duckduckgo.com",
                        source_type="web",
                        credibility_score=0.55,
                    ))

        except Exception as e:
            logger.warning("DuckDuckGo search failed for '{}': {}", query, e)

        return results


class RealResearchEngine:
    """
    Orchestrates real API calls for live research queries.
    Used in demo mode — replaces simulation with actual web research.
    """

    def __init__(self) -> None:
        self.wikipedia = WikipediaAPI()
        self.semantic_scholar = SemanticScholarAPI()
        self.duckduckgo = DuckDuckGoAPI()

    def research(self, query: str, query_type: str = "general") -> Dict:
        """
        Run a real research query using all three APIs.

        Returns structured results with sources, snippets, and metadata.
        """
        logger.info("Running real research for: '{}'", query)
        all_sources = []
        timeline = []

        # Wikipedia search
        t0 = time.time()
        wiki_results = self.wikipedia.search(query, n_results=3)
        timeline.append({"api": "Wikipedia", "results": len(wiki_results), "time": round(time.time() - t0, 2)})
        all_sources.extend(wiki_results)

        # Get full summary for top Wikipedia result
        wiki_summary = None
        if wiki_results:
            wiki_summary = self.wikipedia.get_summary(wiki_results[0].title)

        # Semantic Scholar for academic content
        t0 = time.time()
        academic_results = self.semantic_scholar.search_papers(query, n_results=3)
        timeline.append({"api": "Semantic Scholar", "results": len(academic_results), "time": round(time.time() - t0, 2)})
        all_sources.extend(academic_results)

        # DuckDuckGo for quick facts
        t0 = time.time()
        ddg_results = self.duckduckgo.search(query)
        timeline.append({"api": "DuckDuckGo", "results": len(ddg_results), "time": round(time.time() - t0, 2)})
        all_sources.extend(ddg_results)

        # Score relevance (simple keyword overlap)
        query_words = set(query.lower().split())
        for source in all_sources:
            text_words = set((source.title + " " + source.snippet).lower().split())
            overlap = len(query_words & text_words) / max(len(query_words), 1)
            source.relevance_score = min(1.0, overlap * 2)

        # Sort by combined score
        all_sources.sort(
            key=lambda s: s.relevance_score * 0.5 + s.credibility_score * 0.5,
            reverse=True
        )

        return {
            "query": query,
            "total_sources": len(all_sources),
            "sources": all_sources,
            "wiki_summary": wiki_summary,
            "timeline": timeline,
            "domains": list(set(s.domain for s in all_sources)),
        }