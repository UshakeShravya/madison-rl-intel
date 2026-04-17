"""
Real tool dispatcher — maps ToolName enum values to actual API implementations.

Each tool function takes (query, context_sources) and returns a standardised dict:

    {
      "content":     str,    # retrieved text
      "title":       str,    # document title
      "url":         str,    # source URL
      "domain":      str,    # domain (e.g. "arxiv.org")
      "credibility": float,  # 0–1 credibility score
      "source_type": str,    # "web" | "academic" | "api"
      "latency":     float,  # wall-clock seconds for the call
    }

Tools implemented
-----------------
  web_scraper       → WikipediaAPI  (real Wikipedia REST API)
  api_client        → OpenAlexAPI   (free scholarly works API, no key needed)
  academic_search   → ArXivAPI      (free arXiv XML API, no key needed)
  pdf_parser        → ArXivPDFParser (fetches arXiv abstract as "PDF" content)
  relevance_extractor → RelevanceExtractor (TF-IDF cosine ranking)
  credibility_scorer  → DomainCredibilityScorer (multi-signal heuristic)

All calls are wrapped in try/except so a failed API never crashes an episode.
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


# ── Shared result type ────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    content:     str
    title:       str
    url:         str
    domain:      str
    credibility: float
    source_type: str
    latency:     float = 0.0
    metadata:    Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "content":     self.content,
            "title":       self.title,
            "url":         self.url,
            "domain":      self.domain,
            "credibility": self.credibility,
            "source_type": self.source_type,
            "latency":     self.latency,
            **self.metadata,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 12) -> Optional[bytes]:
    """HTTP GET with a polite User-Agent. Returns raw bytes or None on failure."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "MadisonRL/2.0 (academic research project)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        logger.debug("HTTP GET failed for {}: {}", url, e)
        return None


# ── 1. Web Scraper — Wikipedia REST API ──────────────────────────────────────

class WebScraperTool:
    """
    Wikipedia REST API — retrieves encyclopaedic summaries for a query.

    Free, no key, ~0.5 s polite delay.
    Returns the opening section (extract) of the best-matching article.
    """

    SEARCH_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

    def run(self, query: str) -> ToolResult:
        t0 = time.time()
        # Step 1: find best article title
        params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": query, "srlimit": 1,
            "format": "json", "utf8": 1,
        })
        raw = _get(f"{self.SEARCH_URL}?{params}")
        title = query  # fallback
        if raw:
            data = json.loads(raw)
            hits = data.get("query", {}).get("search", [])
            if hits:
                title = hits[0]["title"]

        time.sleep(0.3)

        # Step 2: get summary
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        raw2 = _get(f"{self.SUMMARY_URL}/{encoded}")
        content = ""
        url = f"https://en.wikipedia.org/wiki/{encoded}"
        if raw2:
            summary = json.loads(raw2)
            content = summary.get("extract", "")
            url     = summary.get("content_urls", {}).get("desktop", {}).get("page", url)

        return ToolResult(
            content=content[:1500] or f"Wikipedia article on: {title}",
            title=title,
            url=url,
            domain="wikipedia.org",
            credibility=0.78,
            source_type="web",
            latency=time.time() - t0,
        )


# ── 2. API Client — OpenAlex scholarly works API ─────────────────────────────

class APIClientTool:
    """
    OpenAlex API — largest free open-access scholarly database.
    No key required. Returns paper titles, abstracts, citation counts, DOIs.
    https://openalex.org/
    """

    BASE = "https://api.openalex.org/works"

    def run(self, query: str) -> ToolResult:
        t0 = time.time()
        params = urllib.parse.urlencode({
            "search": query,
            "per-page": 3,
            "select": "id,title,abstract_inverted_index,cited_by_count,publication_year,doi",
            "mailto": "research@madison-rl.edu",  # polite param
        })
        raw = _get(f"{self.BASE}?{params}")
        if not raw:
            return _fallback_result("api_client", query, t0)

        try:
            data = json.loads(raw)
            works = data.get("results", [])
            if not works:
                return _fallback_result("api_client", query, t0)

            best = works[0]
            title = best.get("title") or query

            # Reconstruct abstract from inverted index
            inv = best.get("abstract_inverted_index") or {}
            abstract = _invert_abstract(inv)

            citations = best.get("cited_by_count", 0) or 0
            year      = best.get("publication_year", 2020) or 2020
            doi       = best.get("doi", "")
            url       = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{best.get('id','')}"

            credibility = min(0.97, 0.65 + min(citations, 5000) / 5000 * 0.32)
            recency     = min(1.0, max(0.3, (year - 2015) / 10))

            return ToolResult(
                content=abstract[:1500] or f"Academic paper: {title}",
                title=title,
                url=url,
                domain="openalex.org",
                credibility=float(credibility * recency),
                source_type="academic",
                latency=time.time() - t0,
                metadata={"citations": citations, "year": year},
            )
        except Exception as e:
            logger.warning("OpenAlex parse error: {}", e)
            return _fallback_result("api_client", query, t0)


def _invert_abstract(inv: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted-index format."""
    if not inv:
        return ""
    word_pos: List[tuple] = []
    for word, positions in inv.items():
        for pos in positions:
            word_pos.append((pos, word))
    word_pos.sort()
    return " ".join(w for _, w in word_pos)


# ── 3. Academic Search — arXiv API ───────────────────────────────────────────

class AcademicSearchTool:
    """
    arXiv API — open-access preprint search, no key required.
    Returns real paper titles, abstracts, authors, and PDF links.
    https://arxiv.org/help/api/
    """

    BASE = "https://export.arxiv.org/api/query"
    NS   = "http://www.w3.org/2005/Atom"

    def run(self, query: str) -> ToolResult:
        t0 = time.time()
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 3,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        raw = _get(f"{self.BASE}?{params}")
        if not raw:
            return _fallback_result("academic_search", query, t0)

        try:
            root    = ET.fromstring(raw)
            entries = root.findall(f"{{{self.NS}}}entry")
            if not entries:
                return _fallback_result("academic_search", query, t0)

            entry   = entries[0]
            title   = (entry.findtext(f"{{{self.NS}}}title") or query).strip().replace("\n", " ")
            summary = (entry.findtext(f"{{{self.NS}}}summary") or "").strip()
            url     = entry.findtext(f"{{{self.NS}}}id") or "https://arxiv.org"
            authors = [a.findtext(f"{{{self.NS}}}name") or ""
                       for a in entry.findall(f"{{{self.NS}}}author")]
            published = entry.findtext(f"{{{self.NS}}}published") or "2020-01-01"
            year    = int(published[:4])

            recency     = min(1.0, max(0.5, (year - 2017) / 8))
            credibility = 0.88 * recency  # arXiv preprints: generally high quality

            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            content = f"{summary}\n\nAuthors: {author_str}"

            return ToolResult(
                content=content[:1500],
                title=title,
                url=url.strip(),
                domain="arxiv.org",
                credibility=float(credibility),
                source_type="academic",
                latency=time.time() - t0,
                metadata={"year": year, "authors": authors[:5]},
            )
        except Exception as e:
            logger.warning("arXiv parse error: {}", e)
            return _fallback_result("academic_search", query, t0)


# ── 4. PDF Parser — arXiv abstract fetcher ───────────────────────────────────

class PDFParserTool:
    """
    PDF Parser — fetches and parses arXiv paper abstracts as structured content.

    In a real deployment this would use pdfplumber on downloaded PDFs.
    Here it retrieves the arXiv abstract page (the metadata that precedes the PDF)
    and parses it into clean structured text — equivalent capability for an RL agent.

    Falls back to pdfplumber if a local PDF path is supplied.
    """

    _arxiv = AcademicSearchTool()

    def run(self, query: str, pdf_path: Optional[str] = None) -> ToolResult:
        t0 = time.time()

        # If a local PDF is provided, parse it with pdfplumber
        if pdf_path:
            return self._parse_local(pdf_path, query, t0)

        # Otherwise fetch from arXiv (abstract = structured PDF metadata)
        result = self._arxiv.run(query)
        result.source_type = "pdf"
        result.domain      = "arxiv.org (abstract)"
        result.latency     = time.time() - t0
        return result

    def _parse_local(self, path: str, query: str, t0: float) -> ToolResult:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join(
                    page.extract_text() or "" for page in pdf.pages[:5]
                )
            return ToolResult(
                content=text[:2000],
                title=f"PDF: {path}",
                url=f"file://{path}",
                domain="local",
                credibility=0.7,
                source_type="pdf",
                latency=time.time() - t0,
            )
        except Exception as e:
            logger.warning("pdfplumber failed for {}: {}", path, e)
            return _fallback_result("pdf_parser", query, t0)


# ── 5. Relevance Extractor — TF-IDF passage ranker ───────────────────────────

class RelevanceExtractorTool:
    """
    Relevance Extractor — ranks passages from a source by query relevance.

    Uses the custom RelevanceExtractor (TF-IDF + cosine similarity) from
    src/tools/relevance_extractor.py. When semantic mode is enabled it
    uses SemanticRelevanceExtractor for better accuracy on paraphrased text.
    """

    def __init__(self, use_semantic: bool = True) -> None:
        self.use_semantic = use_semantic
        self._extractor   = None
        self._sem         = None

    def _get_extractor(self):
        if self._extractor is None:
            from src.tools.relevance_extractor import RelevanceExtractor
            self._extractor = RelevanceExtractor()
        return self._extractor

    def _get_semantic(self):
        if self._sem is None:
            from src.tools.semantic_extractor import SemanticRelevanceExtractor
            self._sem = SemanticRelevanceExtractor()
        return self._sem

    def run(self, query: str, document: str = "", source_url: str = "") -> ToolResult:
        t0 = time.time()
        if not document:
            document = query

        top_passages = []
        if self.use_semantic:
            try:
                passages = self._get_semantic().extract(document, query, top_k=3, source_url=source_url)
                top_passages = [p.text for p in passages]
            except Exception:
                pass

        if not top_passages:
            try:
                passages = self._get_extractor().extract(document, query, top_k=3, source_url=source_url)
                top_passages = [p.text for p in passages]
            except Exception:
                top_passages = [document[:300]]

        content = "\n---\n".join(top_passages) if top_passages else document[:400]

        return ToolResult(
            content=content,
            title=f"Relevant passages for: {query[:60]}",
            url=source_url or "",
            domain="internal",
            credibility=0.85,
            source_type="extracted",
            latency=time.time() - t0,
        )


# ── 6. Credibility Scorer — multi-signal domain reputation ────────────────────

class CredibilityScorerTool:
    """
    Credibility Scorer — multi-signal heuristic for source quality.

    Signals used:
      1. Domain reputation (hardcoded whitelist of ~40 domains)
      2. HTTPS check (plain HTTP penalised)
      3. Recency signal (year extracted from URL or content)
      4. Academic source bonus (arxiv, doi, semanticscholar)
      5. Citation count (when available from OpenAlex/SemanticScholar)

    Returns a 0–1 credibility score.
    """

    # Curated domain reputation scores (based on academic/journalistic standards)
    DOMAIN_SCORES: Dict[str, float] = {
        # Academic
        "arxiv.org": 0.90,      "nature.com": 0.97,
        "science.org": 0.97,    "cell.com": 0.96,
        "semanticscholar.org": 0.88, "openalex.org": 0.88,
        "scholar.google.com": 0.87,  "pubmed.ncbi.nlm.nih.gov": 0.95,
        "ncbi.nlm.nih.gov": 0.94,    "ieeexplore.ieee.org": 0.92,
        "dl.acm.org": 0.91,          "springer.com": 0.90,
        "sciencedirect.com": 0.90,   "jstor.org": 0.89,
        # Encyclopaedic
        "wikipedia.org": 0.74,       "britannica.com": 0.82,
        "britannica.co.uk": 0.82,    "stanford.edu": 0.90,
        "mit.edu": 0.91,             "oxford.ac.uk": 0.91,
        # News / journalism
        "reuters.com": 0.85,         "apnews.com": 0.85,
        "bbc.com": 0.82,             "bbc.co.uk": 0.82,
        "nytimes.com": 0.80,         "theguardian.com": 0.79,
        "economist.com": 0.83,       "ft.com": 0.83,
        # Tech / dev
        "github.com": 0.65,          "stackoverflow.com": 0.66,
        "docs.python.org": 0.88,     "developer.mozilla.org": 0.87,
        # Lower quality
        "medium.com": 0.42,          "substack.com": 0.40,
        "techcrunch.com": 0.62,      "reddit.com": 0.35,
        "quora.com": 0.38,           "blogspot.com": 0.30,
    }

    def run(self, query: str, url: str = "", content: str = "",
            citations: int = 0) -> ToolResult:
        t0 = time.time()

        # Extract domain from URL
        domain = _extract_domain(url)
        base   = self.DOMAIN_SCORES.get(domain, 0.50)

        # HTTPS bonus
        if url.startswith("https://"):
            base = min(1.0, base + 0.02)

        # Academic source bonus
        if any(kw in domain for kw in ["arxiv", "doi", "scholar", "pubmed",
                                        "ieee", "acm", "springer", "ncbi"]):
            base = min(1.0, base + 0.05)

        # Citation count bonus (logarithmic)
        if citations > 0:
            import math
            citation_bonus = min(0.10, math.log10(citations + 1) / 50)
            base = min(1.0, base + citation_bonus)

        # Recency signal: look for year in URL or content
        year = _extract_year(url + " " + content[:200])
        if year:
            recency = min(1.0, max(0.6, (year - 2015) / 10))
            base    = base * (0.85 + 0.15 * recency)

        score_str = f"{base:.3f} for {domain or 'unknown domain'}"
        return ToolResult(
            content=f"Credibility assessment: {score_str}",
            title=f"Credibility score: {base:.3f}",
            url=url,
            domain=domain,
            credibility=float(np.clip(base, 0.0, 1.0)),
            source_type="assessment",
            latency=time.time() - t0,
            metadata={"raw_domain_score": self.DOMAIN_SCORES.get(domain, 0.50),
                      "citations": citations, "year": year},
        )


# ── Tool dispatcher ───────────────────────────────────────────────────────────

_WEB_SCRAPER         = WebScraperTool()
_API_CLIENT          = APIClientTool()
_ACADEMIC_SEARCH     = AcademicSearchTool()
_PDF_PARSER          = PDFParserTool()
_RELEVANCE_EXTRACTOR = RelevanceExtractorTool()
_CREDIBILITY_SCORER  = CredibilityScorerTool()

TOOL_MAP = {
    "web_scraper":         _WEB_SCRAPER,
    "api_client":          _API_CLIENT,
    "academic_search":     _ACADEMIC_SEARCH,
    "pdf_parser":          _PDF_PARSER,
    "relevance_extractor": _RELEVANCE_EXTRACTOR,
    "credibility_scorer":  _CREDIBILITY_SCORER,
}


def execute_tool(tool_name: str, query: str, **kwargs) -> ToolResult:
    """
    Run a named tool and return a ToolResult.

    Args:
        tool_name: One of the ToolName enum values (e.g. "web_scraper")
        query:     The research query string
        **kwargs:  Tool-specific kwargs (e.g. url=, document=, citations=)

    Returns:
        ToolResult with content, credibility, latency etc.
    """
    tool = TOOL_MAP.get(tool_name)
    if tool is None:
        logger.warning("Unknown tool: {} — returning fallback", tool_name)
        return _fallback_result(tool_name, query, time.time())
    try:
        return tool.run(query, **kwargs)
    except Exception as e:
        logger.warning("Tool {} raised: {}", tool_name, e)
        return _fallback_result(tool_name, query, time.time())


# ── Utility helpers ───────────────────────────────────────────────────────────

def _fallback_result(tool_name: str, query: str, t0: float) -> ToolResult:
    return ToolResult(
        content=f"[{tool_name}] No results for: {query}",
        title=query[:60],
        url="",
        domain="unknown",
        credibility=0.40,
        source_type="fallback",
        latency=time.time() - t0,
    )


def _extract_domain(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
        host   = parsed.netloc.lower()
        # Strip www. prefix
        return re.sub(r"^www\.", "", host)
    except Exception:
        return ""


def _extract_year(text: str) -> Optional[int]:
    """Find the most recent plausible year in a string."""
    matches = re.findall(r'\b(20[0-2]\d)\b', text)
    years   = [int(y) for y in matches if 2010 <= int(y) <= 2026]
    return max(years) if years else None


# numpy needed for clip in CredibilityScorerTool
import numpy as np
