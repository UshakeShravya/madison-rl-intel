# Custom Tools Documentation

This document describes all six tools available to research agents in the Madison RL system.
Each tool maps to a `ToolName` enum value and is dispatched via `execute_tool(tool_name, query)`.

---

## 1. RelevanceExtractor (TF-IDF)

**File:** `src/tools/relevance_extractor.py`
**Enum:** `ToolName.RELEVANCE_EXTRACTOR`
**Used by:** SynthesisAgent, DeepDiveAgent

Extracts the most relevant passages from documents using TF-IDF vectorization + cosine similarity.

```python
from src.tools.relevance_extractor import RelevanceExtractor, RelevanceExtractorConfig

extractor = RelevanceExtractor()
passages = extractor.extract(document_text, query, top_k=5)
for p in passages:
    print(f"Score: {p.relevance_score:.3f} | {p.text[:100]}")

# Batch across multiple documents
passages = extractor.batch_extract(documents, query, top_k_per_doc=3)

# Keyword extraction for follow-up searches
keywords = extractor.get_query_keywords(query, top_n=10)
```

| Parameter | Default | Description |
|---|---|---|
| max_features | 5000 | TF-IDF vocabulary size |
| ngram_range | (1,2) | Unigrams + bigrams |
| min_passage_length | 20 | Min chars to include passage |
| top_k | 5 | Passages returned |
| sublinear_tf | True | Log-normalize TF |

---

## 2. SemanticRelevanceExtractor ⭐ Custom Tool

**File:** `src/tools/semantic_extractor.py`
**Enum:** `ToolName.RELEVANCE_EXTRACTOR` (upgraded mode)
**Used by:** SynthesisAgent, DeepDiveAgent, LiveResearchSession

Uses `sentence-transformers` (all-MiniLM-L6-v2, 384-dim) for **true semantic similarity**
between query and passages. Captures meaning, paraphrasing, and synonyms — unlike TF-IDF
which only matches shared tokens.

**Why it's better than TF-IDF:**
- "machine learning model" matches "neural network training" even with no shared terms
- Handles cross-lingual paraphrasing
- Combined score = 0.7 × semantic + 0.3 × TF-IDF for robustness on short texts

```python
from src.tools.semantic_extractor import SemanticRelevanceExtractor

extractor = SemanticRelevanceExtractor()

# Extract passages
passages = extractor.extract(document, "quantum computing advances", top_k=5)
for p in passages:
    print(f"semantic={p.semantic_score:.3f} tfidf={p.tfidf_score:.3f} | {p.text[:80]}")

# Embed a query for use in PPO observation vector
embedding = extractor.embed_query("reinforcement learning for robotics")  # shape: (384,)

# Score subtopic coverage across retrieved texts
coverage = extractor.score_query_coverage(
    query="quantum computing",
    subtopics=["algorithms", "hardware", "applications"],
    sources=["IBM released a 127-qubit processor...", "Shor's algorithm..."]
)
# Returns: {"algorithms": 0.72, "hardware": 0.81, "applications": 0.54}
```

Falls back to pure TF-IDF if sentence-transformers fails to load (no internet on first run).

---

## 3. WebScraperTool

**File:** `src/tools/real_tools.py`
**Enum:** `ToolName.WEB_SCRAPER`
**Used by:** SearchAgent, EvaluatorAgent, DeepDiveAgent
**API:** Wikipedia REST API (free, no key)

Retrieves encyclopaedic summaries from Wikipedia. Returns the full opening section
of the best-matching article for any query.

```python
from src.tools.real_tools import execute_tool
result = execute_tool("web_scraper", "quantum computing")
print(result.content[:300])   # Wikipedia extract
print(result.credibility)     # 0.78
```

---

## 4. APIClientTool

**File:** `src/tools/real_tools.py`
**Enum:** `ToolName.API_CLIENT`
**Used by:** SearchAgent, DeepDiveAgent
**API:** OpenAlex Scholarly Works API (free, no key)

Queries the world's largest open scholarly database (OpenAlex). Returns paper titles,
reconstructed abstracts, citation counts, and DOIs.

Credibility scales with citation count: `min(0.97, 0.65 + citations/5000 × 0.32)`.

```python
result = execute_tool("api_client", "PPO proximal policy optimization")
print(result.title)
print(f"credibility={result.credibility:.2f}  domain={result.domain}")
```

---

## 5. AcademicSearchTool

**File:** `src/tools/real_tools.py`
**Enum:** `ToolName.ACADEMIC_SEARCH`
**Used by:** SearchAgent
**API:** arXiv XML API (free, no key)

Searches arXiv for preprints. Returns real paper abstracts with author lists and
publication dates. Credibility = 0.88 × recency_factor.

```python
result = execute_tool("academic_search", "multi-agent reinforcement learning survey")
print(result.title)
print(result.content[:300])   # Paper abstract
print(result.url)             # https://arxiv.org/abs/...
```

---

## 6. PDFParserTool

**File:** `src/tools/real_tools.py`
**Enum:** `ToolName.PDF_PARSER`
**Used by:** SynthesisAgent
**Backend:** pdfplumber (local PDFs) or arXiv abstract API (remote)

Parses PDF content into clean text. For remote queries, retrieves the arXiv
abstract page (the structured metadata that precedes the PDF). For local files,
uses pdfplumber to extract text from the first 5 pages.

```python
# Remote (arXiv abstract)
result = execute_tool("pdf_parser", "transformer attention mechanism")

# Local PDF
from src.tools.real_tools import PDFParserTool
parser = PDFParserTool()
result = parser.run("my query", pdf_path="/path/to/paper.pdf")
```

---

## 7. CredibilityScorerTool

**File:** `src/tools/real_tools.py`
**Enum:** `ToolName.CREDIBILITY_SCORER`
**Used by:** EvaluatorAgent

Multi-signal credibility scoring using:
1. **Domain reputation** — curated whitelist of 40+ academic, news, and tech domains
2. **HTTPS check** — +0.02 for secure connections
3. **Academic source bonus** — +0.05 for arxiv, doi, pubmed, ieee, acm, springer
4. **Citation count bonus** — logarithmic scaling up to +0.10
5. **Recency signal** — year extracted from URL or content, scaled to [0.6, 1.0]

```python
result = execute_tool("credibility_scorer", "test",
                      url="https://arxiv.org/abs/2301.00001",
                      citations=150)
print(f"credibility={result.credibility:.3f}")  # ~0.97
```

Domain score reference:

| Domain | Score | Domain | Score |
|---|---|---|---|
| nature.com | 0.97 | wikipedia.org | 0.74 |
| arxiv.org | 0.90 | github.com | 0.65 |
| reuters.com | 0.85 | medium.com | 0.42 |
| bbc.com | 0.82 | reddit.com | 0.35 |

---

## Tool Dispatcher

```python
from src.tools.real_tools import execute_tool, TOOL_MAP

# Run any tool by name
result = execute_tool("web_scraper", "my query")
result = execute_tool("credibility_scorer", "test", url="https://arxiv.org/abs/1234")
result = execute_tool("relevance_extractor", "query", document="long text...")

# All available tools
print(list(TOOL_MAP.keys()))
# ['web_scraper', 'api_client', 'academic_search', 'pdf_parser',
#  'relevance_extractor', 'credibility_scorer']
```

---

## RealResearchEngine (legacy demo)

**File:** `src/tools/real_apis.py`

Orchestrates Wikipedia + Semantic Scholar + DuckDuckGo for quick demos.
Superseded by `LiveResearchSession` (src/inference.py) which adds RL policy
orchestration on top of real API calls.

```python
from src.tools.real_apis import RealResearchEngine
engine = RealResearchEngine()
results = engine.research("quantum computing advances")
```
