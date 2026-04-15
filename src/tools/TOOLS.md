# Custom Tools Documentation

## RelevanceExtractor

**Purpose**: Extracts the most relevant passages from documents given a research query. Used by SynthesisAgent and DeepDiveAgent to focus on useful content rather than processing entire documents.

**Algorithm**: TF-IDF vectorization + cosine similarity ranking

### Usage

```python
from src.tools.relevance_extractor import RelevanceExtractor, RelevanceExtractorConfig

# Default config
extractor = RelevanceExtractor()

# Custom config for academic text
config = RelevanceExtractorConfig(
    max_features=10000,   # Larger vocabulary
    ngram_range=(1, 3),   # Include trigrams
    min_passage_length=50, # Skip very short passages
    top_k=3,              # Return top 3 passages
)
extractor = RelevanceExtractor(config)

# Extract passages
passages = extractor.extract(document_text, query, top_k=5)
for p in passages:
    print(f"Score: {p.relevance_score:.3f} | {p.text[:100]}")

# Batch extraction across multiple documents
passages = extractor.batch_extract(documents, query, top_k_per_doc=3)

# Keyword extraction for follow-up searches
keywords = extractor.get_query_keywords(query, top_n=10)
```

### Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| max_features | 5000 | TF-IDF vocabulary size |
| ngram_range | (1,2) | Unigrams + bigrams |
| min_passage_length | 20 | Min chars to include passage |
| top_k | 5 | Default passages returned |
| relevance_threshold | 0.0 | Min score to include |
| sublinear_tf | True | Log-normalize term frequency |
| stop_words | "english" | Stop word language |

### Integration with Agents

- **SynthesisAgent**: calls `extract()` to focus on relevant sections before combining findings
- **DeepDiveAgent**: calls `get_query_keywords()` to generate refined follow-up search terms

---

## RealResearchEngine

**Purpose**: Orchestrates real API calls to Wikipedia, Semantic Scholar, and DuckDuckGo for live research queries. Used in demo mode.

### Usage

```python
from src.tools.real_apis import RealResearchEngine

engine = RealResearchEngine()
results = engine.research("quantum computing advances")

print(f"Sources: {results['total_sources']}")
print(f"Domains: {results['domains']}")
for source in results['sources']:
    print(f"[{source.source_type}] {source.title}")
    print(f"  relevance={source.relevance_score:.2f} credibility={source.credibility_score:.2f}")
```

### APIs Used

| API | Free | Key Required | Rate Limit |
|---|---|---|---|
| Wikipedia REST API | Yes | No | Polite use (0.5s delay) |
| Semantic Scholar | Yes | No | ~1 req/sec |
| DuckDuckGo Instant | Yes | No | No strict limit |

### Graceful Degradation

All API calls are wrapped in try/except. If any API fails (rate limit, timeout, network error), the engine continues with results from the remaining APIs. The system never crashes due to a single API failure.