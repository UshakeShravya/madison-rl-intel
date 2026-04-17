"""Custom tools for research agents."""

from src.tools.relevance_extractor import (
    RelevanceExtractor,
    RelevanceExtractorConfig,
    Passage,
)
from src.tools.semantic_extractor import (
    SemanticRelevanceExtractor,
    SemanticExtractorConfig,
    SemanticPassage,
)
from src.tools.real_apis import (
    RealResearchEngine,
    WikipediaAPI,
    SemanticScholarAPI,
    DuckDuckGoAPI,
    RealSource,
)
from src.tools.real_tools import (
    execute_tool,
    ToolResult,
    WebScraperTool,
    APIClientTool,
    AcademicSearchTool,
    PDFParserTool,
    RelevanceExtractorTool,
    CredibilityScorerTool,
    TOOL_MAP,
)
