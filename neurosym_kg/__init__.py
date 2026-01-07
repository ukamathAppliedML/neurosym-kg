"""
NeuroSym-KG: A Unified Neuro-Symbolic Reasoning Framework

A modular, extensible framework for neuro-symbolic AI that unifies
Knowledge Graphs (KGs) with Large Language Models (LLMs).

Example:
    >>> from neurosym_kg import InMemoryKG, ThinkOnGraphReasoner, MockLLMBackend, Triple
    >>> 
    >>> # Create a simple knowledge graph
    >>> kg = InMemoryKG()
    >>> kg.add_triples([
    ...     Triple("Einstein", "born_in", "Ulm"),
    ...     Triple("Ulm", "located_in", "Germany"),
    ... ])
    >>> 
    >>> # Set up reasoner
    >>> llm = MockLLMBackend()
    >>> reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm)
    >>> 
    >>> # Reason!
    >>> result = reasoner.reason("Where was Einstein born?")
    >>> print(result.answer)
"""

__version__ = "0.1.0"

# Core types
from neurosym_kg.core import (
    # Data types
    Entity,
    EntityType,
    Relation,
    RelationType,
    Triple,
    ReasoningPath,
    Subgraph,
    ReasoningResult,
    ReasoningResultStatus,
    LLMResponse,
    Message,
    # Interfaces
    KnowledgeGraph,
    MutableKnowledgeGraph,
    LLMBackend,
    Reasoner,
    # Configuration
    Config,
    get_config,
    set_config,
    # Exceptions
    NeuroSymError,
    KnowledgeGraphError,
    LLMError,
    ReasoningError,
)

# Knowledge graph backends
from neurosym_kg.knowledge_graphs import (
    BaseKnowledgeGraph,
    InMemoryKG,
    WikidataKG,
)

# LLM backends
from neurosym_kg.llm_backends import (
    BaseLLMBackend,
    MockLLMBackend,
)

# Conditionally import optional backends
try:
    from neurosym_kg.llm_backends import OpenAIBackend
except ImportError:
    pass

# Reasoners
from neurosym_kg.reasoners import (
    BaseReasoner,
    ThinkOnGraphReasoner,
    ReasoningOnGraphs,
    GraphRAGReasoner,
    SubgraphRAGReasoner,
)

# Symbolic modules
from neurosym_kg.symbolic import (
    ConstraintChecker,
    PathValidator,
    RuleEngine,
    Rule,
    TriplePattern,
)

# Evaluation
from neurosym_kg.evaluation import (
    WebQSP,
    CWQ,
    MetaQA,
    BenchmarkRunner,
    RunConfig,
    EvaluationResult,
    exact_match,
    f1_score,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "Entity",
    "EntityType",
    "Relation",
    "RelationType",
    "Triple",
    "ReasoningPath",
    "Subgraph",
    "ReasoningResult",
    "ReasoningResultStatus",
    "LLMResponse",
    "Message",
    # Interfaces
    "KnowledgeGraph",
    "MutableKnowledgeGraph",
    "LLMBackend",
    "Reasoner",
    # Configuration
    "Config",
    "get_config",
    "set_config",
    # Exceptions
    "NeuroSymError",
    "KnowledgeGraphError",
    "LLMError",
    "ReasoningError",
    # Knowledge graphs
    "BaseKnowledgeGraph",
    "InMemoryKG",
    "WikidataKG",
    # LLM backends
    "BaseLLMBackend",
    "MockLLMBackend",
    "OpenAIBackend",
    # Reasoners
    "BaseReasoner",
    "ThinkOnGraphReasoner",
    "ReasoningOnGraphs",
    "GraphRAGReasoner",
    "SubgraphRAGReasoner",
    # Symbolic
    "ConstraintChecker",
    "PathValidator",
    "RuleEngine",
    "Rule",
    "TriplePattern",
    # Evaluation
    "WebQSP",
    "CWQ",
    "MetaQA",
    "BenchmarkRunner",
    "RunConfig",
    "EvaluationResult",
    "exact_match",
    "f1_score",
]
