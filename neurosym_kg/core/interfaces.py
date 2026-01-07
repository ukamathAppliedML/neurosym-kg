"""
Core interfaces (protocols) for the NeuroSym-KG framework.

This module defines the abstract interfaces that all implementations must follow.
Using Protocol classes allows for structural subtyping and better flexibility.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from neurosym_kg.core.types import (
        Entity,
        LLMResponse,
        Message,
        ReasoningResult,
        Relation,
        Subgraph,
        Triple,
    )


@runtime_checkable
class KnowledgeGraph(Protocol):
    """
    Protocol for Knowledge Graph backends.

    This defines the interface that all KG implementations must provide,
    whether it's Wikidata, Freebase, Neo4j, or an in-memory graph.
    """

    @property
    def name(self) -> str:
        """Human-readable name of the KG backend."""
        ...

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None:
        """
        Retrieve an entity by its ID.

        Args:
            entity_id: The unique identifier of the entity

        Returns:
            The Entity if found, None otherwise
        """
        ...

    @abstractmethod
    def get_entity_by_name(self, name: str, limit: int = 10) -> list[Entity]:
        """
        Search for entities by name.

        Args:
            name: The name to search for
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_filter: list[str] | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """
        Get neighboring triples for an entity.

        Args:
            entity_id: The entity to get neighbors for
            direction: "outgoing", "incoming", or "both"
            relation_filter: Optional list of relation IDs to filter by
            limit: Maximum number of triples to return

        Returns:
            List of triples involving the entity
        """
        ...

    @abstractmethod
    def get_relations(self, entity_id: str) -> list[Relation]:
        """
        Get all relations connected to an entity.

        Args:
            entity_id: The entity ID

        Returns:
            List of relations involving the entity
        """
        ...

    @abstractmethod
    def get_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """
        Query triples with optional filters.

        Args:
            subject: Filter by subject ID
            predicate: Filter by predicate ID
            obj: Filter by object ID
            limit: Maximum number of results

        Returns:
            List of matching triples
        """
        ...

    @abstractmethod
    def get_subgraph(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
        max_triples: int = 100,
    ) -> Subgraph:
        """
        Extract a subgraph centered on given entities.

        Args:
            entity_ids: Center entities for the subgraph
            max_hops: Maximum number of hops from center entities
            max_triples: Maximum triples in the subgraph

        Returns:
            The extracted subgraph
        """
        ...

    @abstractmethod
    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        max_paths: int = 10,
    ) -> list[list[Triple]]:
        """
        Find paths between two entities.

        Args:
            source: Source entity ID
            target: Target entity ID
            max_hops: Maximum path length
            max_paths: Maximum number of paths to return

        Returns:
            List of paths, each path is a list of triples
        """
        ...


@runtime_checkable
class MutableKnowledgeGraph(KnowledgeGraph, Protocol):
    """
    Protocol for mutable Knowledge Graphs that support write operations.
    """

    @abstractmethod
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        ...

    @abstractmethod
    def add_triple(self, triple: Triple) -> bool:
        """Add a triple to the graph."""
        ...

    @abstractmethod
    def add_triples(self, triples: list[Triple]) -> int:
        """Add multiple triples. Returns count of added triples."""
        ...

    @abstractmethod
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its triples."""
        ...

    @abstractmethod
    def remove_triple(self, triple: Triple) -> bool:
        """Remove a specific triple."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entities and triples."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """
    Protocol for LLM backends.

    This defines the interface for all LLM implementations,
    whether OpenAI, Anthropic, HuggingFace, or local models.
    """

    @property
    def model_name(self) -> str:
        """The name/identifier of the model."""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum tokens the model can handle."""
        ...

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **kwargs: Additional model-specific parameters

        Returns:
            LLM response
        """
        ...

    @abstractmethod
    async def agenerate(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of generate."""
        ...


@runtime_checkable
class StreamingLLMBackend(LLMBackend, Protocol):
    """LLM backend that supports streaming responses."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response tokens."""
        ...

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream response tokens."""
        ...


@runtime_checkable
class Reasoner(Protocol):
    """
    Protocol for reasoning engines.

    This defines the interface for all reasoning paradigms,
    including ToG, RoG, GraphRAG, etc.
    """

    @property
    def name(self) -> str:
        """Name of the reasoning paradigm."""
        ...

    @abstractmethod
    def reason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Perform reasoning to answer a question.

        Args:
            question: The question to answer
            context: Optional additional context
            **kwargs: Reasoner-specific parameters

        Returns:
            The reasoning result
        """
        ...

    @abstractmethod
    async def areason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """Async version of reason."""
        ...


@runtime_checkable
class EntityLinker(Protocol):
    """
    Protocol for entity linking/recognition.

    Maps text mentions to KG entities.
    """

    @abstractmethod
    def link_entities(
        self,
        text: str,
        kg: KnowledgeGraph,
        top_k: int = 5,
    ) -> list[tuple[str, list[Entity]]]:
        """
        Link entity mentions in text to KG entities.

        Args:
            text: Text containing entity mentions
            kg: Knowledge graph to link against
            top_k: Number of candidate entities per mention

        Returns:
            List of (mention, candidate_entities) tuples
        """
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """
    Protocol for text embedding models.
    """

    @property
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        ...

    @abstractmethod
    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed."""
        ...


@runtime_checkable
class SymbolicReasoner(Protocol):
    """
    Protocol for symbolic reasoning modules.

    These perform logical inference, constraint checking, etc.
    """

    @abstractmethod
    def check_entailment(
        self,
        premises: list[Triple],
        conclusion: Triple,
    ) -> tuple[bool, float]:
        """
        Check if premises entail the conclusion.

        Returns:
            (entails, confidence)
        """
        ...

    @abstractmethod
    def infer(
        self,
        facts: list[Triple],
        rules: list[str],
        query: str,
    ) -> list[Triple]:
        """
        Perform logical inference.

        Args:
            facts: Known facts
            rules: Inference rules
            query: What to infer

        Returns:
            Inferred triples
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """
    Protocol for retrieval components.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        kg: KnowledgeGraph,
        top_k: int = 10,
    ) -> list[Triple]:
        """
        Retrieve relevant triples for a query.

        Args:
            query: The query text
            kg: Knowledge graph to retrieve from
            top_k: Number of triples to retrieve

        Returns:
            List of relevant triples
        """
        ...


@runtime_checkable
class Cache(Protocol):
    """
    Protocol for caching implementations.
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache with optional TTL in seconds."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        ...
