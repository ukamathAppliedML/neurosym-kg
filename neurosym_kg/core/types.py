"""
Core data types for the NeuroSym-KG framework.

This module defines the fundamental data structures used throughout the framework,
including entities, relations, triples, and reasoning results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EntityType(str, Enum):
    """Types of entities in a knowledge graph."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    WORK = "work"  # Books, movies, etc.
    THING = "thing"
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Categories of relations in a knowledge graph."""

    ATTRIBUTE = "attribute"  # has_property, born_in
    TEMPORAL = "temporal"  # before, after, during
    SPATIAL = "spatial"  # located_in, near
    CAUSAL = "causal"  # causes, leads_to
    PART_OF = "part_of"  # contains, member_of
    INSTANCE_OF = "instance_of"  # type, class
    SOCIAL = "social"  # spouse, colleague
    OTHER = "other"


class Entity(BaseModel):
    """
    Represents an entity in a knowledge graph.

    Attributes:
        id: Unique identifier (e.g., Q42 for Wikidata)
        name: Human-readable name
        aliases: Alternative names/labels
        description: Brief description of the entity
        entity_type: Category of the entity
        properties: Additional key-value properties
        embedding: Optional vector embedding
    """

    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    properties: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Entity):
            return self.id == other.id
        return False


class Relation(BaseModel):
    """
    Represents a relation/predicate in a knowledge graph.

    Attributes:
        id: Unique identifier (e.g., P31 for Wikidata "instance of")
        name: Human-readable name
        description: Description of what the relation represents
        relation_type: Category of the relation
        inverse_id: ID of the inverse relation if exists
        properties: Additional key-value properties
    """

    id: str
    name: str
    description: str = ""
    relation_type: RelationType = RelationType.OTHER
    inverse_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Relation):
            return self.id == other.id
        return False


class Triple(BaseModel):
    """
    Represents a triple (subject, predicate, object) in a knowledge graph.

    This is the fundamental unit of knowledge in a KG.

    Attributes:
        subject: The subject entity or its ID
        predicate: The relation/predicate or its ID
        object: The object entity, ID, or literal value
        confidence: Confidence score for the triple (0-1)
        source: Origin of the triple (e.g., "wikidata", "extracted")
        metadata: Additional information about the triple
    """

    model_config = ConfigDict(extra="allow")

    subject: Union[str, Entity] = ""
    predicate: Union[str, Relation] = ""
    object: Union[str, Entity] = ""
    confidence: float = 1.0
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        subject: Optional[Union[str, Entity]] = None,
        predicate: Optional[Union[str, Relation]] = None,
        object: Optional[Union[str, Entity]] = None,
        **data: Any,
    ) -> None:
        """Allow positional arguments for subject, predicate, object."""
        if subject is not None:
            data["subject"] = subject
        if predicate is not None:
            data["predicate"] = predicate
        if object is not None:
            data["object"] = object
        super().__init__(**data)

    @property
    def subject_id(self) -> str:
        """Get subject ID regardless of whether it's an Entity or string."""
        return self.subject.id if isinstance(self.subject, Entity) else self.subject

    @property
    def predicate_id(self) -> str:
        """Get predicate ID regardless of whether it's a Relation or string."""
        return self.predicate.id if isinstance(self.predicate, Relation) else self.predicate

    @property
    def object_id(self) -> str:
        """Get object ID regardless of whether it's an Entity or string."""
        return self.object.id if isinstance(self.object, Entity) else self.object

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to simple tuple of IDs."""
        return (self.subject_id, self.predicate_id, self.object_id)

    def to_text(self) -> str:
        """Convert to human-readable text."""
        subj = self.subject.name if isinstance(self.subject, Entity) else self.subject
        pred = self.predicate.name if isinstance(self.predicate, Relation) else self.predicate
        obj = self.object.name if isinstance(self.object, Entity) else self.object
        return f"({subj}, {pred}, {obj})"


@dataclass
class ReasoningPath:
    """
    Represents a path of reasoning through the knowledge graph.

    A path consists of a sequence of triples that form a logical chain
    from a starting point to a conclusion.

    Attributes:
        triples: Ordered list of triples in the path
        score: Relevance/confidence score for this path
        source_entity: Starting entity of the path
        target_entity: Ending entity of the path
        metadata: Additional path information
    """

    triples: list[Triple] = field(default_factory=list)
    score: float = 0.0
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Number of hops in the path."""
        return len(self.triples)

    @property
    def relations(self) -> list[str]:
        """Get the sequence of relation IDs in the path."""
        return [t.predicate_id for t in self.triples]

    def to_text(self) -> str:
        """Convert path to human-readable text."""
        if not self.triples:
            return "(empty path)"
        parts = []
        for t in self.triples:
            parts.append(t.to_text())
        return " -> ".join(parts)


@dataclass
class Subgraph:
    """
    Represents a subgraph extracted from a knowledge graph.

    Attributes:
        triples: Set of triples in the subgraph
        entities: Set of entities in the subgraph
        center_entity: The focal entity of the subgraph (if any)
        metadata: Additional subgraph information
    """

    triples: list[Triple] = field(default_factory=list)
    entities: set[str] = field(default_factory=set)
    center_entity: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate entities from triples if not provided."""
        if not self.entities and self.triples:
            for t in self.triples:
                self.entities.add(t.subject_id)
                self.entities.add(t.object_id)

    @property
    def size(self) -> int:
        """Number of triples in the subgraph."""
        return len(self.triples)

    def to_text(self, max_triples: int = 50) -> str:
        """Convert subgraph to text representation."""
        lines = []
        for i, t in enumerate(self.triples[:max_triples]):
            lines.append(t.to_text())
        if len(self.triples) > max_triples:
            lines.append(f"... and {len(self.triples) - max_triples} more triples")
        return "\n".join(lines)


class ReasoningResultStatus(str, Enum):
    """Status of a reasoning result."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Answer found but with low confidence
    NO_ANSWER = "no_answer"  # Could not find answer
    ERROR = "error"


@dataclass
class ReasoningResult:
    """
    The result of a reasoning operation.

    Attributes:
        answer: The final answer(s)
        status: Status of the reasoning
        paths: Reasoning paths that led to the answer
        subgraph: Retrieved subgraph (if applicable)
        confidence: Confidence score for the answer
        explanation: Human-readable explanation
        raw_llm_output: Raw LLM response (for debugging)
        metadata: Additional result information
        latency_ms: Time taken for reasoning in milliseconds
    """

    answer: str | list[str]
    status: ReasoningResultStatus = ReasoningResultStatus.SUCCESS
    paths: list[ReasoningPath] = field(default_factory=list)
    subgraph: Optional[Subgraph] = None
    confidence: float = 1.0
    explanation: str = ""
    raw_llm_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def is_successful(self) -> bool:
        """Check if reasoning was successful."""
        return self.status == ReasoningResultStatus.SUCCESS

    @property
    def primary_answer(self) -> str:
        """Get the primary answer as a string."""
        if isinstance(self.answer, list):
            return self.answer[0] if self.answer else ""
        return self.answer


@dataclass
class LLMResponse:
    """
    Response from an LLM backend.

    Attributes:
        content: The text content of the response
        model: Model that generated the response
        usage: Token usage information
        finish_reason: Why the generation stopped
        metadata: Additional response information
    """

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.usage.get("total_tokens", 0)


@dataclass
class Message:
    """A message in a conversation with an LLM."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}
