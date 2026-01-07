"""
Base class for reasoning engine implementations.

Provides common functionality for all reasoning paradigms.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from neurosym_kg.core.interfaces import KnowledgeGraph, LLMBackend
from neurosym_kg.core.types import (
    Entity,
    Message,
    ReasoningPath,
    ReasoningResult,
    ReasoningResultStatus,
    Subgraph,
    Triple,
)


class BaseReasoner(ABC):
    """
    Abstract base class for reasoning engines.

    Provides common utilities and defines the interface that all
    reasoners must implement.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        name: str = "BaseReasoner",
        verbose: bool = False,
    ) -> None:
        self._kg = kg
        self._llm = llm
        self._name = name
        self._verbose = verbose

        self._stats: dict[str, Any] = {
            "queries": 0,
            "kg_calls": 0,
            "llm_calls": 0,
            "successful": 0,
            "failed": 0,
            "total_latency_ms": 0.0,
        }

    @property
    def name(self) -> str:
        """Name of the reasoning paradigm."""
        return self._name

    @property
    def kg(self) -> KnowledgeGraph:
        """The knowledge graph backend."""
        return self._kg

    @property
    def llm(self) -> LLMBackend:
        """The LLM backend."""
        return self._llm

    @property
    def stats(self) -> dict[str, Any]:
        """Reasoning statistics."""
        stats = self._stats.copy()
        if stats["queries"] > 0:
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["queries"]
            stats["success_rate"] = stats["successful"] / stats["queries"]
        return stats

    def _log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self._verbose:
            print(f"[{self._name}] {message}")

    def _extract_entities(self, question: str) -> list[Entity]:
        """
        Extract potential entities from a question using the LLM.

        This is a simple implementation that can be overridden.
        """
        prompt = f"""Extract the key entities (people, places, things, concepts) from this question.
Return only the entity names, one per line, without any explanation.

Question: {question}

Entities:"""

        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        entities = []
        for line in response.strip().split("\n"):
            name = line.strip().strip("-").strip("•").strip()
            if name:
                # Try to find in KG
                matches = self._kg.get_entity_by_name(name, limit=1)
                if matches:
                    entities.append(matches[0])
                else:
                    # Create placeholder entity
                    entities.append(Entity(id=name.replace(" ", "_"), name=name))

        return entities

    def _link_entities(self, question: str) -> list[Entity]:
        """
        Link entity mentions in question to KG entities.

        Uses LLM to extract mentions, then searches KG for matches.
        """
        return self._extract_entities(question)

    def _triples_to_text(
        self,
        triples: list[Triple],
        max_triples: int = 50,
    ) -> str:
        """Convert triples to text for LLM prompt."""
        lines = []
        for t in triples[:max_triples]:
            lines.append(t.to_text())
        if len(triples) > max_triples:
            lines.append(f"... and {len(triples) - max_triples} more facts")
        return "\n".join(lines)

    def _create_error_result(
        self,
        reason: str,
        latency_ms: float = 0.0,
    ) -> ReasoningResult:
        """Create an error result."""
        self._stats["failed"] += 1
        return ReasoningResult(
            answer="",
            status=ReasoningResultStatus.ERROR,
            explanation=reason,
            latency_ms=latency_ms,
        )

    def _create_no_answer_result(
        self,
        reason: str,
        paths: list[ReasoningPath] | None = None,
        subgraph: Subgraph | None = None,
        latency_ms: float = 0.0,
    ) -> ReasoningResult:
        """Create a no-answer result."""
        self._stats["failed"] += 1
        return ReasoningResult(
            answer="",
            status=ReasoningResultStatus.NO_ANSWER,
            paths=paths or [],
            subgraph=subgraph,
            explanation=reason,
            latency_ms=latency_ms,
        )

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

    async def areason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Async version of reason.

        Default implementation runs sync version.
        Override for true async support.
        """
        return self.reason(question, context, **kwargs)

    def __call__(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """Allow calling reasoner directly."""
        return self.reason(question, context, **kwargs)
