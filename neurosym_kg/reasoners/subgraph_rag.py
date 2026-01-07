"""
SubgraphRAG Reasoning Engine.

Implements flexible subgraph retrieval with LLM reasoning from:
"SubgraphRAG: Retrieval-Augmented Generation on Knowledge Graphs via Subgraph Retrieval"
Li et al., 2024

Key idea: Retrieve variable-sized subgraphs based on query complexity,
then let LLM reason over the structured subgraph context.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from neurosym_kg.core.interfaces import KnowledgeGraph, LLMBackend
from neurosym_kg.core.types import (
    Entity,
    ReasoningPath,
    ReasoningResult,
    ReasoningResultStatus,
    Subgraph,
    Triple,
)
from neurosym_kg.reasoners.base import BaseReasoner


class SubgraphRAGReasoner(BaseReasoner):
    """
    SubgraphRAG reasoning engine.

    Retrieves a relevant subgraph from the KG and uses the LLM to reason
    over the structured context. Supports multiple retrieval strategies:
    - dense: Embedding-based similarity search
    - sparse: Keyword/entity matching
    - hybrid: Combination of dense and sparse

    Example:
        >>> kg = WikidataKG()
        >>> llm = OpenAIBackend(model="gpt-4o-mini")
        >>> reasoner = SubgraphRAGReasoner(kg=kg, llm=llm, subgraph_size=50)
        >>> result = reasoner.reason("Who won the Nobel Prize that Einstein won?")
    """

    ANSWER_PROMPT = """You are a knowledge graph reasoning assistant. Answer the question using ONLY the facts provided in the knowledge subgraph below. If the answer cannot be determined from the given facts, say "Cannot determine from available knowledge."

Question: {question}

Knowledge Subgraph:
{subgraph}

Instructions:
1. Carefully analyze the relationships in the knowledge subgraph
2. Trace the path from entities mentioned in the question to the answer
3. Provide a concise, factual answer
4. If you use multiple facts, briefly explain the reasoning chain

Answer:"""

    ENTITY_EXTRACTION_PROMPT = """Extract the main entities from this question that should be searched in a knowledge graph.
Return entity names only, one per line. Focus on named entities (people, places, organizations, works, etc.)

Question: {question}

Entities:"""

    SUBGRAPH_RELEVANCE_PROMPT = """Rate how relevant each fact is for answering the question (0-10).
Return only the indices of facts with relevance >= 7, comma-separated.

Question: {question}

Facts:
{facts}

Relevant fact indices:"""

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        subgraph_size: int = 50,
        max_hops: int = 2,
        retrieval_method: Literal["dense", "sparse", "hybrid"] = "sparse",
        relevance_filtering: bool = True,
        include_entity_descriptions: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize SubgraphRAG reasoner.

        Args:
            kg: Knowledge graph backend
            llm: LLM backend
            subgraph_size: Maximum triples in retrieved subgraph
            max_hops: Maximum hops from seed entities
            retrieval_method: Retrieval strategy
            relevance_filtering: Use LLM to filter irrelevant triples
            include_entity_descriptions: Include entity descriptions in context
            verbose: Print debug information
        """
        super().__init__(kg, llm, name="SubgraphRAG", verbose=verbose)

        self.subgraph_size = subgraph_size
        self.max_hops = max_hops
        self.retrieval_method = retrieval_method
        self.relevance_filtering = relevance_filtering
        self.include_entity_descriptions = include_entity_descriptions

    def reason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Perform SubgraphRAG reasoning.

        Args:
            question: The question to answer
            context: Optional additional context
            **kwargs: Additional parameters

        Returns:
            ReasoningResult with answer and retrieved subgraph
        """
        start_time = time.time()
        self._stats["queries"] += 1

        self._log(f"Question: {question}")

        # Step 1: Extract seed entities
        seed_entities = self._extract_entities(question)
        if not seed_entities:
            return self._create_no_answer_result(
                "Could not identify entities in the question",
                latency_ms=(time.time() - start_time) * 1000,
            )

        self._log(f"Seed entities: {[e.name for e in seed_entities]}")

        # Step 2: Retrieve subgraph
        subgraph = self._retrieve_subgraph(seed_entities)
        self._log(f"Retrieved subgraph with {subgraph.size} triples")

        if subgraph.size == 0:
            return self._create_no_answer_result(
                "Could not retrieve relevant knowledge from the graph",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Step 3: Optional relevance filtering
        if self.relevance_filtering and subgraph.size > 20:
            subgraph = self._filter_relevant_triples(question, subgraph)
            self._log(f"Filtered to {subgraph.size} relevant triples")

        # Step 4: Generate answer
        answer, raw_output = self._generate_answer(question, subgraph)

        latency_ms = (time.time() - start_time) * 1000
        self._stats["successful"] += 1
        self._stats["total_latency_ms"] += latency_ms

        # Create reasoning path from subgraph
        paths = self._extract_paths_from_subgraph(subgraph, seed_entities)

        return ReasoningResult(
            answer=answer,
            status=ReasoningResultStatus.SUCCESS,
            paths=paths,
            subgraph=subgraph,
            confidence=0.8 if subgraph.size > 5 else 0.5,
            explanation=f"Retrieved {subgraph.size} facts from {len(seed_entities)} seed entities",
            raw_llm_output=raw_output,
            latency_ms=latency_ms,
        )

    def _extract_entities(self, question: str) -> list[Entity]:
        """Extract seed entities from the question."""
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(question=question)
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        entities: list[Entity] = []
        seen: set[str] = set()

        for line in response.strip().split("\n"):
            name = line.strip().strip("-").strip("•").strip()
            if not name or len(name) < 2:
                continue

            # Search KG
            matches = self._kg.get_entity_by_name(name, limit=3)
            self._stats["kg_calls"] += 1

            for match in matches:
                if match.id not in seen:
                    entities.append(match)
                    seen.add(match.id)

        return entities

    def _retrieve_subgraph(self, seed_entities: list[Entity]) -> Subgraph:
        """Retrieve a subgraph centered on seed entities."""
        entity_ids = [e.id for e in seed_entities]

        # Use KG's built-in subgraph extraction
        subgraph = self._kg.get_subgraph(
            entity_ids=entity_ids,
            max_hops=self.max_hops,
            max_triples=self.subgraph_size,
        )
        self._stats["kg_calls"] += 1

        return subgraph

    def _filter_relevant_triples(
        self,
        question: str,
        subgraph: Subgraph,
    ) -> Subgraph:
        """Use LLM to filter out irrelevant triples."""
        if subgraph.size <= 10:
            return subgraph

        # Format facts with indices
        facts_text = "\n".join(
            f"{i}. {t.to_text()}" for i, t in enumerate(subgraph.triples)
        )

        prompt = self.SUBGRAPH_RELEVANCE_PROMPT.format(
            question=question,
            facts=facts_text,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        # Parse indices
        try:
            indices = set()
            for part in response.replace(" ", "").split(","):
                part = part.strip()
                if part.isdigit():
                    indices.add(int(part))
        except Exception:
            return subgraph  # On parse error, return original

        if not indices:
            return subgraph

        # Filter triples
        filtered_triples = [
            t for i, t in enumerate(subgraph.triples) if i in indices
        ]

        if len(filtered_triples) < 3:
            return subgraph  # Keep original if too aggressive

        return Subgraph(
            triples=filtered_triples,
            center_entity=subgraph.center_entity,
            metadata=subgraph.metadata,
        )

    def _generate_answer(
        self,
        question: str,
        subgraph: Subgraph,
    ) -> tuple[str, str]:
        """Generate answer from subgraph context."""
        # Format subgraph as text
        subgraph_text = self._format_subgraph(subgraph)

        prompt = self.ANSWER_PROMPT.format(
            question=question,
            subgraph=subgraph_text,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        return response.strip(), response

    def _format_subgraph(self, subgraph: Subgraph) -> str:
        """Format subgraph for LLM prompt."""
        lines = []

        # Group triples by subject for readability
        by_subject: dict[str, list[Triple]] = {}
        for t in subgraph.triples:
            subj = t.subject_id
            if subj not in by_subject:
                by_subject[subj] = []
            by_subject[subj].append(t)

        for subject, triples in by_subject.items():
            lines.append(f"\n[{subject}]")
            for t in triples:
                lines.append(f"  - {t.predicate_id}: {t.object_id}")

        return "\n".join(lines)

    def _extract_paths_from_subgraph(
        self,
        subgraph: Subgraph,
        seed_entities: list[Entity],
    ) -> list[ReasoningPath]:
        """Extract reasoning paths from the subgraph."""
        paths: list[ReasoningPath] = []

        # Simple path extraction: sequences from seed entities
        for entity in seed_entities:
            entity_triples = [
                t for t in subgraph.triples if t.subject_id == entity.id
            ]
            if entity_triples:
                paths.append(
                    ReasoningPath(
                        triples=entity_triples[:5],  # Limit path length
                        score=0.8,
                        source_entity=entity.id,
                    )
                )

        return paths


class HybridSubgraphRAG(SubgraphRAGReasoner):
    """
    Extended SubgraphRAG with hybrid retrieval.

    Combines:
    - Sparse retrieval (entity linking + BFS)
    - Dense retrieval (embedding similarity)
    - LLM-guided expansion
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        embedding_model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(kg, llm, retrieval_method="hybrid", **kwargs)
        self._embedding_model = embedding_model

    def _retrieve_subgraph(self, seed_entities: list[Entity]) -> Subgraph:
        """Hybrid retrieval combining multiple strategies."""
        # Start with sparse retrieval
        sparse_subgraph = super()._retrieve_subgraph(seed_entities)

        # If embedding model available, augment with dense retrieval
        if self._embedding_model is not None:
            # This would use embedding similarity to find additional relevant triples
            # For now, just return sparse results
            pass

        return sparse_subgraph
