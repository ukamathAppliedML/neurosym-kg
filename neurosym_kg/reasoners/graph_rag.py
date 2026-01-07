"""
GraphRAG Reasoning Engine.

Implements community-based hierarchical retrieval from:
"From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
Edge et al., Microsoft 2024

Key idea: Build a hierarchical index of entity communities with summaries,
then retrieve relevant communities for query answering.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class Community:
    """Represents a community of related entities."""

    id: str
    entities: list[str] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    summary: str = ""
    level: int = 0  # Hierarchy level
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.entities)


class GraphRAGReasoner(BaseReasoner):
    """
    GraphRAG reasoning engine with community-based retrieval.

    Process:
    1. Build: Extract entities/relations, detect communities, generate summaries
    2. Index: Create hierarchical community index
    3. Query: Retrieve relevant communities, synthesize answer

    This implementation provides a simplified version suitable for
    real-time querying without pre-building a full index.

    Example:
        >>> kg = InMemoryKG.from_triples([...])
        >>> llm = OpenAIBackend(model="gpt-4o-mini")
        >>> reasoner = GraphRAGReasoner(kg=kg, llm=llm)
        >>> result = reasoner.reason("What are the main themes in this knowledge?")
    """

    COMMUNITY_SUMMARY_PROMPT = """Summarize the key information in this group of related facts.
Focus on: main entities, their relationships, and important attributes.
Keep the summary concise (2-3 sentences).

Facts:
{facts}

Summary:"""

    RELEVANCE_RANKING_PROMPT = """Given a question and community summaries, rank which communities are most relevant.
Return community IDs in order of relevance, most relevant first.

Question: {question}

Communities:
{communities}

Relevant community IDs (comma-separated, most relevant first):"""

    ANSWER_SYNTHESIS_PROMPT = """Answer the question using information from the relevant knowledge communities below.
Synthesize information across communities when needed. Be specific and cite facts.

Question: {question}

{community_context}

Answer:"""

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        max_community_size: int = 20,
        max_communities: int = 10,
        summary_max_tokens: int = 150,
        use_hierarchical: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize GraphRAG reasoner.

        Args:
            kg: Knowledge graph backend
            llm: LLM backend
            max_community_size: Maximum entities per community
            max_communities: Maximum communities to consider
            summary_max_tokens: Max tokens for community summaries
            use_hierarchical: Use hierarchical community structure
            verbose: Print debug information
        """
        super().__init__(kg, llm, name="GraphRAG", verbose=verbose)

        self.max_community_size = max_community_size
        self.max_communities = max_communities
        self.summary_max_tokens = summary_max_tokens
        self.use_hierarchical = use_hierarchical

        # Cache for communities (would be pre-built in production)
        self._communities: dict[str, Community] = {}
        self._entity_to_community: dict[str, str] = {}

    def reason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Perform GraphRAG reasoning.

        Args:
            question: The question to answer
            context: Optional additional context
            **kwargs: Additional parameters

        Returns:
            ReasoningResult with answer and community context
        """
        start_time = time.time()
        self._stats["queries"] += 1

        self._log(f"Question: {question}")

        # Step 1: Extract seed entities
        seed_entities = self._link_entities(question)
        self._log(f"Seed entities: {[e.name for e in seed_entities]}")

        # Step 2: Build/retrieve relevant communities
        communities = self._get_relevant_communities(question, seed_entities)
        self._log(f"Found {len(communities)} relevant communities")

        if not communities:
            return self._create_no_answer_result(
                "Could not find relevant knowledge communities",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Step 3: Generate summaries for communities (if not cached)
        for community in communities:
            if not community.summary:
                community.summary = self._generate_community_summary(community)

        # Step 4: Rank communities by relevance
        ranked_communities = self._rank_communities(question, communities)

        # Step 5: Synthesize answer
        answer, raw_output = self._synthesize_answer(question, ranked_communities)

        latency_ms = (time.time() - start_time) * 1000
        self._stats["successful"] += 1
        self._stats["total_latency_ms"] += latency_ms

        # Collect all triples from communities
        all_triples = []
        for c in ranked_communities:
            all_triples.extend(c.triples)

        return ReasoningResult(
            answer=answer,
            status=ReasoningResultStatus.SUCCESS,
            subgraph=Subgraph(triples=all_triples[:100]),
            confidence=0.8,
            explanation=f"Synthesized from {len(ranked_communities)} knowledge communities",
            raw_llm_output=raw_output,
            metadata={"communities": [c.id for c in ranked_communities]},
            latency_ms=latency_ms,
        )

    def _get_relevant_communities(
        self,
        question: str,
        seed_entities: list[Entity],
    ) -> list[Community]:
        """Get or build communities relevant to the question."""
        communities: list[Community] = []

        # For each seed entity, build a local community
        for entity in seed_entities[: self.max_communities]:
            community = self._build_entity_community(entity)
            if community.size > 0:
                communities.append(community)

        return communities

    def _build_entity_community(self, entity: Entity) -> Community:
        """Build a community centered on an entity."""
        # Check cache
        if entity.id in self._entity_to_community:
            cached_id = self._entity_to_community[entity.id]
            if cached_id in self._communities:
                return self._communities[cached_id]

        # Get neighbors
        neighbors = self._kg.get_neighbors(
            entity.id,
            direction="both",
            limit=self.max_community_size * 2,
        )
        self._stats["kg_calls"] += 1

        # Collect entities and triples
        entities = {entity.id}
        triples = []

        for triple in neighbors:
            triples.append(triple)
            entities.add(triple.subject_id)
            entities.add(triple.object_id)

            if len(entities) >= self.max_community_size:
                break

        community = Community(
            id=f"community_{entity.id}",
            entities=list(entities),
            triples=triples,
            level=0,
        )

        # Cache
        self._communities[community.id] = community
        self._entity_to_community[entity.id] = community.id

        return community

    def _generate_community_summary(self, community: Community) -> str:
        """Generate a summary for a community."""
        if not community.triples:
            return ""

        facts = "\n".join(t.to_text() for t in community.triples[:30])

        prompt = self.COMMUNITY_SUMMARY_PROMPT.format(facts=facts)
        response = self._llm.generate_text(prompt, max_tokens=self.summary_max_tokens)
        self._stats["llm_calls"] += 1

        return response.strip()

    def _rank_communities(
        self,
        question: str,
        communities: list[Community],
    ) -> list[Community]:
        """Rank communities by relevance to the question."""
        if len(communities) <= 2:
            return communities

        # Format communities for ranking
        community_text = "\n\n".join(
            f"[{c.id}]\nEntities: {', '.join(c.entities[:5])}\nSummary: {c.summary}"
            for c in communities
        )

        prompt = self.RELEVANCE_RANKING_PROMPT.format(
            question=question,
            communities=community_text,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        # Parse ranking
        ranked_ids = [
            cid.strip() for cid in response.replace(" ", "").split(",")
        ]

        # Reorder communities
        id_to_community = {c.id: c for c in communities}
        ranked = []
        for cid in ranked_ids:
            if cid in id_to_community:
                ranked.append(id_to_community[cid])

        # Add any missed communities
        for c in communities:
            if c not in ranked:
                ranked.append(c)

        return ranked

    def _synthesize_answer(
        self,
        question: str,
        communities: list[Community],
    ) -> tuple[str, str]:
        """Synthesize final answer from ranked communities."""
        # Build community context
        context_parts = []
        for i, c in enumerate(communities[:5]):  # Top 5 communities
            context_parts.append(f"=== Community {i+1}: {c.id} ===")
            context_parts.append(f"Summary: {c.summary}")
            context_parts.append("Key facts:")
            for t in c.triples[:10]:
                context_parts.append(f"  - {t.to_text()}")
            context_parts.append("")

        context = "\n".join(context_parts)

        prompt = self.ANSWER_SYNTHESIS_PROMPT.format(
            question=question,
            community_context=context,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        return response.strip(), response

    def build_index(self, max_entities: int = 1000) -> None:
        """
        Pre-build community index for the entire KG.

        This is optional but improves query performance.
        In production, this would use proper community detection
        algorithms like Leiden or Louvain.
        """
        self._log("Building community index...")

        # This is a simplified implementation
        # A full implementation would:
        # 1. Load all entities
        # 2. Build entity graph
        # 3. Run Leiden/Louvain community detection
        # 4. Build hierarchical structure
        # 5. Generate and cache summaries

        self._log("Index building not fully implemented in this version")


class LocalGraphRAG(GraphRAGReasoner):
    """
    Lightweight GraphRAG for local/small KGs.

    Optimized for smaller knowledge graphs where we can
    afford to examine more of the graph per query.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kg,
            llm,
            max_community_size=30,
            max_communities=5,
            **kwargs,
        )

    def _get_relevant_communities(
        self,
        question: str,
        seed_entities: list[Entity],
    ) -> list[Community]:
        """Build overlapping communities for better coverage."""
        communities = super()._get_relevant_communities(question, seed_entities)

        # Also add 2-hop communities for more context
        for entity in seed_entities[:3]:
            neighbors = self._kg.get_neighbors(entity.id, limit=10)
            for n in neighbors[:3]:
                next_entity = Entity(id=n.object_id, name=n.object_id)
                extended = self._build_entity_community(next_entity)
                if extended.size > 2 and extended.id not in [c.id for c in communities]:
                    communities.append(extended)

        return communities[: self.max_communities]
