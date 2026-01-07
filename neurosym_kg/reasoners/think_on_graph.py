"""
Think-on-Graph (ToG) Reasoning Engine.

Implements the ToG paradigm from:
"Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph"
Sun et al., ICLR 2024

The LLM acts as an agent that iteratively explores the KG using beam search,
pruning unpromising paths, and reasoning until sufficient information is gathered.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from neurosym_kg.core.interfaces import KnowledgeGraph, LLMBackend
from neurosym_kg.core.types import (
    Entity,
    Message,
    ReasoningPath,
    ReasoningResult,
    ReasoningResultStatus,
    Triple,
)
from neurosym_kg.reasoners.base import BaseReasoner


@dataclass
class ToGSearchState:
    """State of a beam search path in ToG."""

    entity_id: str
    entity_name: str
    path: list[Triple] = field(default_factory=list)
    score: float = 1.0
    visited_entities: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.visited_entities.add(self.entity_id)


class ThinkOnGraphReasoner(BaseReasoner):
    """
    Think-on-Graph (ToG) reasoning engine.

    The LLM iteratively explores the KG:
    1. Initialize: Extract seed entities from question
    2. Explore: Use LLM to select promising relations/neighbors
    3. Prune: Score and keep top-K candidate paths
    4. Reason: Check if enough info to answer, else iterate
    5. Generate: Produce answer from best reasoning paths

    Example:
        >>> kg = WikidataKG()
        >>> llm = OpenAIBackend(model="gpt-4o-mini")
        >>> reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm, max_depth=3, beam_width=5)
        >>> result = reasoner.reason("Who directed Inception?")
        >>> print(result.answer)
    """

    # Prompt templates
    ENTITY_EXTRACTION_PROMPT = """Extract the key entities from this question that should be looked up in a knowledge graph.
Return only entity names, one per line.

Question: {question}

Entities:"""

    RELATION_SELECTION_PROMPT = """Given a question and an entity with its relations, select the most relevant relations to explore.
Return only the relation names that would help answer the question, one per line.

Question: {question}
Current Entity: {entity}
Available Relations:
{relations}

Most relevant relations (up to {max_relations}):"""

    ENTITY_SELECTION_PROMPT = """Given a question and candidate entities reached via a relation, select the most promising entities to continue exploring.
Return only the entity names that are most likely to help answer the question, one per line.

Question: {question}
Current path: {path}
Relation: {relation}
Candidate Entities:
{entities}

Most promising entities (up to {max_entities}):"""

    REASONING_CHECK_PROMPT = """Given a question and the knowledge gathered from a knowledge graph, determine if we have enough information to answer the question.

Question: {question}

Knowledge gathered:
{knowledge}

Can this question be answered with the gathered knowledge? Respond with only "YES" or "NO", followed by a brief explanation."""

    ANSWER_GENERATION_PROMPT = """Based on the following knowledge from a knowledge graph, answer the question.
Provide a concise, factual answer.

Question: {question}

Relevant Knowledge:
{knowledge}

Answer:"""

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        max_depth: int = 3,
        beam_width: int = 5,
        max_relations_per_hop: int = 10,
        max_entities_per_relation: int = 5,
        pruning_threshold: float = 0.3,
        early_stopping: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the ToG reasoner.

        Args:
            kg: Knowledge graph backend
            llm: LLM backend
            max_depth: Maximum number of hops in the KG
            beam_width: Number of paths to maintain in beam search
            max_relations_per_hop: Max relations to consider per hop
            max_entities_per_relation: Max entities to expand per relation
            pruning_threshold: Minimum score to keep a path
            early_stopping: Stop when answer is found
            verbose: Print debug information
        """
        super().__init__(kg, llm, name="ThinkOnGraph", verbose=verbose)

        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_relations_per_hop = max_relations_per_hop
        self.max_entities_per_relation = max_entities_per_relation
        self.pruning_threshold = pruning_threshold
        self.early_stopping = early_stopping

    def reason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Perform ToG reasoning to answer a question.

        Args:
            question: The question to answer
            context: Optional additional context
            **kwargs: Additional parameters

        Returns:
            ReasoningResult with answer and reasoning paths
        """
        start_time = time.time()
        self._stats["queries"] += 1

        self._log(f"Question: {question}")

        # Step 1: Extract seed entities
        seed_entities = self._extract_seed_entities(question)
        if not seed_entities:
            return self._create_no_answer_result(
                "Could not identify any entities in the question",
                latency_ms=(time.time() - start_time) * 1000,
            )

        self._log(f"Seed entities: {[e.name for e in seed_entities]}")

        # Initialize beam with seed entities
        beam: list[ToGSearchState] = []
        for entity in seed_entities[: self.beam_width]:
            beam.append(
                ToGSearchState(
                    entity_id=entity.id,
                    entity_name=entity.name,
                )
            )

        all_paths: list[ReasoningPath] = []
        collected_knowledge: list[Triple] = []

        # Step 2-4: Iterative exploration
        for depth in range(self.max_depth):
            self._log(f"Depth {depth + 1}/{self.max_depth}, beam size: {len(beam)}")

            if not beam:
                break

            # Expand each state in the beam
            new_candidates: list[ToGSearchState] = []

            for state in beam:
                expanded = self._expand_state(question, state)
                new_candidates.extend(expanded)

                # Collect knowledge from this state's path
                for triple in state.path:
                    if triple not in collected_knowledge:
                        collected_knowledge.append(triple)

            # Prune and select top-K
            new_candidates.sort(key=lambda s: s.score, reverse=True)
            beam = new_candidates[: self.beam_width]

            # Filter by threshold
            beam = [s for s in beam if s.score >= self.pruning_threshold]

            # Check if we have enough information (early stopping)
            if self.early_stopping and collected_knowledge:
                has_answer = self._check_sufficient_knowledge(question, collected_knowledge)
                if has_answer:
                    self._log("Early stopping: sufficient knowledge found")
                    break

        # Collect final paths
        for state in beam:
            if state.path:
                all_paths.append(
                    ReasoningPath(
                        triples=state.path,
                        score=state.score,
                        source_entity=state.path[0].subject_id if state.path else None,
                    )
                )
                # Add remaining knowledge
                for triple in state.path:
                    if triple not in collected_knowledge:
                        collected_knowledge.append(triple)

        # Step 5: Generate answer
        if not collected_knowledge:
            return self._create_no_answer_result(
                "Could not find relevant knowledge in the graph",
                paths=all_paths,
                latency_ms=(time.time() - start_time) * 1000,
            )

        answer, raw_output = self._generate_answer(question, collected_knowledge)
        latency_ms = (time.time() - start_time) * 1000

        self._stats["successful"] += 1
        self._stats["total_latency_ms"] += latency_ms

        return ReasoningResult(
            answer=answer,
            status=ReasoningResultStatus.SUCCESS,
            paths=all_paths,
            confidence=max(p.score for p in all_paths) if all_paths else 0.5,
            explanation=f"Found {len(collected_knowledge)} relevant facts via {len(all_paths)} reasoning paths",
            raw_llm_output=raw_output,
            latency_ms=latency_ms,
        )

    def _extract_seed_entities(self, question: str) -> list[Entity]:
        """Extract seed entities from the question."""
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(question=question)
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        entities: list[Entity] = []
        seen_ids: set[str] = set()

        for line in response.strip().split("\n"):
            name = line.strip().strip("-").strip("•").strip()
            if not name:
                continue

            # Search in KG
            matches = self._kg.get_entity_by_name(name, limit=3)
            self._stats["kg_calls"] += 1

            for match in matches:
                if match.id not in seen_ids:
                    entities.append(match)
                    seen_ids.add(match.id)

        return entities

    def _expand_state(
        self,
        question: str,
        state: ToGSearchState,
    ) -> list[ToGSearchState]:
        """Expand a search state by exploring its neighbors."""
        new_states: list[ToGSearchState] = []

        # Get available relations
        neighbors = self._kg.get_neighbors(
            state.entity_id,
            direction="outgoing",
            limit=50,
        )
        self._stats["kg_calls"] += 1

        if not neighbors:
            return []

        # Group by relation
        relations: dict[str, list[Triple]] = {}
        for triple in neighbors:
            rel = triple.predicate_id
            if rel not in relations:
                relations[rel] = []
            relations[rel].append(triple)

        # Select most relevant relations using LLM
        relation_names = "\n".join(
            f"- {rel}: connects to {len(triples)} entities"
            for rel, triples in list(relations.items())[:20]
        )

        prompt = self.RELATION_SELECTION_PROMPT.format(
            question=question,
            entity=state.entity_name,
            relations=relation_names,
            max_relations=self.max_relations_per_hop,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        selected_relations = set()
        for line in response.strip().split("\n"):
            rel = line.strip().strip("-").strip("•").strip()
            if rel:
                # Match to actual relation IDs
                for rel_id in relations:
                    if rel.lower() in rel_id.lower() or rel_id.lower() in rel.lower():
                        selected_relations.add(rel_id)

        if not selected_relations:
            # Fallback: use first few relations
            selected_relations = set(list(relations.keys())[: self.max_relations_per_hop])

        # For each selected relation, pick entities to expand
        for rel_id in selected_relations:
            rel_triples = relations.get(rel_id, [])
            if not rel_triples:
                continue

            # Select entities
            selected_triples = self._select_entities(
                question,
                state,
                rel_id,
                rel_triples,
            )

            # Create new states
            for triple in selected_triples:
                next_entity_id = triple.object_id
                if next_entity_id in state.visited_entities:
                    continue

                # Calculate path score (simple heuristic)
                new_score = state.score * 0.9  # Decay with depth

                new_state = ToGSearchState(
                    entity_id=next_entity_id,
                    entity_name=str(triple.object),
                    path=state.path + [triple],
                    score=new_score,
                    visited_entities=state.visited_entities | {next_entity_id},
                )
                new_states.append(new_state)

        return new_states

    def _select_entities(
        self,
        question: str,
        state: ToGSearchState,
        relation_id: str,
        triples: list[Triple],
    ) -> list[Triple]:
        """Select which entities to expand for a given relation."""
        if len(triples) <= self.max_entities_per_relation:
            return triples

        # Use LLM to select
        entities_list = "\n".join(
            f"- {t.object_id}" for t in triples[: 20]
        )

        path_str = " -> ".join(t.to_text() for t in state.path) if state.path else state.entity_name

        prompt = self.ENTITY_SELECTION_PROMPT.format(
            question=question,
            path=path_str,
            relation=relation_id,
            entities=entities_list,
            max_entities=self.max_entities_per_relation,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        selected_names = set()
        for line in response.strip().split("\n"):
            name = line.strip().strip("-").strip("•").strip()
            if name:
                selected_names.add(name.lower())

        # Match to triples
        selected = []
        for triple in triples:
            obj_id = triple.object_id.lower()
            obj_name = str(triple.object).lower()
            if obj_id in selected_names or obj_name in selected_names:
                selected.append(triple)
            elif any(s in obj_id or s in obj_name for s in selected_names):
                selected.append(triple)

        return selected[: self.max_entities_per_relation] or triples[: self.max_entities_per_relation]

    def _check_sufficient_knowledge(
        self,
        question: str,
        knowledge: list[Triple],
    ) -> bool:
        """Check if we have enough knowledge to answer the question."""
        knowledge_text = self._triples_to_text(knowledge, max_triples=30)

        prompt = self.REASONING_CHECK_PROMPT.format(
            question=question,
            knowledge=knowledge_text,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        return response.strip().upper().startswith("YES")

    def _generate_answer(
        self,
        question: str,
        knowledge: list[Triple],
    ) -> tuple[str, str]:
        """Generate the final answer from collected knowledge."""
        knowledge_text = self._triples_to_text(knowledge, max_triples=50)

        prompt = self.ANSWER_GENERATION_PROMPT.format(
            question=question,
            knowledge=knowledge_text,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        return response.strip(), response
