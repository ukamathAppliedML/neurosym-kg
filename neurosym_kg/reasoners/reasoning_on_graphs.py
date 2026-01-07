"""
Reasoning on Graphs (RoG) Engine.

Implements faithful planning-based reasoning from:
"Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning"
Luo et al., ICLR 2024

Key idea: Generate relation-level reasoning plans, retrieve paths following
those plans, then reason over the grounded evidence.
"""

from __future__ import annotations

import re
import time
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
class ReasoningPlan:
    """A reasoning plan consisting of relation sequences."""

    relations: list[str]
    score: float = 1.0
    description: str = ""

    def __str__(self) -> str:
        return " -> ".join(self.relations)


class ReasoningOnGraphs(BaseReasoner):
    """
    Reasoning on Graphs (RoG) engine.

    Three-stage process:
    1. Planning: LLM generates relation-level reasoning paths
    2. Retrieval: Follow plans on KG to retrieve evidence
    3. Reasoning: LLM reasons over retrieved paths

    This approach provides:
    - Faithful reasoning grounded in KG structure
    - Interpretable reasoning chains
    - Reduced hallucination through explicit grounding

    Example:
        >>> kg = WikidataKG()
        >>> llm = OpenAIBackend(model="gpt-4o-mini")
        >>> reasoner = ReasoningOnGraphs(kg=kg, llm=llm)
        >>> result = reasoner.reason("Who is the president of the country where the Eiffel Tower is located?")
    """

    PLAN_GENERATION_PROMPT = """Generate reasoning paths to answer this question using a knowledge graph.
Each path should be a sequence of relations that connects the question entities to the answer.

Question: {question}
Topic Entity: {topic_entity}

Available relations in the KG include general ones like: born_in, located_in, capital_of, president_of, spouse_of, director_of, founded_by, part_of, instance_of, etc.

Generate {num_plans} different reasoning paths. Format each path as:
Path N: relation1 -> relation2 -> relation3

Reasoning Paths:"""

    PATH_SCORING_PROMPT = """Score how likely each reasoning path is to lead to the correct answer (0-10).

Question: {question}
Topic Entity: {topic_entity}

Paths:
{paths}

For each path, provide: Path N: score
Scores:"""

    ANSWER_FROM_PATHS_PROMPT = """Answer the question using ONLY the evidence from the knowledge graph paths below.
If the paths don't contain enough information, say "Cannot determine."

Question: {question}

Evidence Paths:
{evidence}

Based on these paths, the answer is:"""

    FAITHFUL_REASONING_PROMPT = """You are a faithful reasoning assistant. Answer the question by following the reasoning paths through the knowledge graph.

Question: {question}

Step-by-step reasoning using the knowledge graph:
{reasoning_steps}

Therefore, the answer is:"""

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMBackend,
        num_plans: int = 3,
        max_path_length: int = 4,
        max_paths_per_plan: int = 5,
        use_beam_search: bool = True,
        faithful_mode: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize RoG reasoner.

        Args:
            kg: Knowledge graph backend
            llm: LLM backend
            num_plans: Number of reasoning plans to generate
            max_path_length: Maximum relations in a plan
            max_paths_per_plan: Max concrete paths per plan
            use_beam_search: Use beam search for path finding
            faithful_mode: Enforce faithful reasoning from paths
            verbose: Print debug information
        """
        super().__init__(kg, llm, name="ReasoningOnGraphs", verbose=verbose)

        self.num_plans = num_plans
        self.max_path_length = max_path_length
        self.max_paths_per_plan = max_paths_per_plan
        self.use_beam_search = use_beam_search
        self.faithful_mode = faithful_mode

    def reason(
        self,
        question: str,
        context: str | None = None,
        **kwargs: Any,
    ) -> ReasoningResult:
        """
        Perform RoG reasoning.

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

        # Step 1: Extract topic entity
        topic_entities = self._link_entities(question)
        if not topic_entities:
            return self._create_no_answer_result(
                "Could not identify topic entity in question",
                latency_ms=(time.time() - start_time) * 1000,
            )

        topic_entity = topic_entities[0]
        self._log(f"Topic entity: {topic_entity.name}")

        # Step 2: Generate reasoning plans
        plans = self._generate_plans(question, topic_entity)
        self._log(f"Generated {len(plans)} reasoning plans")

        if not plans:
            return self._create_no_answer_result(
                "Could not generate valid reasoning plans",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Step 3: Retrieve paths following plans
        all_paths: list[ReasoningPath] = []
        for plan in plans:
            paths = self._retrieve_paths(topic_entity, plan)
            all_paths.extend(paths)
            self._log(f"Plan '{plan}' yielded {len(paths)} paths")

        if not all_paths:
            return self._create_no_answer_result(
                "Could not find valid paths in the knowledge graph",
                paths=[],
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Step 4: Generate answer from paths
        if self.faithful_mode:
            answer, raw_output = self._faithful_reasoning(question, all_paths)
        else:
            answer, raw_output = self._generate_answer(question, all_paths)

        latency_ms = (time.time() - start_time) * 1000
        self._stats["successful"] += 1
        self._stats["total_latency_ms"] += latency_ms

        # Collect all triples
        all_triples = []
        for path in all_paths:
            all_triples.extend(path.triples)

        return ReasoningResult(
            answer=answer,
            status=ReasoningResultStatus.SUCCESS,
            paths=all_paths,
            subgraph=Subgraph(triples=all_triples),
            confidence=max(p.score for p in all_paths) if all_paths else 0.5,
            explanation=f"Reasoned over {len(all_paths)} paths from {len(plans)} plans",
            raw_llm_output=raw_output,
            metadata={"plans": [str(p) for p in plans]},
            latency_ms=latency_ms,
        )

    def _generate_plans(
        self,
        question: str,
        topic_entity: Entity,
    ) -> list[ReasoningPlan]:
        """Generate reasoning plans using the LLM."""
        prompt = self.PLAN_GENERATION_PROMPT.format(
            question=question,
            topic_entity=topic_entity.name,
            num_plans=self.num_plans,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        # Parse plans
        plans: list[ReasoningPlan] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Match "Path N: rel1 -> rel2 -> rel3"
            match = re.match(r"Path\s*\d*:?\s*(.+)", line, re.IGNORECASE)
            if match:
                path_str = match.group(1)
                # Split by arrow variations
                relations = re.split(r"\s*[-→>]+\s*", path_str)
                relations = [r.strip().lower().replace(" ", "_") for r in relations if r.strip()]

                if relations and len(relations) <= self.max_path_length:
                    plans.append(ReasoningPlan(relations=relations))

        # Score plans
        if len(plans) > 1:
            plans = self._score_plans(question, topic_entity, plans)

        return plans[: self.num_plans]

    def _score_plans(
        self,
        question: str,
        topic_entity: Entity,
        plans: list[ReasoningPlan],
    ) -> list[ReasoningPlan]:
        """Score and rank reasoning plans."""
        paths_text = "\n".join(
            f"Path {i+1}: {' -> '.join(p.relations)}"
            for i, p in enumerate(plans)
        )

        prompt = self.PATH_SCORING_PROMPT.format(
            question=question,
            topic_entity=topic_entity.name,
            paths=paths_text,
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        # Parse scores
        for line in response.strip().split("\n"):
            match = re.match(r"Path\s*(\d+):?\s*(\d+(?:\.\d+)?)", line)
            if match:
                idx = int(match.group(1)) - 1
                score = float(match.group(2)) / 10.0
                if 0 <= idx < len(plans):
                    plans[idx].score = score

        # Sort by score
        plans.sort(key=lambda p: p.score, reverse=True)
        return plans

    def _retrieve_paths(
        self,
        topic_entity: Entity,
        plan: ReasoningPlan,
    ) -> list[ReasoningPath]:
        """Retrieve concrete paths following a reasoning plan."""
        paths: list[ReasoningPath] = []

        # BFS/beam search following the plan
        current_entities = [(topic_entity.id, [])]  # (entity_id, path_so_far)

        for relation in plan.relations:
            next_entities: list[tuple[str, list[Triple]]] = []

            for entity_id, path_so_far in current_entities:
                # Get neighbors with matching relation
                neighbors = self._kg.get_neighbors(
                    entity_id,
                    direction="outgoing",
                    limit=50,
                )
                self._stats["kg_calls"] += 1

                for triple in neighbors:
                    # Flexible relation matching
                    pred = triple.predicate_id.lower().replace(" ", "_")
                    if self._relation_matches(relation, pred):
                        new_path = path_so_far + [triple]
                        next_entities.append((triple.object_id, new_path))

                        if len(next_entities) >= self.max_paths_per_plan * 3:
                            break

                if len(next_entities) >= self.max_paths_per_plan * 3:
                    break

            # Prune to beam width
            current_entities = next_entities[: self.max_paths_per_plan * 2]

            if not current_entities:
                break

        # Convert to ReasoningPaths
        for entity_id, triple_path in current_entities:
            if triple_path:
                paths.append(
                    ReasoningPath(
                        triples=triple_path,
                        score=plan.score,
                        source_entity=topic_entity.id,
                        target_entity=entity_id,
                        metadata={"plan": str(plan)},
                    )
                )

        return paths[: self.max_paths_per_plan]

    def _relation_matches(self, plan_relation: str, kg_relation: str) -> bool:
        """Check if a KG relation matches a plan relation (fuzzy matching)."""
        plan_rel = plan_relation.lower().replace("_", "").replace(" ", "")
        kg_rel = kg_relation.lower().replace("_", "").replace(" ", "")

        # Exact match
        if plan_rel == kg_rel:
            return True

        # Substring match
        if plan_rel in kg_rel or kg_rel in plan_rel:
            return True

        # Common synonyms
        synonyms = {
            "locatedin": ["country", "location", "place", "in"],
            "bornin": ["birthplace", "placeofbirth"],
            "presidentof": ["president", "leader", "headof"],
            "capitalof": ["capital"],
            "spouseof": ["spouse", "marriedto", "wife", "husband"],
            "directorof": ["director", "directedby"],
        }

        for key, syns in synonyms.items():
            if plan_rel in [key] + syns and kg_rel in [key] + syns:
                return True

        return False

    def _generate_answer(
        self,
        question: str,
        paths: list[ReasoningPath],
    ) -> tuple[str, str]:
        """Generate answer from retrieved paths."""
        evidence = []
        for i, path in enumerate(paths[:10]):
            evidence.append(f"Path {i+1}: {path.to_text()}")

        prompt = self.ANSWER_FROM_PATHS_PROMPT.format(
            question=question,
            evidence="\n".join(evidence),
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        return response.strip(), response

    def _faithful_reasoning(
        self,
        question: str,
        paths: list[ReasoningPath],
    ) -> tuple[str, str]:
        """Generate answer with explicit step-by-step reasoning."""
        # Build reasoning steps from paths
        steps = []
        for i, path in enumerate(paths[:5]):
            step_parts = []
            for j, triple in enumerate(path.triples):
                step_parts.append(
                    f"  Step {j+1}: From {triple.subject_id}, "
                    f"following '{triple.predicate_id}', we reach {triple.object_id}"
                )
            steps.append(f"Reasoning Path {i+1}:\n" + "\n".join(step_parts))

        prompt = self.FAITHFUL_REASONING_PROMPT.format(
            question=question,
            reasoning_steps="\n\n".join(steps),
        )
        response = self._llm.generate_text(prompt)
        self._stats["llm_calls"] += 1

        return response.strip(), response
