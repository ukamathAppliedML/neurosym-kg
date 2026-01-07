"""
Symbolic constraint checking and verification.

Provides tools for verifying LLM outputs against KG facts
and enforcing logical constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from neurosym_kg.core.interfaces import KnowledgeGraph
from neurosym_kg.core.types import Entity, ReasoningResult, Triple


class ConstraintType(str, Enum):
    """Types of constraints that can be checked."""

    ENTITY_EXISTS = "entity_exists"
    RELATION_EXISTS = "relation_exists"
    TRIPLE_EXISTS = "triple_exists"
    PATH_VALID = "path_valid"
    TYPE_CONSTRAINT = "type_constraint"
    CARDINALITY = "cardinality"
    CUSTOM = "custom"


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""

    constraint_type: ConstraintType
    message: str
    severity: str = "warning"  # "warning", "error"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Result of verifying an answer against constraints."""

    is_valid: bool
    violations: list[ConstraintViolation] = field(default_factory=list)
    verified_facts: list[Triple] = field(default_factory=list)
    confidence_adjustment: float = 0.0
    explanation: str = ""

    @property
    def has_errors(self) -> bool:
        """Check if there are error-level violations."""
        return any(v.severity == "error" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if there are warning-level violations."""
        return any(v.severity == "warning" for v in self.violations)


class ConstraintChecker:
    """
    Verifies LLM outputs against knowledge graph constraints.

    Features:
    - Entity existence verification
    - Relation validity checking
    - Path verification
    - Type constraint enforcement
    - Custom constraint rules

    Example:
        >>> checker = ConstraintChecker(kg)
        >>> result = checker.verify_answer(
        ...     answer="Paris",
        ...     question="What is the capital of France?",
        ...     reasoning_paths=paths
        ... )
        >>> if result.is_valid:
        ...     print("Answer verified!")
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        strict_mode: bool = False,
    ) -> None:
        """
        Initialize the constraint checker.

        Args:
            kg: Knowledge graph for verification
            strict_mode: If True, warnings become errors
        """
        self.kg = kg
        self.strict_mode = strict_mode
        self._custom_constraints: list[Callable[[str, str, list[Triple]], list[ConstraintViolation]]] = []

    def add_custom_constraint(
        self,
        constraint_fn: Callable[[str, str, list[Triple]], list[ConstraintViolation]],
    ) -> None:
        """
        Add a custom constraint function.

        Args:
            constraint_fn: Function taking (answer, question, paths) and returning violations
        """
        self._custom_constraints.append(constraint_fn)

    def verify_answer(
        self,
        answer: str,
        question: str,
        reasoning_paths: list[list[Triple]] | None = None,
        expected_type: str | None = None,
    ) -> VerificationResult:
        """
        Verify an answer against KG constraints.

        Args:
            answer: The answer to verify
            question: The original question
            reasoning_paths: Reasoning paths that led to the answer
            expected_type: Expected entity type of the answer

        Returns:
            VerificationResult with validation status and details
        """
        violations: list[ConstraintViolation] = []
        verified_facts: list[Triple] = []

        # 1. Check if answer entity exists in KG
        entity_violations = self._check_entity_exists(answer)
        violations.extend(entity_violations)

        # 2. Verify reasoning paths
        if reasoning_paths:
            path_violations, verified = self._verify_paths(reasoning_paths)
            violations.extend(path_violations)
            verified_facts.extend(verified)

        # 3. Check type constraints
        if expected_type:
            type_violations = self._check_type_constraint(answer, expected_type)
            violations.extend(type_violations)

        # 4. Run custom constraints
        for constraint_fn in self._custom_constraints:
            custom_violations = constraint_fn(
                answer,
                question,
                [t for path in (reasoning_paths or []) for t in path],
            )
            violations.extend(custom_violations)

        # Apply strict mode
        if self.strict_mode:
            for v in violations:
                if v.severity == "warning":
                    v.severity = "error"

        # Calculate result
        is_valid = not any(v.severity == "error" for v in violations)
        confidence_adjustment = -0.1 * len([v for v in violations if v.severity == "error"])
        confidence_adjustment -= 0.05 * len([v for v in violations if v.severity == "warning"])

        explanation_parts = []
        if verified_facts:
            explanation_parts.append(f"Verified {len(verified_facts)} facts in KG")
        if violations:
            explanation_parts.append(f"Found {len(violations)} constraint issues")

        return VerificationResult(
            is_valid=is_valid,
            violations=violations,
            verified_facts=verified_facts,
            confidence_adjustment=confidence_adjustment,
            explanation="; ".join(explanation_parts) if explanation_parts else "No constraints checked",
        )

    def _check_entity_exists(self, answer: str) -> list[ConstraintViolation]:
        """Check if the answer entity exists in the KG."""
        violations = []

        # Try to find entity
        matches = self.kg.get_entity_by_name(answer, limit=5)

        if not matches:
            # Try with normalized form
            normalized = answer.strip().replace(" ", "_")
            entity = self.kg.get_entity(normalized)

            if not entity:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.ENTITY_EXISTS,
                        message=f"Answer entity '{answer}' not found in knowledge graph",
                        severity="warning",
                        details={"answer": answer},
                    )
                )

        return violations

    def _verify_paths(
        self,
        paths: list[list[Triple]],
    ) -> tuple[list[ConstraintViolation], list[Triple]]:
        """Verify that reasoning paths exist in the KG."""
        violations = []
        verified = []

        for path_idx, path in enumerate(paths):
            for triple_idx, triple in enumerate(path):
                # Check if triple exists
                existing = self.kg.get_triples(
                    subject=triple.subject_id,
                    predicate=triple.predicate_id,
                    obj=triple.object_id,
                    limit=1,
                )

                if existing:
                    verified.append(triple)
                else:
                    # Triple not found - check if it's approximately correct
                    partial_match = self.kg.get_triples(
                        subject=triple.subject_id,
                        predicate=triple.predicate_id,
                        limit=10,
                    )

                    if partial_match:
                        violations.append(
                            ConstraintViolation(
                                constraint_type=ConstraintType.TRIPLE_EXISTS,
                                message=f"Triple object mismatch in path {path_idx + 1}",
                                severity="warning",
                                details={
                                    "claimed": triple.to_text(),
                                    "found_objects": [t.object_id for t in partial_match[:3]],
                                },
                            )
                        )
                    else:
                        violations.append(
                            ConstraintViolation(
                                constraint_type=ConstraintType.TRIPLE_EXISTS,
                                message=f"Triple not found in KG: {triple.to_text()}",
                                severity="error",
                                details={"triple": triple.to_text(), "path_index": path_idx},
                            )
                        )

        return violations, verified

    def _check_type_constraint(
        self,
        answer: str,
        expected_type: str,
    ) -> list[ConstraintViolation]:
        """Check if the answer matches the expected type."""
        violations = []

        # Find the answer entity
        matches = self.kg.get_entity_by_name(answer, limit=1)
        if not matches:
            return violations  # Can't check type if entity not found

        entity = matches[0]

        # Check instance_of / type relations
        type_triples = self.kg.get_triples(
            subject=entity.id,
            predicate="P31",  # Wikidata instance_of
            limit=10,
        )

        if not type_triples:
            # Try other type predicates
            type_triples = self.kg.get_neighbors(
                entity.id,
                direction="outgoing",
                relation_filter=["type", "instance_of", "rdf:type"],
                limit=10,
            )

        # Check if any type matches
        type_names = [t.object_id.lower() for t in type_triples]
        expected_lower = expected_type.lower()

        if not any(expected_lower in tn or tn in expected_lower for tn in type_names):
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.TYPE_CONSTRAINT,
                    message=f"Answer '{answer}' may not be of expected type '{expected_type}'",
                    severity="warning",
                    details={
                        "answer": answer,
                        "expected_type": expected_type,
                        "found_types": type_names[:5],
                    },
                )
            )

        return violations

    def verify_reasoning_result(
        self,
        result: ReasoningResult,
        question: str,
        expected_type: str | None = None,
    ) -> tuple[ReasoningResult, VerificationResult]:
        """
        Verify a complete reasoning result.

        Returns the result with adjusted confidence and the verification details.
        """
        paths = [[t for t in path.triples] for path in result.paths]

        verification = self.verify_answer(
            answer=result.primary_answer,
            question=question,
            reasoning_paths=paths if paths else None,
            expected_type=expected_type,
        )

        # Adjust confidence based on verification
        new_confidence = max(0.0, min(1.0, result.confidence + verification.confidence_adjustment))

        # Create updated result
        updated_result = ReasoningResult(
            answer=result.answer,
            status=result.status,
            paths=result.paths,
            subgraph=result.subgraph,
            confidence=new_confidence,
            explanation=f"{result.explanation}; Verification: {verification.explanation}",
            raw_llm_output=result.raw_llm_output,
            metadata={**result.metadata, "verification": verification},
            latency_ms=result.latency_ms,
        )

        return updated_result, verification


class PathValidator:
    """
    Validates reasoning paths for logical consistency.

    Checks:
    - Path connectivity (each step connects to previous)
    - Relation validity
    - Cycle detection
    - Semantic coherence
    """

    def __init__(self, kg: KnowledgeGraph) -> None:
        self.kg = kg

    def validate_path(self, path: list[Triple]) -> tuple[bool, list[str]]:
        """
        Validate a reasoning path.

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        if not path:
            return True, []

        # Check connectivity
        for i in range(1, len(path)):
            prev_obj = path[i - 1].object_id
            curr_subj = path[i].subject_id

            if prev_obj != curr_subj:
                issues.append(
                    f"Path disconnected at step {i}: {prev_obj} != {curr_subj}"
                )

        # Check for cycles
        visited = set()
        for triple in path:
            if triple.subject_id in visited:
                issues.append(f"Cycle detected: revisited entity {triple.subject_id}")
            visited.add(triple.subject_id)

        return len(issues) == 0, issues

    def find_valid_paths(
        self,
        paths: list[list[Triple]],
    ) -> list[list[Triple]]:
        """Filter to only valid paths."""
        valid = []
        for path in paths:
            is_valid, _ = self.validate_path(path)
            if is_valid:
                valid.append(path)
        return valid
