"""
Symbolic rule representation and matching.

Provides a simple rule engine for logical inference over KG facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

from neurosym_kg.core.types import Triple


class RuleOperator(str, Enum):
    """Operators for rule conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex
    IN = "in"
    EXISTS = "exists"


@dataclass
class RuleCondition:
    """A condition in a rule antecedent."""

    variable: str  # e.g., "?x", "?rel", "$subject"
    operator: RuleOperator
    value: Any
    negate: bool = False

    def evaluate(self, bindings: dict[str, Any]) -> bool:
        """Evaluate this condition against variable bindings."""
        if self.variable not in bindings:
            result = self.operator == RuleOperator.EXISTS and not self.negate
            return result

        bound_value = bindings[self.variable]

        if self.operator == RuleOperator.EQUALS:
            result = bound_value == self.value
        elif self.operator == RuleOperator.NOT_EQUALS:
            result = bound_value != self.value
        elif self.operator == RuleOperator.CONTAINS:
            result = self.value in str(bound_value)
        elif self.operator == RuleOperator.STARTS_WITH:
            result = str(bound_value).startswith(self.value)
        elif self.operator == RuleOperator.ENDS_WITH:
            result = str(bound_value).endswith(self.value)
        elif self.operator == RuleOperator.MATCHES:
            result = bool(re.match(self.value, str(bound_value)))
        elif self.operator == RuleOperator.IN:
            result = bound_value in self.value
        elif self.operator == RuleOperator.EXISTS:
            result = True
        else:
            result = False

        return not result if self.negate else result


@dataclass
class TriplePattern:
    """
    A pattern that matches triples.

    Variables start with ? (e.g., ?x, ?person)
    Constants are literal values
    """

    subject: str  # Variable like ?x or constant
    predicate: str
    object: str

    def is_variable(self, term: str) -> bool:
        """Check if a term is a variable."""
        return term.startswith("?")

    def match(self, triple: Triple) -> dict[str, str] | None:
        """
        Try to match this pattern against a triple.

        Returns variable bindings if match, None otherwise.
        """
        bindings: dict[str, str] = {}

        # Match subject
        if self.is_variable(self.subject):
            bindings[self.subject] = triple.subject_id
        elif self.subject != triple.subject_id:
            return None

        # Match predicate
        if self.is_variable(self.predicate):
            bindings[self.predicate] = triple.predicate_id
        elif self.predicate != triple.predicate_id:
            return None

        # Match object
        if self.is_variable(self.object):
            bindings[self.object] = triple.object_id
        elif self.object != triple.object_id:
            return None

        return bindings

    def to_text(self) -> str:
        """Convert to readable text."""
        return f"({self.subject}, {self.predicate}, {self.object})"


@dataclass
class Rule:
    """
    A logical rule with antecedent patterns and consequent.

    Example:
        IF (?x, parent_of, ?y) AND (?y, parent_of, ?z)
        THEN (?x, grandparent_of, ?z)
    """

    name: str
    antecedent: list[TriplePattern]
    consequent: TriplePattern
    conditions: list[RuleCondition] = field(default_factory=list)
    priority: int = 0
    description: str = ""

    def to_text(self) -> str:
        """Convert to readable text."""
        ant_str = " AND ".join(p.to_text() for p in self.antecedent)
        return f"IF {ant_str} THEN {self.consequent.to_text()}"


class RuleEngine:
    """
    Simple forward-chaining rule engine.

    Features:
    - Pattern matching over triples
    - Variable binding and unification
    - Forward chaining inference
    - Rule priority ordering

    Example:
        >>> engine = RuleEngine()
        >>> engine.add_rule(Rule(
        ...     name="grandparent",
        ...     antecedent=[
        ...         TriplePattern("?x", "parent_of", "?y"),
        ...         TriplePattern("?y", "parent_of", "?z"),
        ...     ],
        ...     consequent=TriplePattern("?x", "grandparent_of", "?z"),
        ... ))
        >>> facts = [
        ...     Triple("Alice", "parent_of", "Bob"),
        ...     Triple("Bob", "parent_of", "Charlie"),
        ... ]
        >>> inferred = engine.forward_chain(facts)
    """

    def __init__(self) -> None:
        self.rules: list[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules.append(rule)
        # Keep sorted by priority
        self.rules.sort(key=lambda r: -r.priority)

    def add_rules(self, rules: list[Rule]) -> None:
        """Add multiple rules."""
        for rule in rules:
            self.add_rule(rule)

    def clear_rules(self) -> None:
        """Remove all rules."""
        self.rules.clear()

    def _find_bindings(
        self,
        pattern: TriplePattern,
        facts: list[Triple],
        existing_bindings: dict[str, str] | None = None,
    ) -> Iterator[dict[str, str]]:
        """Find all variable bindings that satisfy a pattern."""
        existing = existing_bindings or {}

        for triple in facts:
            bindings = pattern.match(triple)
            if bindings is None:
                continue

            # Check consistency with existing bindings
            consistent = True
            merged = existing.copy()

            for var, val in bindings.items():
                if var in merged:
                    if merged[var] != val:
                        consistent = False
                        break
                else:
                    merged[var] = val

            if consistent:
                yield merged

    def _match_antecedent(
        self,
        patterns: list[TriplePattern],
        facts: list[Triple],
        bindings: dict[str, str] | None = None,
    ) -> Iterator[dict[str, str]]:
        """Find all bindings that satisfy all antecedent patterns."""
        if not patterns:
            yield bindings or {}
            return

        first_pattern = patterns[0]
        rest_patterns = patterns[1:]

        for binding in self._find_bindings(first_pattern, facts, bindings):
            yield from self._match_antecedent(rest_patterns, facts, binding)

    def _apply_bindings(
        self,
        pattern: TriplePattern,
        bindings: dict[str, str],
    ) -> Triple | None:
        """Create a triple by applying bindings to a pattern."""
        try:
            subject = bindings.get(pattern.subject, pattern.subject)
            predicate = bindings.get(pattern.predicate, pattern.predicate)
            obj = bindings.get(pattern.object, pattern.object)

            # Check all variables are bound
            if any(v.startswith("?") for v in [subject, predicate, obj]):
                return None

            return Triple(subject=subject, predicate=predicate, object=obj)
        except Exception:
            return None

    def apply_rule(
        self,
        rule: Rule,
        facts: list[Triple],
    ) -> list[Triple]:
        """Apply a single rule to derive new facts."""
        new_facts = []

        for bindings in self._match_antecedent(rule.antecedent, facts):
            # Check additional conditions
            conditions_met = all(c.evaluate(bindings) for c in rule.conditions)

            if conditions_met:
                new_triple = self._apply_bindings(rule.consequent, bindings)
                if new_triple and new_triple not in facts:
                    new_facts.append(new_triple)

        return new_facts

    def forward_chain(
        self,
        initial_facts: list[Triple],
        max_iterations: int = 100,
    ) -> list[Triple]:
        """
        Perform forward chaining to derive all possible facts.

        Args:
            initial_facts: Starting facts
            max_iterations: Maximum inference iterations

        Returns:
            List of all inferred facts (not including initial facts)
        """
        all_facts = list(initial_facts)
        all_inferred: list[Triple] = []

        for _ in range(max_iterations):
            new_facts: list[Triple] = []

            for rule in self.rules:
                inferred = self.apply_rule(rule, all_facts)
                for fact in inferred:
                    if fact not in all_facts and fact not in new_facts:
                        new_facts.append(fact)

            if not new_facts:
                break

            all_facts.extend(new_facts)
            all_inferred.extend(new_facts)

        return all_inferred

    def query(
        self,
        pattern: TriplePattern,
        facts: list[Triple],
    ) -> list[dict[str, str]]:
        """
        Query facts using a pattern.

        Returns all variable bindings that satisfy the pattern.
        """
        return list(self._find_bindings(pattern, facts))


# Predefined common rules
TRANSITIVITY_RULES = [
    Rule(
        name="subclass_transitivity",
        antecedent=[
            TriplePattern("?x", "subclass_of", "?y"),
            TriplePattern("?y", "subclass_of", "?z"),
        ],
        consequent=TriplePattern("?x", "subclass_of", "?z"),
        description="Subclass relation is transitive",
    ),
    Rule(
        name="part_of_transitivity",
        antecedent=[
            TriplePattern("?x", "part_of", "?y"),
            TriplePattern("?y", "part_of", "?z"),
        ],
        consequent=TriplePattern("?x", "part_of", "?z"),
        description="Part-of relation is transitive",
    ),
    Rule(
        name="located_in_transitivity",
        antecedent=[
            TriplePattern("?x", "located_in", "?y"),
            TriplePattern("?y", "located_in", "?z"),
        ],
        consequent=TriplePattern("?x", "located_in", "?z"),
        description="Located-in relation is transitive",
    ),
]

INHERITANCE_RULES = [
    Rule(
        name="instance_inheritance",
        antecedent=[
            TriplePattern("?x", "instance_of", "?class"),
            TriplePattern("?class", "subclass_of", "?superclass"),
        ],
        consequent=TriplePattern("?x", "instance_of", "?superclass"),
        description="Instances inherit from superclasses",
    ),
]

INVERSE_RULES = [
    Rule(
        name="parent_child_inverse",
        antecedent=[TriplePattern("?x", "parent_of", "?y")],
        consequent=TriplePattern("?y", "child_of", "?x"),
        description="Parent and child are inverse relations",
    ),
    Rule(
        name="spouse_symmetric",
        antecedent=[TriplePattern("?x", "spouse", "?y")],
        consequent=TriplePattern("?y", "spouse", "?x"),
        description="Spouse relation is symmetric",
    ),
]
