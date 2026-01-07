"""
Symbolic reasoning modules for NeuroSym-KG.

Provides tools for:
- Constraint checking and verification
- Rule-based inference
- Path validation
- Logical reasoning
"""

from neurosym_kg.symbolic.constraint_checker import (
    ConstraintChecker,
    ConstraintType,
    ConstraintViolation,
    PathValidator,
    VerificationResult,
)
from neurosym_kg.symbolic.rules import (
    INHERITANCE_RULES,
    INVERSE_RULES,
    TRANSITIVITY_RULES,
    Rule,
    RuleCondition,
    RuleEngine,
    RuleOperator,
    TriplePattern,
)

__all__ = [
    # Constraint checking
    "ConstraintChecker",
    "ConstraintType",
    "ConstraintViolation",
    "VerificationResult",
    "PathValidator",
    # Rules
    "Rule",
    "RuleCondition",
    "RuleEngine",
    "RuleOperator",
    "TriplePattern",
    # Predefined rules
    "TRANSITIVITY_RULES",
    "INHERITANCE_RULES",
    "INVERSE_RULES",
]
