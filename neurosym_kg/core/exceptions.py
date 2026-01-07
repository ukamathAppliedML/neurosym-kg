"""
Custom exceptions for the NeuroSym-KG framework.

Provides a hierarchy of exceptions for different error scenarios,
enabling precise error handling.
"""

from __future__ import annotations

from typing import Any


class NeuroSymError(Exception):
    """Base exception for all NeuroSym-KG errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Knowledge Graph Exceptions
class KnowledgeGraphError(NeuroSymError):
    """Base exception for KG-related errors."""

    pass


class EntityNotFoundError(KnowledgeGraphError):
    """Raised when an entity cannot be found in the KG."""

    def __init__(self, entity_id: str, kg_name: str = "") -> None:
        message = f"Entity '{entity_id}' not found"
        if kg_name:
            message += f" in {kg_name}"
        super().__init__(message, {"entity_id": entity_id, "kg_name": kg_name})
        self.entity_id = entity_id


class RelationNotFoundError(KnowledgeGraphError):
    """Raised when a relation cannot be found in the KG."""

    def __init__(self, relation_id: str, kg_name: str = "") -> None:
        message = f"Relation '{relation_id}' not found"
        if kg_name:
            message += f" in {kg_name}"
        super().__init__(message, {"relation_id": relation_id, "kg_name": kg_name})
        self.relation_id = relation_id


class KGConnectionError(KnowledgeGraphError):
    """Raised when connection to KG backend fails."""

    def __init__(self, backend: str, reason: str = "") -> None:
        message = f"Failed to connect to {backend}"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"backend": backend, "reason": reason})


class KGQueryError(KnowledgeGraphError):
    """Raised when a KG query fails."""

    def __init__(self, query: str, reason: str = "") -> None:
        message = f"KG query failed"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"query": query[:200], "reason": reason})


class KGTimeoutError(KnowledgeGraphError):
    """Raised when a KG operation times out."""

    def __init__(self, operation: str, timeout_seconds: float) -> None:
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"
        super().__init__(message, {"operation": operation, "timeout": timeout_seconds})


# LLM Exceptions
class LLMError(NeuroSymError):
    """Base exception for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM backend fails."""

    def __init__(self, backend: str, reason: str = "") -> None:
        message = f"Failed to connect to LLM backend {backend}"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"backend": backend, "reason": reason})


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded."""

    def __init__(self, backend: str, retry_after: float | None = None) -> None:
        message = f"Rate limit exceeded for {backend}"
        if retry_after:
            message += f", retry after {retry_after}s"
        super().__init__(message, {"backend": backend, "retry_after": retry_after})
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or cannot be parsed."""

    def __init__(self, reason: str, raw_response: str = "") -> None:
        message = f"Invalid LLM response: {reason}"
        super().__init__(message, {"reason": reason, "raw_response": raw_response[:500]})


class LLMContextLengthError(LLMError):
    """Raised when input exceeds LLM context length."""

    def __init__(self, input_tokens: int, max_tokens: int) -> None:
        message = f"Input ({input_tokens} tokens) exceeds context length ({max_tokens})"
        super().__init__(message, {"input_tokens": input_tokens, "max_tokens": max_tokens})


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    def __init__(self, timeout_seconds: float) -> None:
        message = f"LLM request timed out after {timeout_seconds}s"
        super().__init__(message, {"timeout": timeout_seconds})


# Reasoning Exceptions
class ReasoningError(NeuroSymError):
    """Base exception for reasoning-related errors."""

    pass


class NoAnswerFoundError(ReasoningError):
    """Raised when reasoning cannot find an answer."""

    def __init__(self, question: str, reason: str = "") -> None:
        message = "Could not find answer to question"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"question": question[:200], "reason": reason})


class PathNotFoundError(ReasoningError):
    """Raised when no valid reasoning path is found."""

    def __init__(self, source: str, target: str = "", max_hops: int = 0) -> None:
        message = f"No path found from '{source}'"
        if target:
            message += f" to '{target}'"
        if max_hops:
            message += f" within {max_hops} hops"
        super().__init__(message, {"source": source, "target": target, "max_hops": max_hops})


class InvalidReasoningStateError(ReasoningError):
    """Raised when reasoner enters an invalid state."""

    def __init__(self, state: str, reason: str = "") -> None:
        message = f"Invalid reasoning state: {state}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"state": state, "reason": reason})


# Symbolic Reasoning Exceptions
class SymbolicError(NeuroSymError):
    """Base exception for symbolic reasoning errors."""

    pass


class RuleParseError(SymbolicError):
    """Raised when a rule cannot be parsed."""

    def __init__(self, rule: str, reason: str = "") -> None:
        message = f"Failed to parse rule"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"rule": rule[:200], "reason": reason})


class InferenceError(SymbolicError):
    """Raised when symbolic inference fails."""

    def __init__(self, reason: str = "") -> None:
        message = "Symbolic inference failed"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"reason": reason})


# Configuration Exceptions
class ConfigurationError(NeuroSymError):
    """Base exception for configuration errors."""

    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str) -> None:
        message = f"Missing required configuration: {config_key}"
        super().__init__(message, {"config_key": config_key})


class InvalidConfigError(ConfigurationError):
    """Raised when configuration value is invalid."""

    def __init__(self, config_key: str, value: Any, reason: str = "") -> None:
        message = f"Invalid configuration for '{config_key}': {value}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"config_key": config_key, "value": value, "reason": reason})


# Validation Exceptions
class ValidationError(NeuroSymError):
    """Base exception for validation errors."""

    pass


class InvalidTripleError(ValidationError):
    """Raised when a triple is invalid."""

    def __init__(self, triple_str: str, reason: str = "") -> None:
        message = f"Invalid triple: {triple_str}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"triple": triple_str, "reason": reason})


class InvalidEntityError(ValidationError):
    """Raised when an entity is invalid."""

    def __init__(self, entity_id: str, reason: str = "") -> None:
        message = f"Invalid entity: {entity_id}"
        if reason:
            message += f" ({reason})"
        super().__init__(message, {"entity_id": entity_id, "reason": reason})
