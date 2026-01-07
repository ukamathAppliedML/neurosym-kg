"""
Base class for LLM backend implementations.

Provides common functionality for all LLM backends.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator

from tenacity import retry, stop_after_attempt, wait_exponential

from neurosym_kg.core.config import get_config
from neurosym_kg.core.types import LLMResponse, Message


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    Provides common utilities and defines the interface that all
    LLM implementations must follow.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries

        self._stats: dict[str, Any] = {
            "calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "errors": 0,
            "latency_total_ms": 0.0,
        }

    @property
    def model_name(self) -> str:
        """The name/identifier of the model."""
        return self._model

    @property
    def max_tokens(self) -> int:
        """Maximum tokens for generation."""
        return self._default_max_tokens

    @property
    def stats(self) -> dict[str, Any]:
        """Usage statistics."""
        stats = self._stats.copy()
        if stats["calls"] > 0:
            stats["avg_latency_ms"] = stats["latency_total_ms"] / stats["calls"]
        return stats

    def _update_stats(self, response: LLMResponse, latency_ms: float) -> None:
        """Update usage statistics."""
        self._stats["calls"] += 1
        self._stats["total_tokens"] += response.usage.get("total_tokens", 0)
        self._stats["prompt_tokens"] += response.usage.get("prompt_tokens", 0)
        self._stats["completion_tokens"] += response.usage.get("completion_tokens", 0)
        self._stats["latency_total_ms"] += latency_ms

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to dictionaries."""
        return [m.to_dict() for m in messages]

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    async def agenerate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of generate."""
        ...

    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience method for simple text generation."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        response = self.generate(messages, **kwargs)
        return response.content

    async def agenerate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Async convenience method for simple text generation."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))

        response = await self.agenerate(messages, **kwargs)
        return response.content


class BaseStreamingLLMBackend(BaseLLMBackend):
    """
    Base class for LLM backends that support streaming.
    """

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response tokens."""
        ...

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream response tokens."""
        ...
