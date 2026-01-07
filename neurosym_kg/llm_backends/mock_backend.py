"""
Mock LLM backend for testing.

Provides configurable responses for unit tests without API calls.
"""

from __future__ import annotations

import re
import time
from typing import Any, AsyncIterator, Callable, Iterator

from neurosym_kg.core.types import LLMResponse, Message
from neurosym_kg.llm_backends.base import BaseStreamingLLMBackend


class MockLLMBackend(BaseStreamingLLMBackend):
    """
    Mock LLM backend for testing.

    Features:
    - Configurable default responses
    - Pattern-based response matching
    - Latency simulation
    - Call history tracking

    Example:
        >>> llm = MockLLMBackend(default_response="I am a mock LLM.")
        >>> llm.add_response(r".*capital.*France.*", "Paris")
        >>> response = llm.generate([Message(role="user", content="What is the capital of France?")])
        >>> print(response.content)  # "Paris"
    """

    def __init__(
        self,
        model: str = "mock-model",
        default_response: str = "This is a mock response.",
        simulate_latency_ms: float = 0.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout=60.0)

        self._default_response = default_response
        self._simulate_latency_ms = simulate_latency_ms
        self._call_history: list[dict[str, Any]] = []

        # Pattern -> response mapping
        self._response_patterns: list[tuple[re.Pattern, str | Callable[[str], str]]] = []

        # Queue of responses (FIFO)
        self._response_queue: list[str] = []

    @property
    def call_history(self) -> list[dict[str, Any]]:
        """History of all generate calls."""
        return self._call_history.copy()

    @property
    def call_count(self) -> int:
        """Number of generate calls made."""
        return len(self._call_history)

    def add_response(
        self,
        pattern: str | re.Pattern,
        response: str | Callable[[str], str],
    ) -> None:
        """
        Add a pattern-based response.

        Args:
            pattern: Regex pattern to match against user messages
            response: Response string or callable that takes the message and returns response
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern, re.IGNORECASE)
        self._response_patterns.append((pattern, response))

    def queue_response(self, response: str) -> None:
        """Add a response to the queue (will be used in order)."""
        self._response_queue.append(response)

    def queue_responses(self, responses: list[str]) -> None:
        """Add multiple responses to the queue."""
        self._response_queue.extend(responses)

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    def clear_patterns(self) -> None:
        """Clear response patterns."""
        self._response_patterns.clear()

    def clear_queue(self) -> None:
        """Clear response queue."""
        self._response_queue.clear()

    def reset(self) -> None:
        """Reset all state."""
        self.clear_history()
        self.clear_patterns()
        self.clear_queue()

    def _get_response(self, messages: list[Message]) -> str:
        """Determine the response based on messages."""
        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        # Check queue first
        if self._response_queue:
            return self._response_queue.pop(0)

        # Check patterns
        for pattern, response in self._response_patterns:
            if pattern.search(user_message):
                if callable(response):
                    return response(user_message)
                return response

        return self._default_response

    def _simulate_delay(self) -> None:
        """Simulate API latency."""
        if self._simulate_latency_ms > 0:
            time.sleep(self._simulate_latency_ms / 1000.0)

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock response."""
        start_time = time.time()
        self._simulate_delay()

        response_text = self._get_response(messages)

        # Simulate token counts
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        completion_tokens = len(response_text.split())

        latency_ms = (time.time() - start_time) * 1000

        result = LLMResponse(
            content=response_text,
            model=self._model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
        )

        # Record call
        self._call_history.append(
            {
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response": response_text,
                "kwargs": kwargs,
            }
        )

        self._update_stats(result, latency_ms)
        return result

    async def agenerate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async generate (same as sync for mock)."""
        return self.generate(messages, temperature, max_tokens, stop_sequences, **kwargs)

    def stream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream mock response tokens."""
        response = self.generate(messages, temperature, max_tokens, **kwargs)
        # Yield word by word
        for word in response.content.split():
            yield word + " "
            if self._simulate_latency_ms > 0:
                time.sleep(self._simulate_latency_ms / 1000.0 / 10)

    async def astream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream mock response tokens."""
        response = await self.agenerate(messages, temperature, max_tokens, **kwargs)
        for word in response.content.split():
            yield word + " "


class ReplayLLMBackend(MockLLMBackend):
    """
    LLM backend that replays recorded responses.

    Useful for deterministic testing with real response data.

    Example:
        >>> llm = ReplayLLMBackend()
        >>> llm.load_responses("test_responses.json")
        >>> # Responses will be played back in order
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._replay_index = 0
        self._recorded_responses: list[dict[str, Any]] = []

    def load_responses(self, responses: list[dict[str, Any]]) -> None:
        """Load responses for replay."""
        self._recorded_responses = responses
        self._replay_index = 0

    def _get_response(self, messages: list[Message]) -> str:
        """Get next response from recording."""
        if self._replay_index < len(self._recorded_responses):
            response = self._recorded_responses[self._replay_index]
            self._replay_index += 1
            return response.get("content", self._default_response)
        return self._default_response

    def reset_replay(self) -> None:
        """Reset replay index to beginning."""
        self._replay_index = 0
