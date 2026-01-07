"""
OpenAI LLM backend implementation.

Supports GPT-4, GPT-3.5, and other OpenAI models.
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Iterator

from neurosym_kg.core.config import get_config
from neurosym_kg.core.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from neurosym_kg.core.types import LLMResponse, Message
from neurosym_kg.llm_backends.base import BaseStreamingLLMBackend


class OpenAIBackend(BaseStreamingLLMBackend):
    """
    OpenAI API backend.

    Supports:
    - GPT-4o, GPT-4o-mini
    - GPT-4, GPT-4-turbo
    - GPT-3.5-turbo

    Example:
        >>> llm = OpenAIBackend(model="gpt-4o-mini")
        >>> response = llm.generate([Message(role="user", content="Hello!")])
        >>> print(response.content)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        config = get_config()

        model = model or config.llm.openai_default_model
        super().__init__(model, temperature, max_tokens, timeout, max_retries)

        self._api_key = api_key or config.llm.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url or config.llm.openai_base_url

        if not self._api_key:
            raise LLMConnectionError("OpenAI", "API key not provided")

        # Lazy import to avoid requiring openai package
        try:
            import openai

            self._client = openai.OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._async_client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=timeout,
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIBackend. "
                "Install with: pip install neurosym-kg[openai]"
            )

    def generate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from OpenAI."""
        import openai

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._prepare_messages(messages),
                temperature=temperature if temperature is not None else self._default_temperature,
                max_tokens=max_tokens or self._default_max_tokens,
                stop=stop_sequences,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            result = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason or "stop",
            )

            self._update_stats(result, latency_ms)
            return result

        except openai.RateLimitError as e:
            self._stats["errors"] += 1
            raise LLMRateLimitError("OpenAI")

        except openai.APITimeoutError:
            self._stats["errors"] += 1
            raise LLMTimeoutError(self._timeout)

        except openai.APIConnectionError as e:
            self._stats["errors"] += 1
            raise LLMConnectionError("OpenAI", str(e))

        except openai.APIError as e:
            self._stats["errors"] += 1
            raise LLMResponseError(str(e))

    async def agenerate(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async generate a response from OpenAI."""
        import openai

        start_time = time.time()

        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=self._prepare_messages(messages),
                temperature=temperature if temperature is not None else self._default_temperature,
                max_tokens=max_tokens or self._default_max_tokens,
                stop=stop_sequences,
                **kwargs,
            )

            latency_ms = (time.time() - start_time) * 1000

            result = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason or "stop",
            )

            self._update_stats(result, latency_ms)
            return result

        except openai.RateLimitError:
            self._stats["errors"] += 1
            raise LLMRateLimitError("OpenAI")

        except openai.APITimeoutError:
            self._stats["errors"] += 1
            raise LLMTimeoutError(self._timeout)

        except openai.APIConnectionError as e:
            self._stats["errors"] += 1
            raise LLMConnectionError("OpenAI", str(e))

        except openai.APIError as e:
            self._stats["errors"] += 1
            raise LLMResponseError(str(e))

    def stream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream response tokens from OpenAI."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._prepare_messages(messages),
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens or self._default_max_tokens,
            stream=True,
            **kwargs,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def astream(
        self,
        messages: list[Message],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream response tokens from OpenAI."""
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=self._prepare_messages(messages),
            temperature=temperature if temperature is not None else self._default_temperature,
            max_tokens=max_tokens or self._default_max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
