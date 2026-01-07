"""
Anthropic Claude LLM Backend.

Provides integration with Anthropic's Claude models for high-quality reasoning
in neuro-symbolic pipelines.

Requirements:
    pip install anthropic

Example:
    from neurosym_kg.llm_backends import AnthropicBackend
    
    llm = AnthropicBackend(
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
    )
    
    response = llm.generate([
        Message(role="user", content="What is the capital of France?")
    ])
    print(response.content)
"""

from typing import List, Optional, Dict, Any, Iterator, AsyncIterator
import os
import logging
import time

from .base import BaseLLMBackend
from ..core.types import Message, LLMResponse

logger = logging.getLogger(__name__)


class AnthropicBackend(BaseLLMBackend):
    """
    Anthropic Claude LLM Backend.
    
    Supports all Claude models including:
    - claude-sonnet-4-20250514 (recommended for reasoning)
    - claude-opus-4-20250514 (highest capability)
    - claude-haiku-3-5-20241022 (fastest)
    
    Features:
    - Synchronous and asynchronous generation
    - Streaming support
    - System prompts
    - Temperature and sampling control
    - Token usage tracking
    - Automatic retries with exponential backoff
    """
    
    # Model context windows
    MODEL_CONTEXT_WINDOWS = {
        "claude-sonnet-4-20250514": 200000,
        "claude-opus-4-20250514": 200000,
        "claude-haiku-3-5-20241022": 200000,
        # Legacy models
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize Anthropic backend.
        
        Args:
            model: Claude model to use
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            base_url: Optional custom API base URL
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            system_prompt: Default system prompt for all requests
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        super().__init__(model=model)
        
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Get API key
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or pass api_key parameter."
            )
        
        # Try to import anthropic
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
        
        # Initialize clients
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self._client = self._anthropic.Anthropic(**client_kwargs)
        self._async_client = self._anthropic.AsyncAnthropic(**client_kwargs)
        
        # Usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
        logger.info(f"Initialized Anthropic backend with model: {model}")
    
    @property
    def context_window(self) -> int:
        """Get context window size for current model."""
        return self.MODEL_CONTEXT_WINDOWS.get(self._model, 200000)
    
    @property
    def total_tokens_used(self) -> Dict[str, int]:
        """Get total token usage."""
        return {
            "input": self._total_input_tokens,
            "output": self._total_output_tokens,
            "total": self._total_input_tokens + self._total_output_tokens,
        }
    
    def _convert_messages(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Convert internal Message format to Anthropic format.
        
        Extracts system message (if any) and formats user/assistant messages.
        
        Returns:
            Tuple of (system_prompt, messages_list)
        """
        system = self.system_prompt
        converted = []
        
        for msg in messages:
            if msg.role == "system":
                # Anthropic uses system as a separate parameter
                system = msg.content
            else:
                converted.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        # Ensure messages alternate user/assistant
        # Claude requires first message to be from user
        if converted and converted[0]["role"] != "user":
            converted.insert(0, {"role": "user", "content": "Hello"})
        
        return system, converted
    
    def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response synchronously.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop_sequences: Optional stop sequences
            **kwargs: Additional parameters passed to API
            
        Returns:
            LLMResponse with generated content
        """
        self._stats["calls"] += 1
        start_time = time.time()
        
        system, converted_messages = self._convert_messages(messages)
        
        try:
            response = self._client.messages.create(
                model=self._model,
                messages=converted_messages,
                system=system or self._anthropic.NOT_GIVEN,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stop_sequences=stop_sequences or self._anthropic.NOT_GIVEN,
                **kwargs,
            )
            
            # Track usage
            if hasattr(response, "usage"):
                self._total_input_tokens += response.usage.input_tokens
                self._total_output_tokens += response.usage.output_tokens
            
            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text
            
            latency = (time.time() - start_time) * 1000
            self._stats["latency_total_ms"] += latency
            
            return LLMResponse(
                content=content,
                model=self._model,
                usage={
                    "prompt_tokens": response.usage.input_tokens if hasattr(response, "usage") else 0,
                    "completion_tokens": response.usage.output_tokens if hasattr(response, "usage") else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, "usage") else 0,
                },
                finish_reason=response.stop_reason if hasattr(response, "stop_reason") else "stop",
                metadata={"latency_ms": latency},
            )
            
        except self._anthropic.APIError as e:
            self._stats["errors"] += 1
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def agenerate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response asynchronously.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop_sequences: Optional stop sequences
            **kwargs: Additional parameters passed to API
            
        Returns:
            LLMResponse with generated content
        """
        self._stats["calls"] += 1
        start_time = time.time()
        
        system, converted_messages = self._convert_messages(messages)
        
        try:
            response = await self._async_client.messages.create(
                model=self._model,
                messages=converted_messages,
                system=system or self._anthropic.NOT_GIVEN,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stop_sequences=stop_sequences or self._anthropic.NOT_GIVEN,
                **kwargs,
            )
            
            # Track usage
            if hasattr(response, "usage"):
                self._total_input_tokens += response.usage.input_tokens
                self._total_output_tokens += response.usage.output_tokens
            
            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text
            
            latency = (time.time() - start_time) * 1000
            self._stats["latency_total_ms"] += latency
            
            return LLMResponse(
                content=content,
                model=self._model,
                usage={
                    "prompt_tokens": response.usage.input_tokens if hasattr(response, "usage") else 0,
                    "completion_tokens": response.usage.output_tokens if hasattr(response, "usage") else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, "usage") else 0,
                },
                finish_reason=response.stop_reason if hasattr(response, "stop_reason") else "stop",
                metadata={"latency_ms": latency},
            )
            
        except self._anthropic.APIError as e:
            self._stats["errors"] += 1
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Generate a streaming response.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters
            
        Yields:
            Content chunks as they're generated
        """
        system, converted_messages = self._convert_messages(messages)
        
        with self._client.messages.stream(
            model=self._model,
            messages=converted_messages,
            system=system or self._anthropic.NOT_GIVEN,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    async def agenerate_stream(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response asynchronously.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters
            
        Yields:
            Content chunks as they're generated
        """
        system, converted_messages = self._convert_messages(messages)
        
        async with self._async_client.messages.stream(
            model=self._model,
            messages=converted_messages,
            system=system or self._anthropic.NOT_GIVEN,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Note: This is an approximation. For exact counts, use the API.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 characters per token for English
        # Claude's tokenizer is similar to GPT's
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self._model,
            "provider": "anthropic",
            "context_window": self.context_window,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "total_tokens_used": self.total_tokens_used,
        }


class ClaudeBackend(AnthropicBackend):
    """
    Alias for AnthropicBackend.
    
    Provides a more intuitive name for users.
    """
    pass
