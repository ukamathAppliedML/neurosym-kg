"""
Ollama LLM Backend.

Provides integration with Ollama for local model inference.
Supports all models available in Ollama including Llama, Mistral, Qwen, etc.

Requirements:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama: ollama serve
    3. Pull a model: ollama pull llama3.2

Example:
    from neurosym_kg.llm_backends import OllamaBackend
    
    llm = OllamaBackend(model="llama3.2")
    response = llm.generate([
        Message(role="user", content="What is the capital of France?")
    ])
    print(response.content)
"""

from typing import List, Optional, Dict, Any, Iterator
import time
import logging

import httpx

from .base import BaseLLMBackend
from ..core.types import Message, LLMResponse

logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLMBackend):
    """
    Ollama LLM Backend for local model inference.
    
    Supports all Ollama models including:
    - llama3.2, llama3.1, llama2
    - mistral, mistral-nemo
    - qwen2.5-coder, qwen2.5
    - codellama, deepseek-coder
    - phi3, gemma2
    - and many more
    
    Features:
    - Synchronous and asynchronous generation
    - Streaming support
    - System prompts
    - Temperature and sampling control
    - Local inference (data never leaves your machine)
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 120.0,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize Ollama backend.
        
        Args:
            model: Ollama model name (e.g., "llama3.2", "qwen2.5-coder:7b")
            base_url: Ollama server URL
            temperature: Sampling temperature (0.0 = deterministic)
            timeout: Request timeout in seconds
            system_prompt: Default system prompt for all requests
        """
        super().__init__(model=model)
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.system_prompt = system_prompt
        
        # Verify connection
        self._verify_connection()
        
        logger.info(f"Initialized Ollama backend with model: {model}")
    
    def _verify_connection(self) -> None:
        """Verify Ollama server is running."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            logger.warning(f"Could not verify Ollama connection: {e}")
    
    def _convert_messages(
        self,
        messages: List[Message],
    ) -> List[Dict[str, str]]:
        """Convert internal Message format to Ollama format."""
        converted = []
        
        # Add system prompt if configured and not already in messages
        has_system = any(m.role == "system" for m in messages)
        if self.system_prompt and not has_system:
            converted.append({"role": "system", "content": self.system_prompt})
        
        for msg in messages:
            converted.append({
                "role": msg.role,
                "content": msg.content,
            })
        
        return converted
    
    def generate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response synchronously.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate (num_predict in Ollama)
            **kwargs: Additional parameters passed to Ollama API
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        ollama_messages = self._convert_messages(messages)
        
        # Build options
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": options,
                    **kwargs,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            
            latency = (time.time() - start_time) * 1000
            
            # Extract token counts if available
            usage = {}
            if "prompt_eval_count" in data:
                usage["prompt_tokens"] = data["prompt_eval_count"]
            if "eval_count" in data:
                usage["completion_tokens"] = data["eval_count"]
            if "prompt_eval_count" in data and "eval_count" in data:
                usage["total_tokens"] = data["prompt_eval_count"] + data["eval_count"]
            
            result = LLMResponse(
                content=data["message"]["content"],
                model=self._model,
                usage=usage,
                finish_reason=data.get("done_reason") or "stop",
                metadata={"latency_ms": latency},
            )
            
            # Update stats using base class method
            self._update_stats(result, latency)
            
            return result
            
        except httpx.ConnectError:
            self._stats["errors"] += 1
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Ollama API error: {e}")
            raise
    
    async def agenerate(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response asynchronously.
        
        Args:
            messages: Conversation messages
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to Ollama API
            
        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()
        
        ollama_messages = self._convert_messages(messages)
        
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": ollama_messages,
                        "stream": False,
                        "options": options,
                        **kwargs,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
            
            latency = (time.time() - start_time) * 1000
            
            usage = {}
            if "prompt_eval_count" in data:
                usage["prompt_tokens"] = data["prompt_eval_count"]
            if "eval_count" in data:
                usage["completion_tokens"] = data["eval_count"]
            if "prompt_eval_count" in data and "eval_count" in data:
                usage["total_tokens"] = data["prompt_eval_count"] + data["eval_count"]
            
            result = LLMResponse(
                content=data["message"]["content"],
                model=self._model,
                usage=usage,
                finish_reason=data.get("done_reason") or "stop",
                metadata={"latency_ms": latency},
            )
            
            # Update stats using base class method
            self._update_stats(result, latency)
            
            return result
            
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Ollama API error: {e}")
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
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Yields:
            Content chunks as they're generated
        """
        ollama_messages = self._convert_messages(messages)
        
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": self._model,
                "messages": ollama_messages,
                "stream": True,
                "options": options,
                **kwargs,
            },
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        response = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    
    def pull_model(self, model: str) -> None:
        """Pull a model from Ollama library."""
        logger.info(f"Pulling model: {model}")
        response = httpx.post(
            f"{self.base_url}/api/pull",
            json={"name": model},
            timeout=None,  # No timeout for downloads
        )
        response.raise_for_status()
        logger.info(f"Model {model} pulled successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            response = httpx.post(
                f"{self.base_url}/api/show",
                json={"name": self._model},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "model": self._model,
                "provider": "ollama",
                "base_url": self.base_url,
                "parameters": data.get("parameters"),
                "template": data.get("template"),
                "details": data.get("details"),
            }
        except Exception:
            return {
                "model": self._model,
                "provider": "ollama",
                "base_url": self.base_url,
            }
