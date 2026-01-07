"""
LLM backend implementations for NeuroSym-KG.

Provides connectors for various LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude) - coming soon
- HuggingFace (local models) - coming soon
- vLLM (local deployment) - coming soon
- Mock (for testing)
"""

from neurosym_kg.llm_backends.base import BaseLLMBackend, BaseStreamingLLMBackend
from neurosym_kg.llm_backends.mock_backend import MockLLMBackend, ReplayLLMBackend

# Conditional imports for optional backends
try:
    from neurosym_kg.llm_backends.openai_backend import OpenAIBackend
except ImportError:
    OpenAIBackend = None  # type: ignore

try:
    from neurosym_kg.llm_backends.anthropic_backend import AnthropicBackend, ClaudeBackend
except ImportError:
    AnthropicBackend = None  # type: ignore
    ClaudeBackend = None  # type: ignore

try:
    from neurosym_kg.llm_backends.ollama_backend import OllamaBackend
except ImportError:
    OllamaBackend = None  # type: ignore



__all__ = [
    "BaseLLMBackend",
    "BaseStreamingLLMBackend",
    "MockLLMBackend",
    "ReplayLLMBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "ClaudeBackend",
    "OllamaBackend",
]
