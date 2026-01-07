"""
Configuration management for the NeuroSym-KG framework.

Supports environment variables, config files, and programmatic configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Configuration for LLM backends."""

    model_config = SettingsConfigDict(env_prefix="NEUROSYM_LLM_")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = "https://api.openai.com/v1"
    openai_default_model: str = "gpt-4o-mini"

    # Anthropic
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_default_model: str = "claude-sonnet-4-20250514"

    # HuggingFace
    huggingface_token: str = Field(default="", alias="HF_TOKEN")
    huggingface_default_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    # vLLM
    vllm_base_url: str = "http://localhost:8000"

    # Common settings
    default_temperature: float = 0.0
    default_max_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_retries: int = 3


class KGConfig(BaseSettings):
    """Configuration for Knowledge Graph backends."""

    model_config = SettingsConfigDict(env_prefix="NEUROSYM_KG_")

    # Wikidata
    wikidata_endpoint: str = "https://query.wikidata.org/sparql"
    wikidata_user_agent: str = "NeuroSym-KG/0.1 (research framework)"

    # Freebase (usually via local dump)
    freebase_path: str = ""

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")
    neo4j_database: str = "neo4j"

    # Common settings
    default_limit: int = 100
    max_hops: int = 3
    timeout_seconds: float = 30.0


class CacheConfig(BaseSettings):
    """Configuration for caching."""

    model_config = SettingsConfigDict(env_prefix="NEUROSYM_CACHE_")

    enabled: bool = True
    directory: str = str(Path.home() / ".cache" / "neurosym_kg")
    ttl_seconds: int = 86400  # 24 hours
    max_size_mb: int = 1024  # 1GB


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""

    model_config = SettingsConfigDict(env_prefix="NEUROSYM_EMBED_")

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "auto"  # "cpu", "cuda", "mps", or "auto"
    normalize: bool = True


class LoggingConfig(BaseSettings):
    """Configuration for logging."""

    model_config = SettingsConfigDict(env_prefix="NEUROSYM_LOG_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "console"
    file_path: str = ""  # Empty means no file logging


class Config(BaseSettings):
    """
    Main configuration class for the NeuroSym-KG framework.

    Can be configured via:
    - Environment variables (NEUROSYM_* prefix)
    - Programmatic instantiation
    - Config file (NEUROSYM_CONFIG_PATH env var)
    """

    model_config = SettingsConfigDict(
        env_prefix="NEUROSYM_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Sub-configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    kg: KGConfig = Field(default_factory=KGConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Global settings
    rate_limit_rpm: int = 60  # Requests per minute
    async_batch_size: int = 10
    debug_mode: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from a dictionary."""
        return cls(**data)

    def get_cache_dir(self) -> Path:
        """Get the cache directory, creating it if needed."""
        path = Path(os.path.expanduser(self.cache.directory))
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global default config instance
_default_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _default_config
    _default_config = config
