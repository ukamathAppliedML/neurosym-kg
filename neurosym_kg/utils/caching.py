"""
Caching utilities for NeuroSym-KG.

Provides disk-based and in-memory caching for LLM responses and KG queries.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.

    Example:
        >>> cache = InMemoryCache(ttl_seconds=3600)
        >>> cache.set("key", "value")
        >>> cache.get("key")
        'value'
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000) -> None:
        self._cache: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl_seconds
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache."""
        # Evict if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_oldest()

        self._cache[key] = (value, time.time())

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def _evict_oldest(self) -> None:
        """Evict the oldest entry."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest_key]

    def __len__(self) -> int:
        return len(self._cache)


class DiskCache:
    """
    Disk-based cache using JSON files.

    Example:
        >>> cache = DiskCache("/tmp/cache")
        >>> cache.set("key", {"data": "value"})
        >>> cache.get("key")
        {'data': 'value'}
    """

    def __init__(
        self,
        cache_dir: str | Path,
        ttl_seconds: int = 86400,
        max_size_mb: int = 1024,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._max_size_bytes = max_size_mb * 1024 * 1024

    def _key_to_path(self, key: str) -> Path:
        """Convert a key to a file path."""
        # Hash the key for a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        path = self._key_to_path(key)

        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)

            # Check TTL
            if time.time() - data["timestamp"] > self._ttl:
                path.unlink()
                return None

            return data["value"]

        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value in cache."""
        path = self._key_to_path(key)

        data = {
            "key": key,
            "value": value,
            "timestamp": time.time(),
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except (TypeError, OSError):
            pass  # Silently fail for non-serializable values

    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all cached values."""
        for path in self._cache_dir.glob("*.json"):
            try:
                path.unlink()
            except OSError:
                pass


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a cache key from arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def cached(
    cache: InMemoryCache | DiskCache,
    key_fn: Callable[..., str] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to cache function results.

    Example:
        >>> cache = InMemoryCache()
        >>> @cached(cache)
        ... def expensive_function(x):
        ...     return x * 2
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = cache_key(func.__name__, *args, **kwargs)

            # Try cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result)

            return result

        return wrapper

    return decorator
