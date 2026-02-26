"""
Translation caching for improved performance.
Supports in-memory and Redis-based caching.
"""
import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class CacheManager(ABC):
    """Abstract base class for cache managers."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

    @staticmethod
    def create_cache_key(
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Create a deterministic cache key.

        Args:
            text: Input text
            source_lang: Source language
            target_lang: Target language

        Returns:
            Cache key string
        """
        # Create a unique key from inputs
        key_data = f"{source_lang}:{target_lang}:{text}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        return f"translation:{key_hash}"


class MemoryCache(CacheManager):
    """
    In-memory LRU cache implementation.
    Fast but not shared across processes.
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry: Dict[str, float] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
        }
        logger.info(f"Initialized memory cache (max_size={max_size}, ttl={default_ttl}s)")

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        # Check if key exists
        if key not in self.cache:
            self.stats["misses"] += 1
            return None

        # Check if expired
        if key in self.expiry and time.time() > self.expiry[key]:
            self.delete(key)
            self.stats["misses"] += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.stats["hits"] += 1

        return self.cache[key]

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        # Remove oldest if at capacity
        if key not in self.cache and len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
            self.stats["evictions"] += 1

        # Set value
        self.cache[key] = value
        self.cache.move_to_end(key)

        # Set expiry
        ttl = ttl or self.default_ttl
        if ttl > 0:
            self.expiry[key] = time.time() + ttl

        self.stats["sets"] += 1

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        self.expiry.clear()
        logger.info(f"Cleared {count} cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate_percent": round(hit_rate, 2),
        }


class RedisCache(CacheManager):
    """
    Redis-based cache implementation.
    Slower than memory but shared across processes.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600,
        key_prefix: str = "nllb:",
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default TTL in seconds
            key_prefix: Prefix for all keys
        """
        try:
            import redis

            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
            self.redis_client.ping()  # Test connection

            self.default_ttl = default_ttl
            self.key_prefix = key_prefix
            self.stats_key = f"{key_prefix}stats"

            logger.info(f"Connected to Redis at {host}:{port}/{db}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            value = self.redis_client.get(self._prefixed_key(key))
            if value:
                self.redis_client.hincrby(self.stats_key, "hits", 1)
            else:
                self.redis_client.hincrby(self.stats_key, "misses", 1)
            return value
        except Exception as e:
            logger.error(f"Redis GET error: {str(e)}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        try:
            ttl = ttl or self.default_ttl
            prefixed_key = self._prefixed_key(key)

            if ttl > 0:
                self.redis_client.setex(prefixed_key, ttl, value)
            else:
                self.redis_client.set(prefixed_key, value)

            self.redis_client.hincrby(self.stats_key, "sets", 1)

        except Exception as e:
            logger.error(f"Redis SET error: {str(e)}")

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        try:
            self.redis_client.delete(self._prefixed_key(key))
        except Exception as e:
            logger.error(f"Redis DELETE error: {str(e)}")

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            keys = self.redis_client.keys(f"{self.key_prefix}*")
            if keys:
                self.redis_client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
        except Exception as e:
            logger.error(f"Redis CLEAR error: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            stats = self.redis_client.hgetall(self.stats_key)
            hits = int(stats.get("hits", 0))
            misses = int(stats.get("misses", 0))
            sets = int(stats.get("sets", 0))

            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0

            return {
                "hits": hits,
                "misses": misses,
                "sets": sets,
                "hit_rate_percent": round(hit_rate, 2),
                "backend": "redis",
            }
        except Exception as e:
            logger.error(f"Redis STATS error: {str(e)}")
            return {}


def create_cache(cache_type: str = "memory", **kwargs) -> CacheManager:
    """
    Factory function to create cache instance.

    Args:
        cache_type: Type of cache ("memory" or "redis")
        **kwargs: Cache-specific parameters

    Returns:
        CacheManager instance
    """
    if cache_type == "memory":
        return MemoryCache(**kwargs)
    elif cache_type == "redis":
        return RedisCache(**kwargs)
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
