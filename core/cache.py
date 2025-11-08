"""
Async Cache Manager for SkillBot
High-performance Redis-based caching for 1000+ concurrent users
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
import pickle
import hashlib
import time

from config_async import settings, REDIS_URL, DEBUG_MODE

logger = logging.getLogger(__name__)

class CacheManager:
    """Async Redis cache manager with connection pooling"""

    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize Redis connection pool"""
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool.from_url(
                self.redis_url,
                max_connections=settings.CACHE_MAX_CONNECTIONS,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )

            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False  # We'll handle encoding ourselves
            )

            # Test connection
            await self.redis_client.ping()

            logger.info("Cache manager initialized successfully with Redis connection")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {str(e)}")
            if DEBUG_MODE:
                # In development, continue without cache
                logger.warning("Continuing without cache in development mode - install Redis for full functionality")
                logger.info("To enable caching: https://redis.io/download")
                self.redis_client = None
                self.connection_pool = None
                return True
            raise

    async def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value, default=str).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except Exception:
            # Fallback to pickle
            return pickle.dumps(value)

    async def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            # Try JSON first
            decoded = value.decode('utf-8')
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Fallback to pickle
            return pickle.loads(value)

    def _make_key(self, key: str, prefix: str = "skillbot") -> str:
        """Create a properly prefixed cache key"""
        return f"{prefix}:{key}"

    async def get(self, key: str, prefix: str = "skillbot") -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None

        try:
            cache_key = self._make_key(key, prefix)
            value = await self.redis_client.get(cache_key)

            if value is None:
                return None

            return await self._deserialize_value(value)

        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: int = None,
                 prefix: str = "skillbot") -> bool:
        """Set value in cache with optional TTL"""
        if not self.redis_client:
            return False

        try:
            cache_key = self._make_key(key, prefix)
            serialized_value = await self._serialize_value(value)

            if ttl is None:
                ttl = settings.CACHE_TTL

            await self.redis_client.setex(cache_key, ttl, serialized_value)
            return True

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {str(e)}")
            return False

    async def delete(self, key: str, prefix: str = "skillbot") -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            return False

        try:
            cache_key = self._make_key(key, prefix)
            result = await self.redis_client.delete(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {str(e)}")
            return False

    async def exists(self, key: str, prefix: str = "skillbot") -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False

        try:
            cache_key = self._make_key(key, prefix)
            result = await self.redis_client.exists(cache_key)
            return result > 0

        except Exception as e:
            logger.error(f"Cache exists check failed for key {key}: {str(e)}")
            return False

    async def mget(self, keys: List[str], prefix: str = "skillbot") -> Dict[str, Any]:
        """Get multiple values from cache"""
        if not self.redis_client or not keys:
            return {}

        try:
            cache_keys = [self._make_key(key, prefix) for key in keys]
            values = await self.redis_client.mget(cache_keys)

            result = {}
            for i, value in enumerate(values):
                if value is not None:
                    try:
                        result[keys[i]] = await self._deserialize_value(value)
                    except Exception as e:
                        logger.error(f"Failed to deserialize cached value for key {keys[i]}: {str(e)}")

            return result

        except Exception as e:
            logger.error(f"Cache mget failed: {str(e)}")
            return {}

    async def mset(self, data: Dict[str, Any], ttl: int = None,
                  prefix: str = "skillbot") -> bool:
        """Set multiple values in cache"""
        if not self.redis_client or not data:
            return False

        try:
            if ttl is None:
                ttl = settings.CACHE_TTL

            pipe = self.redis_client.pipeline()

            for key, value in data.items():
                cache_key = self._make_key(key, prefix)
                serialized_value = await self._serialize_value(value)
                pipe.setex(cache_key, ttl, serialized_value)

            await pipe.execute()
            return True

        except Exception as e:
            logger.error(f"Cache mset failed: {str(e)}")
            return False

    async def increment(self, key: str, amount: int = 1,
                       prefix: str = "skillbot") -> Optional[int]:
        """Increment numeric value in cache"""
        if not self.redis_client:
            return None

        try:
            cache_key = self._make_key(key, prefix)
            result = await self.redis_client.incrby(cache_key, amount)
            return result

        except Exception as e:
            logger.error(f"Cache increment failed for key {key}: {str(e)}")
            return None

    async def expire(self, key: str, ttl: int, prefix: str = "skillbot") -> bool:
        """Set TTL for existing key"""
        if not self.redis_client:
            return False

        try:
            cache_key = self._make_key(key, prefix)
            result = await self.redis_client.expire(cache_key, ttl)
            return result

        except Exception as e:
            logger.error(f"Cache expire failed for key {key}: {str(e)}")
            return False

    async def clear_pattern(self, pattern: str, prefix: str = "skillbot") -> int:
        """Clear keys matching pattern"""
        if not self.redis_client:
            return 0

        try:
            pattern_key = self._make_key(pattern, prefix)
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern_key):
                keys.append(key)

            if keys:
                return await self.redis_client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Cache clear pattern failed for pattern {pattern}: {str(e)}")
            return 0

    # Specialized caching methods

    async def cache_user_session(self, user_id: str, session_data: Dict[str, Any],
                                ttl: int = 3600) -> bool:
        """Cache user session data"""
        return await self.set(f"session:{user_id}", session_data, ttl, "auth")

    async def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        return await self.get(f"session:{user_id}", "auth")

    async def cache_chat_response(self, session_id: str, message_hash: str,
                                 response: str, ttl: int = 1800) -> bool:
        """Cache chat AI response"""
        key = f"chat:{session_id}:{message_hash}"
        return await self.set(key, response, ttl, "ai")

    async def get_cached_chat_response(self, session_id: str,
                                     message_hash: str) -> Optional[str]:
        """Get cached chat AI response"""
        key = f"chat:{session_id}:{message_hash}"
        return await self.get(key, "ai")

    async def cache_question_results(self, query_hash: str, results: Dict[str, Any],
                                   ttl: int = 7200) -> bool:
        """Cache question search results"""
        key = f"questions:{query_hash}"
        return await self.set(key, results, ttl, "search")

    async def get_cached_question_results(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached question search results"""
        key = f"questions:{query_hash}"
        return await self.get(key, "search")

    def hash_query(self, query_data: Union[str, Dict[str, Any]]) -> str:
        """Create hash for query caching"""
        if isinstance(query_data, dict):
            query_str = json.dumps(query_data, sort_keys=True)
        else:
            query_str = str(query_data)
        return hashlib.md5(query_str.encode()).hexdigest()

    # Rate limiting helpers

    async def rate_limit_check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        """Check rate limit and return (allowed, remaining)"""
        if not self.redis_client:
            return True, limit

        try:
            current_time = int(time.time())
            window_start = current_time - window
            rate_key = self._make_key(f"rate:{key}", "limit")

            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(rate_key, 0, window_start)
            pipe.zcard(rate_key)
            pipe.zadd(rate_key, {str(current_time): current_time})
            pipe.expire(rate_key, window)

            results = await pipe.execute()
            current_count = results[1]

            if current_count < limit:
                return True, limit - current_count - 1
            else:
                return False, 0

        except Exception as e:
            logger.error(f"Rate limit check failed for key {key}: {str(e)}")
            return True, limit  # Allow on error

    async def health_check(self) -> bool:
        """Check cache health"""
        if not self.redis_client:
            # In debug mode, it's OK to not have cache
            return DEBUG_MODE

        try:
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Cache health check failed: {str(e)}")
            return False

    async def get_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"status": "disabled"}

        try:
            info = await self.redis_client.info()
            return {
                "status": "connected",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get cache info: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def close(self) -> None:
        """Close cache connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("🔌 Cache connection closed")
        except Exception as e:
            logger.error(f"Error closing cache connection: {str(e)}")