"""Redis cache for RAG answers.

Keys include a namespace version that is bumped whenever notes are
(re)indexed, so answers never outlive the content they were built from.
"""

import hashlib
import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from vtu_rag.config import CacheSettings, RedisSettings

logger = logging.getLogger(__name__)

_PREFIX = "vtu-rag"
_VERSION_KEY = f"{_PREFIX}:cache-version"


class ResponseCache:
    def __init__(self, redis_settings: RedisSettings, cache_settings: CacheSettings):
        self.enabled = cache_settings.enabled
        self.ttl = cache_settings.ttl_seconds
        self.redis = Redis.from_url(redis_settings.url, decode_responses=True)

    @staticmethod
    def make_key(kind: str, **parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, default=str)
        return f"{kind}:{hashlib.sha256(payload.encode()).hexdigest()}"

    async def _versioned(self, key: str) -> str:
        version = await self.redis.get(_VERSION_KEY) or "0"
        return f"{_PREFIX}:v{version}:{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            raw = await self.redis.get(await self._versioned(key))
        except RedisError as exc:
            logger.warning("Cache read failed: %s", exc)
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            await self.redis.set(
                await self._versioned(key), json.dumps(value, default=str), ex=self.ttl
            )
        except RedisError as exc:
            logger.warning("Cache write failed: %s", exc)

    async def invalidate(self) -> None:
        """Logically clears all cached answers; old entries expire via TTL."""
        try:
            await self.redis.incr(_VERSION_KEY)
        except RedisError as exc:
            logger.warning("Cache invalidation failed: %s", exc)

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    async def close(self) -> None:
        await self.redis.aclose()
