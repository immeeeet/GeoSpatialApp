"""
Redis cache wrapper with fail-silent behaviour.
If Redis is unavailable the application keeps running — cache misses
simply trigger a fresh computation.
"""

import json
import logging

import redis

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Thin wrapper around redis-py with JSON serialisation."""

    def __init__(self) -> None:
        settings = get_settings()
        try:
            self._client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            # Quick connectivity check
            self._client.ping()
            self._available = True
            logger.info("Redis connected at %s", settings.REDIS_URL)
        except (redis.ConnectionError, redis.TimeoutError) as exc:
            logger.warning("Redis unavailable (%s) — caching disabled.", exc)
            self._client = None
            self._available = False

    # ── public API ───────────────────────────────────────────

    def get(self, key: str) -> dict | None:
        """Fetch a cached JSON value. Returns None on miss or error."""
        if not self._available:
            return None
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Redis GET failed for key=%s: %s", key, exc)
            return None

    def set(self, key: str, value: dict, ttl: int | None = None) -> bool:
        """Store a JSON value. Returns True on success."""
        if not self._available:
            return False
        if ttl is None:
            ttl = get_settings().CACHE_TTL_SECONDS
        try:
            self._client.set(key, json.dumps(value, default=str), ex=ttl)
            return True
        except Exception as exc:
            logger.warning("Redis SET failed for key=%s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """Remove a cached key. Returns True on success."""
        if not self._available:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception as exc:
            logger.warning("Redis DELETE failed for key=%s: %s", key, exc)
            return False


# Module-level singleton — import and use directly
cache = RedisCache()
