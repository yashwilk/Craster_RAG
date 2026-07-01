import hashlib
import json
import logging
from typing import Optional

from config import settings


logger = logging.getLogger(__name__)

CACHEABLE_FIELDS = [
    "final_answer",
    "answer",
    "sources",
    "citations",
    "category",
    "confidence_level",
    "can_answer",
]

class CacheClient:
    def __init__(self):
        self._client = None
        self.enabled = False
        self.hits = 0
        self.misses = 0
        self._setup()

    def _setup(self) -> None:
        """
        Connect to Redis if caching is enabled in config.
        Silently disables caching if Redis is unreachable —
        the pipeline must never depend on cache availability.
        """
        if not settings.enable_caching:
            logger.info("Caching disabled via ENABLE_CACHING=false")
            return
        try:
            import redis
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._client.ping()
            self.enabled = True
            logger.info(f"Redis cache connected: {settings.redis_url}")

        except ImportError:
            logger.warning("redis package not installed — pip install redis")

        except Exception as e:
            logger.warning(
                f"Redis unavailable ({e}) — caching disabled, "
                f"pipeline will run on every request"
            )

    def get(self, question: str) -> Optional[dict]:
        """Retrieve a cached answer for this question, if any."""
        if not self.enabled:
            return None

        key = self._build_key(question)
        try:
            cached_json = self._client.get(key)
            if cached_json is None:
                self.misses += 1
                logger.debug(f"Cache MISS: '{question[:50]}...'")
                return None

            self.hits += 1
            logger.info(f"Cache HIT: '{question[:50]}...'")
            return json.loads(cached_json)

        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, question: str, result: dict) -> None:
        """Store an answer in the cache with TTL.

        Only caches answers where can_answer is True —
        we never want to cache "I don't know" responses,
        since better context might be available on retry
        after a future reindex.
        """
        if not self.enabled:
            return
        if not result.get("can_answer", False):
            logger.debug("Skipping cache — can_answer is False")
            return
        key = self._build_key(question)
        payload = {
            field: result.get(field)
            for field in CACHEABLE_FIELDS
        }
        try:
            self._client.setex(
                key,
                settings.cache_ttl_seconds,
                json.dumps(payload),
            )
            logger.debug(
                f"Cached answer for '{question[:50]}...' "
                f"(TTL={settings.cache_ttl_seconds}s)"
            )

        except Exception as e:
            logger.warning(f"Cache set failed: {e}")

    def _build_key(self, question: str) -> str:
        """Build a normalised cache key from a question.

        Normalisation (lowercase, strip whitespace) ensures"""
        normalised = question.strip().lower()
        digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()
        return f"rag_answer:{digest}"

    def invalidate_all(self) -> int:
        """Clear all cached RAG answers.

        Called by scripts/reindex.py after a full reindex,
        since previously cached answers may now be stale.
        """
        if not self.enabled:
            return 0

        try:
            keys = self._client.keys("rag_answer:*")
            if not keys:
                return 0

            deleted = self._client.delete(*keys)
            logger.info(f"Cache invalidated: {deleted} entries cleared")
            return deleted

        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
            return 0

    def stats(self) -> dict:
        """Get cache hit/miss statistics for this process.
        Useful for monitoring cache effectiveness —
        a low hit rate may mean TTL is too short or
        questions are too varied to benefit from caching."""
        total = self.hits + self.misses
        hit_rate = round(self.hits / total, 3) if total > 0 else 0.0

        return {
            "enabled"  : self.enabled,
            "hits"     : self.hits,
            "misses"   : self.misses,
            "hit_rate" : hit_rate,
        }


cache = CacheClient()
