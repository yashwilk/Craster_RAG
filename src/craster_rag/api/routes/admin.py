
import logging
from typing import Optional
 
from fastapi import APIRouter, HTTPException
 
from craster_rag.retrieval.vector_store import VectorStore
from craster_rag.cache.cache_client import cache
from craster_rag.api.models.response import (
    AdminStatsResponse,
    HealthResponse,
)
from config import settings
 
# logger
logger = logging.getLogger(__name__)
 
# router
router = APIRouter(tags=["admin"])
 
# ── Lazy loaded vector store ────────────────────────────
# avoids connecting to Supabase just from importing this
# module (e.g. during testing or tooling)
_vector_store: Optional[VectorStore] = None
 
 
def _get_vector_store() -> VectorStore:
    """Lazily create and cache the VectorStore instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
 
 
@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
)
async def health() -> HealthResponse:
    """
    Health check endpoint.
 
    Used by Docker health checks and
    monitoring systems to verify app is running.
 
    Returns:
        HealthResponse with status and version
    """
    return HealthResponse(
        status  = "ok",
        version = settings.app_version,
    )
 
 
@router.get(
    "/api/v1/admin/stats",
    response_model = AdminStatsResponse,
    summary        = "Vector store statistics",
)
async def get_stats() -> AdminStatsResponse:

    try:
        store = _get_vector_store()
        stats = store.get_stats()
        return AdminStatsResponse(
            total_chunks   = stats["total_chunks"],
            unique_sources = stats["unique_sources"],
            categories     = stats["categories"],
        )

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = "Failed to fetch statistics",
        )

@router.get(
    "/api/v1/admin/cache-stats",
    summary = "Cache hit/miss statistics",
)
async def get_cache_stats() -> dict:
    return cache.stats()