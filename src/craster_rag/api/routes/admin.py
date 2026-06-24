"""
admin.py
────────
Admin endpoints for the RAG API.

Endpoints:
    GET  /api/v1/admin/stats     vector store statistics
    GET  /api/v1/health          health check
    POST /api/v1/admin/reindex   trigger re-indexing

These endpoints are for internal use only.
In production these would be protected by auth.
"""


import logging
from fastapi import APIRouter, Request, HTTPException


from config import settings
from craster_rag.retrieval.vector_store import VectorStore
from craster_rag.api.models.response import (
    AdminStatsResponse,
    HealthResponse,
)


# logger
logger = logging.getLogger(__name__)


# router
router = APIRouter(tags=["admin"])

# vector store instance
_vector_store = VectorStore()

@router.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
)
async def health() -> HealthResponse:
    """
    Used by Docker health checks and
    monitoring systems to verify app is running.
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
        stats = _vector_store.get_stats()
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
