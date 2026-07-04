import logging
from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
 
from craster_rag.api.routes import chat, admin
from craster_rag.api.middleware.rate_limiter import (
    limiter,
    rate_limit_exceeded_handler,
)
from config import settings

# ── Logging setup ──────────────────────────────────────
logging.basicConfig(
    level  = getattr(logging, settings.log_level),
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────
app = FastAPI(
    title       = "Craster HR Policy RAG",
    description = "Multi-agent RAG chatbot for Craster HR policies",
    version     = settings.app_version,
    docs_url    = "/docs",      # Swagger UI at /docs
    redoc_url   = "/redoc",     # ReDoc at /redoc
)

# ── Rate limiting ──────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────
# allows frontend to call API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],      # restrict in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
 

  
# ── Routes ─────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(admin.router)


# ── Startup event ──────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Log app startup."""
    logger.info(
        f"Craster RAG API starting — "
        f"version={settings.app_version}, "
        f"environment={settings.environment.value}"
    )
 
# ── Root endpoint ──────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "name"       : "Craster HR Policy RAG",
        "version"    : settings.app_version,
        "status"     : "running",
        "docs"       : "/docs",
        "health"     : "/health",
        "chat"       : "/api/v1/chat",
    }
 