"""
FastAPI application entry point for Terrascope ML Engine.

Run with: uvicorn backend.app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.router import router as v1_router
from backend.app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    logger.info("Terrascope ML Engine starting on %s:%s", settings.APP_HOST, settings.APP_PORT)
    logger.info("Debug mode: %s", settings.DEBUG)
    yield
    # Shutdown: close DB pool
    try:
        from backend.app.infrastructure.database import close_pool
        close_pool()
    except Exception:
        pass
    logger.info("Terrascope ML Engine shut down.")


app = FastAPI(
    title="Terrascope ML Engine",
    description="Site readiness scoring API — analyse any location for business viability.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount v1 API
app.include_router(v1_router)


@app.get("/health", tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "terrascope-ml-engine"}
