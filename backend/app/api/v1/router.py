"""
v1 API router — aggregates all endpoint modules under /api/v1.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import analyze, heatmap

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Mount endpoint routers
router.include_router(analyze.router)
router.include_router(heatmap.router)
