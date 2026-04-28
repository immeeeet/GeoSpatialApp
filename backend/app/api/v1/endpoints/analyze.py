"""
API endpoint handlers for site analysis, heatmap, and business suggestions.
All heavy work is delegated to AnalysisService — these handlers only
do caching, validation, and response formatting.
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from backend.app.domain.site import (
    AnalyzeRequest, AnalyzeResponse, HeatmapRequest,
    SuggestResponse, SuggestedBusiness, FeatureBreakdown,
    DemandGap, Competitor, CatchmentInfo, MapData,
    ClusteredHex, ClusterInfo, HeatmapPoint, BusinessType,
)
from backend.app.infrastructure.cache import cache
from backend.app.services.analysis import AnalysisService

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared service instance
_service = AnalysisService()


# ── POST /analyze ────────────────────────────────────────────

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_site(request: AnalyzeRequest):
    """
    Analyse a location for site readiness.
    Checks Redis cache first; on miss, runs the full ML pipeline.
    """
    cache_key = f"{request.lat}_{request.lng}_{request.business_type}_{request.radius_km}"

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Cache HIT for %s", cache_key)
        return cached

    logger.info("Cache MISS — running analysis for %s", cache_key)

    try:
        result = _service.analyze_site(
            lat=request.lat,
            lng=request.lng,
            business_type=request.business_type,
            radius_km=request.radius_km,
        )
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    # Cache the result (24hr TTL)
    cache.set(cache_key, result)
    return result


# ── GET /heatmap ─────────────────────────────────────────────

@router.get("/heatmap")
async def get_heatmap(
    business_type: BusinessType,
    city: str | None = Query(default=None, description="City name"),
    bbox: str | None = Query(default=None, description="lat_min,lng_min,lat_max,lng_max"),
):
    """
    Generate H3 hex grid with scores for a city or bounding box.
    Cached aggressively since this is expensive to compute.
    """
    if not city and not bbox:
        raise HTTPException(status_code=400, detail="Provide either 'city' or 'bbox' parameter")

    cache_key = f"heatmap_{city or bbox}_{business_type}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        scored_hexes = _service.generate_heatmap(
            business_type=business_type,
            city=city,
            bbox=bbox,
        )
    except Exception as exc:
        logger.exception("Heatmap generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Heatmap failed: {exc}")

    result = {"hex_grid": scored_hexes, "count": len(scored_hexes)}
    cache.set(cache_key, result)
    return result


# ── GET /suggest/{lat}/{lng}/{radius_km} ─────────────────────

@router.get("/suggest/{lat}/{lng}/{radius_km}", response_model=SuggestResponse)
async def suggest_businesses(lat: float, lng: float, radius_km: float):
    """
    Return top 3 underserved business types for the given location.
    Runs underserved_index for all business types and ranks by opportunity.
    """
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    if radius_km <= 0 or radius_km > 50:
        raise HTTPException(status_code=400, detail="radius_km must be between 0 and 50")

    cache_key = f"suggest_{lat}_{lng}_{radius_km}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        suggestions = _service.suggest_businesses(lat, lng, radius_km)
    except Exception as exc:
        logger.exception("Suggest failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {exc}")

    result = {
        "location": {"lat": lat, "lng": lng},
        "radius_km": radius_km,
        "suggestions": suggestions,
    }
    cache.set(cache_key, result)
    return result
