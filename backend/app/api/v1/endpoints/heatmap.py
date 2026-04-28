"""
Heatmap zones endpoint — city-wide DBSCAN opportunity zone detection.
Returns top 5 opportunity clusters with full GeoJSON for the frontend
heatmap explorer page.
"""

import logging
from fastapi import APIRouter, HTTPException, Query

from backend.app.domain.site import BusinessType
from backend.app.infrastructure.cache import cache
from backend.ml_engine.spatial.cluster_engine import get_opportunity_zones

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/heatmap/zones")
async def get_heatmap_zones(
    business_type: BusinessType,
    city: str | None = Query(default=None, description="City name (e.g. 'ahmedabad')"),
    bbox: str | None = Query(default=None, description="lat_min,lng_min,lat_max,lng_max"),
):
    """
    Find top 5 opportunity zones in a city using DBSCAN clustering.
    Each zone includes cluster stats and full GeoJSON for map rendering.
    """
    if not city and not bbox:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'city' or 'bbox' parameter",
        )

    cache_key = f"zones_{city or bbox}_{business_type}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Cache HIT for heatmap zones: %s", cache_key)
        return cached

    logger.info("Computing opportunity zones for %s / %s", city or bbox, business_type)

    try:
        zones = get_opportunity_zones(
            city=city,
            business_type=business_type,
            bbox=bbox,
        )
    except Exception as exc:
        logger.exception("Opportunity zone detection failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Zone detection failed: {exc}",
        )

    result = {
        "city": city,
        "business_type": business_type,
        "zones": zones,
        "zone_count": len(zones),
    }
    cache.set(cache_key, result)
    return result
