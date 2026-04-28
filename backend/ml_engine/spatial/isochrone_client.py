"""
Drive-time isochrone client using OpenRouteService (ORS).
Generates a GeoJSON polygon for the area reachable by car within N minutes,
then overlays on GHSL raster to estimate catchment population.
"""

import logging
import requests
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_drive_time_isochrone(lat: float, lng: float, minutes: int = 15) -> dict | None:
    """Fetch drive-time isochrone polygon from ORS. Returns GeoJSON Feature or None."""
    settings = get_settings()
    if not settings.ORS_API_KEY:
        logger.warning("ORS_API_KEY not set — isochrone unavailable.")
        return None

    url = "https://api.openrouteservice.org/v2/isochrones/driving-car"
    headers = {"Authorization": settings.ORS_API_KEY, "Content-Type": "application/json"}
    body = {"locations": [[lng, lat]], "range": [minutes * 60], "range_type": "time", "attributes": ["area"]}

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return None
        return features[0]
    except requests.RequestException as exc:
        logger.error("ORS isochrone failed: %s", exc)
        return None


def estimate_catchment_population(lat: float, lng: float, minutes: int = 15) -> tuple[dict | None, int]:
    """Get isochrone + estimate population inside it via GHSL raster overlay."""
    isochrone = get_drive_time_isochrone(lat, lng, minutes)
    if isochrone is None:
        return None, 0
    try:
        import numpy as np
        import rasterio
        from rasterio.mask import mask as rasterio_mask
        from shapely.geometry import shape

        settings = get_settings()
        iso_geom = shape(isochrone["geometry"])
        with rasterio.open(settings.GHSL_RASTER_PATH) as src:
            out_image, _ = rasterio_mask(src, [iso_geom.__geo_interface__], crop=True, nodata=0)
            population = int(np.sum(out_image[0][out_image[0] > 0]))
        return isochrone, population
    except Exception as exc:
        logger.warning("Catchment population estimation failed: %s", exc)
        return isochrone, 0
