"""
Raster reader for GHSL population and VIIRS nightlights GeoTIFFs.

Uses rasterio to open each raster, masks pixels within a circular buffer
around the queried lat/lng, and returns summary statistics.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import Point, mapping

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


def _buffer_geometry(lat: float, lng: float, radius_km: float) -> dict:
    """
    Create a circular buffer around a point.
    Converts radius from km to approximate degrees (1° ≈ 111 km at equator).
    Good enough for masking — exact projection isn't needed here.
    """
    radius_deg = radius_km / 111.0
    point = Point(lng, lat)  # rasterio uses (x=lng, y=lat)
    circle = point.buffer(radius_deg)
    return mapping(circle)


def _read_raster_stats(
    raster_path: str, lat: float, lng: float, radius_km: float
) -> np.ndarray | None:
    """
    Open a GeoTIFF and extract pixel values within the circular buffer.
    Returns the masked pixel array, or None if the file is missing / unreadable.
    """
    path = Path(raster_path)
    if not path.exists():
        logger.warning("Raster file not found: %s", raster_path)
        return None

    try:
        geom = _buffer_geometry(lat, lng, radius_km)
        with rasterio.open(str(path)) as src:
            out_image, _ = rasterio_mask(src, [geom], crop=True, nodata=0)
            data = out_image[0]  # first band
            # Filter out nodata / zero values
            valid = data[data > 0]
            return valid if valid.size > 0 else None
    except Exception as exc:
        logger.error("Error reading raster %s: %s", raster_path, exc)
        return None


# ── Public API ───────────────────────────────────────────────


def read_population_density(lat: float, lng: float, radius_km: float) -> float:
    """
    Sum GHSL 2025 population pixels within the radius.
    Returns total estimated population in the area.
    Returns -1 if the raster is unavailable.
    """
    settings = get_settings()
    pixels = _read_raster_stats(settings.GHSL_RASTER_PATH, lat, lng, radius_km)
    if pixels is None:
        return -1.0

    total_population = float(np.sum(pixels))
    area_km2 = np.pi * radius_km**2
    density = total_population / max(area_km2, 0.01)

    logger.debug(
        "GHSL @ (%.4f, %.4f) r=%skm: pop=%d, density=%.0f/km²",
        lat, lng, radius_km, total_population, density,
    )
    return density


def read_nightlight_intensity(lat: float, lng: float, radius_km: float) -> float:
    """
    Mean VIIRS nightlight pixel intensity within the radius.
    Raw VIIRS values typically range 0–63 (nW/cm²/sr).
    Returns -1 if the raster is unavailable.
    """
    settings = get_settings()
    pixels = _read_raster_stats(settings.VIIRS_RASTER_PATH, lat, lng, radius_km)
    if pixels is None:
        return -1.0

    mean_intensity = float(np.mean(pixels))
    logger.debug(
        "VIIRS @ (%.4f, %.4f) r=%skm: mean=%.2f",
        lat, lng, radius_km, mean_intensity,
    )
    return mean_intensity
