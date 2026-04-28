"""
Elevation lookup and flood risk estimation.

Uses the `elevation` Python module (SRTM data) to get ground height.
Combines elevation with proximity to water bodies to estimate flood risk.
"""

import logging
import math

import requests

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_elevation(lat: float, lng: float) -> float | None:
    """
    Get elevation in metres for a given lat/lng using the Open-Elevation API.
    Returns None if the service is unavailable.
    """
    try:
        # Using open-elevation.com as a free, no-key-needed service
        url = "https://api.open-elevation.com/api/v1/lookup"
        resp = requests.get(
            url,
            params={"locations": f"{lat},{lng}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if results:
            return float(results[0]["elevation"])
        return None
    except Exception as exc:
        logger.warning("Elevation API failed for (%.4f, %.4f): %s", lat, lng, exc)
        return None


def _check_water_proximity(lat: float, lng: float, radius_m: int = 500) -> bool:
    """
    Quick Overpass query to check if there's a water body within radius_m metres.
    Returns True if water is nearby, False otherwise.
    """
    settings = get_settings()
    query = f"""
[out:json][timeout:10];
(
    way["natural"="water"](around:{radius_m},{lat},{lng});
    way["waterway"](around:{radius_m},{lat},{lng});
    relation["natural"="water"](around:{radius_m},{lat},{lng});
);
out count;
"""
    try:
        resp = requests.post(
            settings.OVERPASS_API_URL,
            data={"data": query},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        # "out count" puts the count in elements[0].tags.total
        elements = data.get("elements", [])
        if elements:
            total = int(elements[0].get("tags", {}).get("total", 0))
            return total > 0
        return False
    except Exception as exc:
        logger.warning("Water proximity check failed: %s", exc)
        return False  # Assume no water if we can't check


def compute_flood_risk(lat: float, lng: float) -> tuple[float, str]:
    """
    Estimate flood risk score (0–100, higher = safer).

    Logic:
      - Elevation < 10m AND near water → high risk (score 20–40)
      - Elevation < 10m, no water     → moderate risk (score 50–60)
      - Elevation 10–30m, near water  → moderate risk (score 55–65)
      - Elevation > 30m               → low risk (score 80–95)
      - API failure                   → neutral fallback (score 75)
    """
    elevation = get_elevation(lat, lng)

    if elevation is None:
        logger.warning(
            "Elevation unavailable for (%.4f, %.4f), defaulting flood_risk=75",
            lat, lng,
        )
        return 75.0, "elevation: unknown, defaulting to neutral"

    near_water = _check_water_proximity(lat, lng)

    # Score calculation
    if elevation < 10:
        if near_water:
            score = max(20.0, 40.0 - (10.0 - elevation) * 2)
            risk_level = "high risk"
        else:
            score = 55.0
            risk_level = "moderate risk (low elevation)"
    elif elevation < 30:
        if near_water:
            score = 60.0
            risk_level = "moderate risk (near water)"
        else:
            score = 75.0
            risk_level = "low-moderate risk"
    elif elevation < 100:
        score = 85.0
        risk_level = "low risk"
    else:
        score = 95.0
        risk_level = "very low risk"

    desc = f"elevation: {elevation:.0f}m, {risk_level}"
    if near_water:
        desc += ", water body within 500m"

    return round(score, 1), desc
