"""
Live competitor and POI queries via the Overpass API.

Falls back to PostGIS OSM data (osm_parser.count_pois_from_postgis)
if the Overpass API is unreachable or rate-limited.
"""

import logging
import math

import requests

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Business type → Overpass tag mapping ─────────────────────
# Each business type maps to one or more (key, value) OSM tag pairs
BUSINESS_TAG_MAP: dict[str, list[tuple[str, str]]] = {
    "gym": [("leisure", "fitness_centre"), ("amenity", "gym")],
    "restaurant": [("amenity", "restaurant")],
    "clinic": [("amenity", "clinic"), ("amenity", "doctors")],
    "pharmacy": [("amenity", "pharmacy")],
    "retail_store": [("shop", "supermarket"), ("shop", "convenience"), ("shop", "department_store")],
    "ev_charging": [("amenity", "charging_station")],
    "warehouse": [("building", "warehouse"), ("landuse", "industrial")],
    "cafe": [("amenity", "cafe")],
    "school": [("amenity", "school")],
    "bank": [("amenity", "bank")],
}


def _build_overpass_query(
    lat: float, lng: float, radius_m: float, tags: list[tuple[str, str]]
) -> str:
    """
    Build an Overpass QL query that finds nodes/ways matching any of the
    given tags within a radius around the point.
    """
    tag_filters = []
    for key, value in tags:
        tag_filters.append(f'node["{key}"="{value}"](around:{radius_m},{lat},{lng});')
        tag_filters.append(f'way["{key}"="{value}"](around:{radius_m},{lat},{lng});')

    query_body = "\n    ".join(tag_filters)
    return f"""
[out:json][timeout:25];
(
    {query_body}
);
out center;
"""


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance between two points in metres (Haversine formula)."""
    R = 6371000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_overpass_elements(
    elements: list[dict], origin_lat: float, origin_lng: float
) -> list[dict]:
    """
    Extract name, lat, lng, distance_m from Overpass JSON elements.
    Handles both nodes (direct lat/lon) and ways (center lat/lon).
    """
    results = []
    for el in elements:
        # Ways have their centre under the "center" key
        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lng = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is None or el_lng is None:
            continue

        name = el.get("tags", {}).get("name", "Unnamed")
        dist = _haversine_m(origin_lat, origin_lng, el_lat, el_lng)

        results.append({
            "name": name,
            "lat": round(el_lat, 6),
            "lng": round(el_lng, 6),
            "distance_m": round(dist),
        })

    # Sort by distance
    results.sort(key=lambda x: x["distance_m"])
    return results


# ── Public API ───────────────────────────────────────────────


def find_competitors(
    lat: float, lng: float, radius_km: float, business_type: str
) -> list[dict]:
    """
    Find competing businesses around a point using Overpass API.
    Returns list of {name, lat, lng, distance_m}.
    Falls back to PostGIS if Overpass is unreachable.
    """
    tags = BUSINESS_TAG_MAP.get(business_type)
    if not tags:
        logger.warning("Unknown business_type '%s', no tags to query.", business_type)
        return []

    settings = get_settings()
    radius_m = radius_km * 1000
    query = _build_overpass_query(lat, lng, radius_m, tags)

    try:
        resp = requests.post(
            settings.OVERPASS_API_URL,
            data={"data": query},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        competitors = _parse_overpass_elements(elements, lat, lng)
        logger.info(
            "Overpass: found %d competitors for '%s' near (%.4f, %.4f)",
            len(competitors), business_type, lat, lng,
        )
        return competitors

    except (requests.RequestException, ValueError) as exc:
        logger.warning(
            "Overpass API failed (%s), falling back to PostGIS.", exc
        )
        return _fallback_postgis(lat, lng, radius_km, tags)


def _fallback_postgis(
    lat: float, lng: float, radius_km: float, tags: list[tuple[str, str]]
) -> list[dict]:
    """
    Silent fallback: query PostGIS OSM data when Overpass is unavailable.
    """
    try:
        from backend.app.infrastructure.database import get_db_connection
        from backend.ml_engine.spatial.osm_parser import count_pois_from_postgis

        with get_db_connection() as conn:
            results = count_pois_from_postgis(lat, lng, radius_km, tags, conn)
            # Add approximate distance
            for r in results:
                r["distance_m"] = round(
                    _haversine_m(lat, lng, r["lat"], r["lng"])
                )
            results.sort(key=lambda x: x["distance_m"])
            logger.info("PostGIS fallback: found %d POIs", len(results))
            return results
    except Exception as exc:
        logger.error("PostGIS fallback also failed: %s", exc)
        return []


def count_pois_by_type(
    lat: float, lng: float, radius_km: float, poi_type: str
) -> int:
    """
    Quick count of any POI type in radius (used for underserved analysis).
    Uses Overpass with a short timeout, falls back to 0.
    """
    tags = BUSINESS_TAG_MAP.get(poi_type, [])
    if not tags:
        return 0

    competitors = find_competitors(lat, lng, radius_km, poi_type)
    return len(competitors)
