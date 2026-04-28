"""
OSM feature queries against PostGIS.

Expects osm2pgsql-imported data with standard tables:
  - planet_osm_line   (roads / highways)
  - planet_osm_polygon (land use, buildings)
  - planet_osm_point   (POIs, amenities)

All spatial queries use ST_DWithin with geography casts for metre-accurate results.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Road type weights — higher value = more important road
ROAD_WEIGHTS: dict[str, int] = {
    "motorway": 5,
    "motorway_link": 4,
    "trunk": 5,
    "trunk_link": 4,
    "primary": 4,
    "primary_link": 3,
    "secondary": 3,
    "secondary_link": 2,
    "tertiary": 2,
    "tertiary_link": 1,
    "residential": 1,
    "unclassified": 1,
}

# Land use suitability scores (generic — business-specific overrides in weights.py)
LAND_USE_SCORES: dict[str, float] = {
    "commercial": 95,
    "retail": 90,
    "industrial": 50,
    "residential": 70,
    "mixed": 80,
    "farmland": 20,
    "forest": 10,
    "meadow": 15,
    "recreation_ground": 40,
}


def _execute_query(conn: Any, sql: str, params: tuple) -> list[tuple]:
    """Run a SQL query and return all rows. Logs errors but doesn't crash."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    except Exception as exc:
        logger.error("PostGIS query failed: %s — %s", exc, sql[:120])
        return []


# ── Roads ────────────────────────────────────────────────────


def count_roads_by_type(
    lat: float, lng: float, radius_km: float, conn: Any
) -> dict[str, int]:
    """
    Count highway segments within radius, grouped by highway type.
    Uses ST_DWithin on geography for metre accuracy.
    """
    radius_m = radius_km * 1000
    sql = """
        SELECT highway, COUNT(*) AS cnt
        FROM planet_osm_line
        WHERE highway IS NOT NULL
          AND ST_DWithin(
              way::geography,
              ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
              %s
          )
        GROUP BY highway
        ORDER BY cnt DESC;
    """
    rows = _execute_query(conn, sql, (lng, lat, radius_m))
    return {row[0]: row[1] for row in rows}


def get_road_score(
    lat: float, lng: float, radius_km: float, conn: Any
) -> tuple[float, str]:
    """
    Weighted road accessibility score (0–100).
    Weights higher-class roads more heavily.
    Returns (score, human-readable description).
    """
    road_counts = count_roads_by_type(lat, lng, radius_km, conn)
    if not road_counts:
        return 30.0, "No road data available"

    weighted_sum = 0
    total_count = 0
    major_roads = 0

    for road_type, count in road_counts.items():
        weight = ROAD_WEIGHTS.get(road_type, 1)
        weighted_sum += count * weight
        total_count += count
        if weight >= 3:
            major_roads += count

    # Normalize: a weighted_sum of 50+ is considered excellent (score=100)
    score = min(100.0, (weighted_sum / 50.0) * 100.0)
    desc = f"{major_roads} major roads within {radius_km}km ({total_count} total segments)"
    return round(score, 1), desc


# ── Land Use ─────────────────────────────────────────────────


def get_land_use(
    lat: float, lng: float, radius_km: float, conn: Any
) -> tuple[str, float]:
    """
    Determine dominant land use category and its suitability score.
    Queries planet_osm_polygon for landuse tags, picks the most common one.
    Returns (category_name, score_0_to_100).
    """
    radius_m = radius_km * 1000
    sql = """
        SELECT landuse, COUNT(*) AS cnt
        FROM planet_osm_polygon
        WHERE landuse IS NOT NULL
          AND ST_DWithin(
              way::geography,
              ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
              %s
          )
        GROUP BY landuse
        ORDER BY cnt DESC
        LIMIT 5;
    """
    rows = _execute_query(conn, sql, (lng, lat, radius_m))
    if not rows:
        return "unknown", 50.0  # neutral fallback

    dominant = rows[0][0]
    score = LAND_USE_SCORES.get(dominant, 50.0)
    return dominant, score


# ── Buildings ────────────────────────────────────────────────


def count_buildings(
    lat: float, lng: float, radius_km: float, conn: Any
) -> int:
    """Count building footprints within the radius."""
    radius_m = radius_km * 1000
    sql = """
        SELECT COUNT(*)
        FROM planet_osm_polygon
        WHERE building IS NOT NULL
          AND ST_DWithin(
              way::geography,
              ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
              %s
          );
    """
    rows = _execute_query(conn, sql, (lng, lat, radius_m))
    return rows[0][0] if rows else 0


# ── POI Counts (fallback for Overpass) ───────────────────────


def count_pois_from_postgis(
    lat: float, lng: float, radius_km: float, tags: list[tuple[str, str]], conn: Any
) -> list[dict]:
    """
    Fallback competitor search when Overpass API is down.
    Searches planet_osm_point for matching amenity/shop/leisure tags.
    Returns list of {name, lat, lng} dicts.
    """
    radius_m = radius_km * 1000
    results = []

    for tag_key, tag_value in tags:
        sql = f"""
            SELECT name,
                   ST_Y(ST_Transform(way, 4326)) AS lat,
                   ST_X(ST_Transform(way, 4326)) AS lng
            FROM planet_osm_point
            WHERE {tag_key} = %s
              AND ST_DWithin(
                  way::geography,
                  ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
                  %s
              )
            LIMIT 50;
        """
        rows = _execute_query(conn, sql, (tag_value, lng, lat, radius_m))
        for row in rows:
            results.append({
                "name": row[0] or "Unnamed",
                "lat": round(row[1], 6),
                "lng": round(row[2], 6),
            })

    return results
