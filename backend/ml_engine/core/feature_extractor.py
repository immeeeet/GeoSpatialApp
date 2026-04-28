"""
Central feature extractor — pulls all location features for a lat/lng + radius
and returns a normalized feature dict ready for scoring.
"""

import logging
from backend.ml_engine.core.weights import NATIONAL_BENCHMARKS
from backend.ml_engine.spatial import raster_reader, overpass_client, elevation_client

logger = logging.getLogger(__name__)

# ── Normalization helpers ────────────────────────────────────

# Population density thresholds (people/km²) for 0–100 normalization
POP_DENSITY_MAX = 50000  # Score 100 at this density

# VIIRS nightlight scale: raw values 0–63, map to 0–100
NIGHTLIGHT_MAX = 63.0

# Land use suitability by category (business-type-specific overrides possible)
LAND_USE_SUITABILITY = {
    "commercial": 95, "retail": 90, "mixed": 80,
    "residential": 70, "industrial": 50, "recreation_ground": 40,
    "farmland": 20, "forest": 10, "meadow": 15, "unknown": 50,
}


def _normalize(value: float, max_val: float) -> float:
    """Clamp and scale a value to 0–100."""
    if value < 0:
        return 50.0  # Neutral fallback for missing data
    return min(100.0, (value / max(max_val, 1)) * 100.0)


def _competition_score(competitor_count: int) -> float:
    """Inverse scoring: fewer competitors = higher score. 0 competitors = 95."""
    if competitor_count == 0:
        return 95.0
    if competitor_count <= 2:
        return 80.0
    if competitor_count <= 5:
        return 60.0
    if competitor_count <= 10:
        return 40.0
    return max(10.0, 30.0 - competitor_count)


def _underserved_score(population: float, competitor_count: int, business_type: str) -> tuple[float, str]:
    """
    Compare people-per-establishment ratio against national benchmark.
    Higher ratio (more underserved) = higher score.
    """
    benchmark = NATIONAL_BENCHMARKS.get(business_type, 5000)
    effective_competitors = max(competitor_count, 1)
    ratio = population / effective_competitors

    # Score: if ratio > benchmark, area is underserved (good, score > 70)
    if ratio >= benchmark * 2:
        score = 95.0
    elif ratio >= benchmark:
        score = 75.0
    elif ratio >= benchmark * 0.5:
        score = 50.0
    else:
        score = 25.0

    desc = f"1 {business_type} per {int(ratio):,} people"
    return score, desc


# ── Main extraction function ────────────────────────────────


def extract_features(
    lat: float, lng: float, radius_km: float, business_type: str,
    conn=None,
) -> dict:
    """
    Extract all location features and return a dict with both
    raw values and normalized 0–100 scores.

    Returns:
        {
            "population_density": {"score": 85, "raw_value": "42,000 people/km²"},
            "wealth_index": {"score": 70, "raw_value": "nightlight intensity: 38.4"},
            ...
        }

    Each feature has a "score" (0–100 normalized) and "raw_value" (human string).
    """
    features = {}

    # 1. Population density from GHSL raster
    pop_density = raster_reader.read_population_density(lat, lng, radius_km)
    pop_score = _normalize(pop_density, POP_DENSITY_MAX)
    features["population_density"] = {
        "score": round(pop_score, 1),
        "raw_value": f"{pop_density:,.0f} people/km²" if pop_density >= 0 else "data unavailable",
    }

    # 2. Wealth index from VIIRS nightlights
    nightlight = raster_reader.read_nightlight_intensity(lat, lng, radius_km)
    wealth_score = _normalize(nightlight, NIGHTLIGHT_MAX)
    features["wealth_index"] = {
        "score": round(wealth_score, 1),
        "raw_value": f"nightlight intensity: {nightlight:.1f}" if nightlight >= 0 else "data unavailable",
    }

    # 3. Road accessibility from PostGIS OSM
    if conn is not None:
        from backend.ml_engine.spatial.osm_parser import get_road_score
        road_score, road_desc = get_road_score(lat, lng, radius_km, conn)
    else:
        road_score, road_desc = 50.0, "PostGIS unavailable"
    features["road_accessibility"] = {
        "score": round(road_score, 1),
        "raw_value": road_desc,
    }

    # 4. Competition density from Overpass (or PostGIS fallback)
    competitors = overpass_client.find_competitors(lat, lng, radius_km, business_type)
    comp_count = len(competitors)
    comp_score = _competition_score(comp_count)
    features["competition_density"] = {
        "score": round(comp_score, 1),
        "raw_value": f"{comp_count} competitors within {radius_km}km",
    }

    # 5. Land use suitability from PostGIS OSM
    if conn is not None:
        from backend.ml_engine.spatial.osm_parser import get_land_use
        land_use_cat, land_score = get_land_use(lat, lng, radius_km, conn)
    else:
        land_use_cat, land_score = "unknown", 50.0
    features["land_use_suitability"] = {
        "score": round(land_score, 1),
        "raw_value": f"{land_use_cat} zone",
    }

    # 6. Flood risk from elevation
    flood_score, flood_desc = elevation_client.compute_flood_risk(lat, lng)
    features["flood_risk"] = {
        "score": round(flood_score, 1),
        "raw_value": flood_desc,
    }

    # 7. Underserved index
    area_km2 = 3.14159 * radius_km ** 2
    total_pop = pop_density * area_km2 if pop_density >= 0 else 10000
    underserved_score, underserved_desc = _underserved_score(total_pop, comp_count, business_type)
    features["underserved_index"] = {
        "score": round(underserved_score, 1),
        "raw_value": underserved_desc,
    }

    # Stash raw competitor list + total pop for the response builder
    features["_competitors"] = competitors
    features["_total_population"] = total_pop
    features["_competitor_count"] = comp_count

    return features


def extract_feature_vector(
    lat: float, lng: float, radius_km: float, business_type: str,
    conn=None,
) -> list[float]:
    """
    Extract features as a flat numeric vector (for ML model input).
    Order matches weights.FEATURE_NAMES.
    """
    from backend.ml_engine.core.weights import FEATURE_NAMES
    feats = extract_features(lat, lng, radius_km, business_type, conn)
    return [feats[name]["score"] for name in FEATURE_NAMES]
