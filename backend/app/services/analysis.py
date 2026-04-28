"""
Analysis service — orchestration layer between API endpoints and ML engine.
Combines feature extraction, scoring, isochrone, and H3 grid generation
into cohesive workflows.
"""

import logging
from backend.ml_engine.core.feature_extractor import extract_features
from backend.ml_engine.core.scoring import compute_score, weighted_formula_score
from backend.ml_engine.core.weights import SUPPORTED_BUSINESS_TYPES, FEATURE_NAMES
from backend.ml_engine.spatial.isochrone_client import estimate_catchment_population
from backend.ml_engine.spatial.h3_grid import (
    generate_hex_grid, score_hex_grid,
    find_clusters, generate_heatmap_points,
)

logger = logging.getLogger(__name__)

# ── City bounding boxes for heatmap endpoint ─────────────────
CITY_BBOXES = {
    "ahmedabad": (22.95, 72.50, 23.10, 72.65),
    "mumbai": (18.90, 72.77, 19.28, 72.98),
    "delhi": (28.50, 76.95, 28.78, 77.35),
    "bangalore": (12.85, 77.50, 13.10, 77.70),
    "hyderabad": (17.30, 78.35, 17.50, 78.55),
    "chennai": (12.95, 80.15, 13.15, 80.30),
    "kolkata": (22.45, 88.28, 22.65, 88.45),
    "pune": (18.45, 73.80, 18.60, 73.95),
    "jaipur": (26.82, 75.72, 26.98, 75.88),
    "surat": (21.13, 72.78, 21.23, 72.88),
}


class AnalysisService:
    """Stateless service that orchestrates the full analysis pipeline."""

    def analyze_site(self, lat: float, lng: float, business_type: str, radius_km: float) -> dict:
        """
        Full pipeline: extract features → enhanced DBSCAN scoring → catchment.
        Returns a dict matching the enhanced AnalyzeResponse schema.
        """
        # 1. Extract features
        conn = self._get_db_conn()
        features = extract_features(lat, lng, radius_km, business_type, conn=conn)

        # 2. Enhanced score with DBSCAN cluster context
        from backend.ml_engine.core.scoring import enhanced_score
        result = enhanced_score(lat, lng, business_type, radius_km, features, conn=conn)

        # 3. Catchment isochrone
        isochrone_geojson, catchment_pop = estimate_catchment_population(lat, lng, 15)
        result["catchment"] = {
            "isochrone_15min": isochrone_geojson,
            "population_within_15min": catchment_pop,
        }

        self._release_conn(conn)
        return result

    def generate_heatmap(self, business_type: str, city: str = None, bbox: str = None) -> list[dict]:
        """
        Score an entire city's H3 grid. Returns list of scored hex dicts.
        Accepts either a city name or a bounding box string.
        """
        # Resolve bounding box
        if city and city.lower() in CITY_BBOXES:
            lat_min, lng_min, lat_max, lng_max = CITY_BBOXES[city.lower()]
        elif bbox:
            parts = [float(x.strip()) for x in bbox.split(",")]
            lat_min, lng_min, lat_max, lng_max = parts
        else:
            logger.warning("No valid city or bbox provided for heatmap")
            return []

        # Centre point and radius
        center_lat = (lat_min + lat_max) / 2
        center_lng = (lng_min + lng_max) / 2
        # Approximate radius from bbox diagonal
        import math
        dlat = (lat_max - lat_min) * 111
        dlng = (lng_max - lng_min) * 111 * math.cos(math.radians(center_lat))
        radius_km = math.sqrt(dlat**2 + dlng**2) / 2

        hex_ids = generate_hex_grid(center_lat, center_lng, radius_km)
        conn = self._get_db_conn()

        def scorer(h_lat, h_lng, biz):
            feats = extract_features(h_lat, h_lng, 1.0, biz, conn=conn)
            return weighted_formula_score(feats, biz)

        # Cap hexes for performance (heatmap can generate thousands)
        scored = score_hex_grid(hex_ids[:200], business_type, scorer)
        self._release_conn(conn)
        return scored

    def suggest_businesses(self, lat: float, lng: float, radius_km: float) -> list[dict]:
        """
        Find the top 3 most underserved business types at this location.
        Runs underserved_index for all business types, ranks by gap.
        """
        conn = self._get_db_conn()
        results = []

        for biz_type in SUPPORTED_BUSINESS_TYPES:
            feats = extract_features(lat, lng, radius_km, biz_type, conn=conn)
            underserved = feats.get("underserved_index", {})
            results.append({
                "business_type": biz_type,
                "underserved_score": underserved.get("score", 50.0),
                "message": underserved.get("raw_value", ""),
            })

        self._release_conn(conn)
        # Sort by underserved_score descending (higher = more opportunity)
        results.sort(key=lambda x: x["underserved_score"], reverse=True)
        return results[:3]

    # ── DB helpers ───────────────────────────────────────────

    def _get_db_conn(self):
        """Try to get a PostGIS connection. Returns None if unavailable."""
        try:
            from backend.app.infrastructure.database import get_pool
            pool = get_pool()
            return pool.getconn()
        except Exception:
            logger.warning("PostGIS unavailable — spatial queries will be limited.")
            return None

    def _release_conn(self, conn):
        """Return connection to pool if it exists."""
        if conn is not None:
            try:
                from backend.app.infrastructure.database import get_pool
                conn.commit()
                get_pool().putconn(conn)
            except Exception:
                pass
