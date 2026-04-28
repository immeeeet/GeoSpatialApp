"""
H3 hexagonal grid generation, scoring, and DBSCAN clustering.
Uses Uber's H3 library for spatial indexing and sklearn DBSCAN
to find clusters of high-scoring vs saturated zones.
"""

import logging
import numpy as np
import h3
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# Default H3 resolution 8 ≈ 0.46 km² per hex (good for city-scale analysis)
DEFAULT_RESOLUTION = 8


def generate_hex_grid(lat: float, lng: float, radius_km: float, resolution: int = DEFAULT_RESOLUTION) -> list[str]:
    """Generate H3 hex IDs covering a circular area around the point."""
    # Convert radius to approximate degrees for the bounding ring
    radius_deg = radius_km / 111.0
    # Get the center hex
    center_hex = h3.latlng_to_cell(lat, lng, resolution)
    # Use k-ring to get surrounding hexes
    # Approximate k from radius: each hex at res 8 ≈ 0.74 km edge length
    edge_km = h3.average_hexagon_edge_length(resolution, unit="km")
    k = max(1, int(radius_km / (edge_km * 1.5)))
    hexes = h3.grid_disk(center_hex, k)
    return list(hexes)


def get_hex_center(h3_index: str) -> tuple[float, float]:
    """Return (lat, lng) center of an H3 cell."""
    lat, lng = h3.cell_to_latlng(h3_index)
    return round(lat, 6), round(lng, 6)


def score_hex_grid(hex_ids: list[str], business_type: str, scorer_fn) -> list[dict]:
    """
    Score each hex center using the provided scorer function.
    scorer_fn(lat, lng, business_type) -> float (0-100).
    Returns list of {h3_index, score, center: [lat, lng]}.
    """
    scored = []
    for h3_id in hex_ids:
        lat, lng = get_hex_center(h3_id)
        try:
            score = scorer_fn(lat, lng, business_type)
        except Exception as exc:
            logger.warning("Scoring failed for hex %s: %s", h3_id, exc)
            score = 0.0
        scored.append({
            "h3_index": h3_id,
            "score": round(score, 1),
            "center": [lat, lng],
        })
    return scored


def find_clusters(scored_hexes: list[dict], min_score: float = 60.0) -> list[dict]:
    """
    Run DBSCAN on high-scoring hexes to find opportunity clusters.
    Returns cluster labels added to each hex dict.
    """
    high_score = [h for h in scored_hexes if h["score"] >= min_score]
    if len(high_score) < 3:
        for h in high_score:
            h["cluster"] = 0
        return high_score

    coords = np.array([[h["center"][0], h["center"][1]] for h in high_score])
    # eps in degrees: ~0.005 ≈ 500m
    db = DBSCAN(eps=0.005, min_samples=2, metric="haversine")
    # DBSCAN with haversine expects radians
    coords_rad = np.radians(coords)
    labels = db.fit_predict(coords_rad)

    for h, label in zip(high_score, labels):
        h["cluster"] = int(label)

    n_clusters = len(set(labels) - {-1})
    logger.info("DBSCAN found %d opportunity clusters from %d high-score hexes", n_clusters, len(high_score))
    return high_score


def generate_heatmap_points(scored_hexes: list[dict]) -> list[dict]:
    """Convert scored hexes to {lat, lng, intensity} for frontend heatmap layer."""
    return [
        {
            "lat": h["center"][0],
            "lng": h["center"][1],
            "intensity": round(h["score"] / 100.0, 3),
        }
        for h in scored_hexes
    ]
