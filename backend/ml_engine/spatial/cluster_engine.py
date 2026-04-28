"""
DBSCAN Spatial Intelligence Layer.

Scores every H3 hexagon in an analysis radius, clusters them with DBSCAN
to find spatial patterns (core markets, edges, isolated spots), and adjusts
the raw score based on market context. This turns a single-pin score into
a market-aware intelligence layer.
"""

import logging
import math
import time

import h3
import numpy as np
from sklearn.cluster import DBSCAN

from backend.app.infrastructure.cache import cache
from backend.ml_engine.core.feature_extractor import extract_features
from backend.ml_engine.core.scoring import weighted_formula_score
from backend.ml_engine.core.weights import FEATURE_NAMES

logger = logging.getLogger(__name__)

# Hex cache TTL — 48 hours for individual hex scores
HEX_CACHE_TTL = 48 * 3600

# DBSCAN tuning constants
DBSCAN_EPS_KM = 0.3           # Cluster radius in km
DBSCAN_MIN_SAMPLES = 5        # Minimum hexes to form a cluster
EARTH_RADIUS_KM = 6371.0      # For radian conversion

# Cluster size threshold for CORE vs SMALL_CLUSTER
CORE_CLUSTER_MIN_SIZE = 20

# Score adjustment multipliers per cluster type
CLUSTER_ADJUSTMENTS = {
    "CORE": 1.12,
    "EDGE": 0.90,
    "SMALL_CLUSTER": 0.95,
    "ISOLATED": 0.78,
}

# Color coding for map rendering
SCORE_COLORS = {
    "high": "#22c55e",     # green  (80-100)
    "medium": "#eab308",   # yellow (60-79)
    "low": "#f97316",      # orange (40-59)
    "very_low": "#ef4444", # red    (0-39)
    "noise": "#6b7280",    # gray
}


# ── Function 1: Generate H3 analysis grid ────────────────────


def generate_analysis_grid(
    lat: float, lng: float, radius_km: float, resolution: int = 9
) -> list[tuple[str, float, float]]:
    """Generate all H3 hexagons within the radius at given resolution."""
    center_hex = h3.latlng_to_cell(lat, lng, resolution)
    edge_km = h3.average_hexagon_edge_length(resolution, unit="km")
    k = max(1, int(radius_km / (edge_km * 1.5)))
    hex_ids = h3.grid_disk(center_hex, k)

    grid = []
    for h3_id in hex_ids:
        clat, clng = h3.cell_to_latlng(h3_id)
        grid.append((h3_id, round(clat, 6), round(clng, 6)))

    logger.info(
        "Generated %d hexes at res %d for (%.4f, %.4f) r=%.1fkm",
        len(grid), resolution, lat, lng, radius_km,
    )
    return grid


# ── Function 2: Score hex grid with Redis caching ────────────


def _score_single_hex(
    h3_id: str, lat: float, lng: float, business_type: str, conn=None
) -> float:
    """Score a single hex — checks cache first, computes on miss."""
    cache_key = f"hex_{h3_id}_{business_type}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached.get("score", 50.0)

    # Extract features and compute score via weighted formula
    features = extract_features(lat, lng, 0.5, business_type, conn=conn)
    score = weighted_formula_score(features, business_type)

    # Cache with 48hr TTL
    cache.set(cache_key, {"score": score}, ttl=HEX_CACHE_TTL)
    return score


def score_hex_grid(
    hex_grid: list[tuple[str, float, float]], business_type: str, conn=None
) -> list[dict]:
    """Score every hex in the grid, using Redis cache to skip already-scored hexes."""
    t0 = time.time()
    cached_count = 0
    scored = []

    for h3_id, lat, lng in hex_grid:
        # Check cache first
        cache_key = f"hex_{h3_id}_{business_type}"
        cached_val = cache.get(cache_key)

        if cached_val is not None:
            score = cached_val.get("score", 50.0)
            cached_count += 1
        else:
            score = _score_single_hex(h3_id, lat, lng, business_type, conn)

        scored.append({
            "h3_index": h3_id,
            "score": round(score, 1),
            "center_lat": lat,
            "center_lng": lng,
        })

    elapsed = time.time() - t0
    logger.info(
        "Scored %d hexes in %.1fs (%d from cache, %d computed)",
        len(scored), elapsed, cached_count, len(scored) - cached_count,
    )
    return scored


# ── Function 3: Run DBSCAN clustering ────────────────────────


def _prepare_spatial_coords(scored_hexes: list[dict]) -> np.ndarray:
    """Build 2D spatial matrix in radians: [lat_rad, lng_rad] for haversine."""
    return np.array([
        [math.radians(h["center_lat"]), math.radians(h["center_lng"])]
        for h in scored_hexes
    ])


def run_dbscan(scored_hexes: list[dict], score_threshold: float = 40.0) -> list[dict]:
    """
    Run DBSCAN on scored hexes using haversine distance on 2D lat/lng.
    Only hexes above score_threshold participate in clustering —
    low-scoring hexes are pre-labeled as noise (-1).
    """
    if len(scored_hexes) < DBSCAN_MIN_SAMPLES:
        for h in scored_hexes:
            h["cluster_label"] = -1
        return scored_hexes

    t0 = time.time()

    # Split into clusterable (above threshold) and noise (below)
    above = [(i, h) for i, h in enumerate(scored_hexes) if h["score"] >= score_threshold]
    below_idxs = {i for i, h in enumerate(scored_hexes) if h["score"] < score_threshold}

    # Pre-label low-scoring hexes as noise
    for i in below_idxs:
        scored_hexes[i]["cluster_label"] = -1

    if len(above) < DBSCAN_MIN_SAMPLES:
        for i, _ in above:
            scored_hexes[i]["cluster_label"] = -1
        return scored_hexes

    # Build 2D coords for haversine (radians)
    coords = np.array([
        [math.radians(h["center_lat"]), math.radians(h["center_lng"])]
        for _, h in above
    ])

    eps_rad = DBSCAN_EPS_KM / EARTH_RADIUS_KM

    db = DBSCAN(
        eps=eps_rad,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = db.fit_predict(coords)

    # Map labels back to original hex list
    for (orig_idx, _), label in zip(above, labels):
        scored_hexes[orig_idx]["cluster_label"] = int(label)

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1)) + len(below_idxs)
    elapsed = time.time() - t0

    logger.info(
        "DBSCAN: %d clusters, %d noise points (%.2fs)",
        n_clusters, n_noise, elapsed,
    )
    return scored_hexes


# ── Function 4: Classify the queried point ───────────────────


def _is_boundary_hex(
    hex_idx: int, cluster_label: int, clustered_hexes: list[dict]
) -> bool:
    """Check if a hex is on the cluster boundary (has noise neighbours)."""
    target = clustered_hexes[hex_idx]
    target_h3 = target["h3_index"]
    neighbors = h3.grid_disk(target_h3, 1)
    neighbor_ids = {n for n in neighbors if n != target_h3}

    for h in clustered_hexes:
        if h["h3_index"] in neighbor_ids and h["cluster_label"] == -1:
            return True
    return False


def classify_queried_point(
    lat: float, lng: float, clustered_hexes: list[dict]
) -> dict:
    """Determine which cluster the queried lat/lng falls into and classify it."""
    # Find the hex containing the queried point
    queried_h3 = h3.latlng_to_cell(lat, lng, 9)
    queried_hex = None
    queried_idx = None

    for i, h in enumerate(clustered_hexes):
        if h["h3_index"] == queried_h3:
            queried_hex = h
            queried_idx = i
            break

    # Fallback: find nearest hex if exact match not found
    if queried_hex is None and clustered_hexes:
        min_dist = float("inf")
        for i, h in enumerate(clustered_hexes):
            d = math.sqrt(
                (h["center_lat"] - lat) ** 2 + (h["center_lng"] - lng) ** 2
            )
            if d < min_dist:
                min_dist = d
                queried_hex = h
                queried_idx = i

    if queried_hex is None:
        return _empty_classification()

    label = queried_hex["cluster_label"]

    # Count neighboring clusters (other cluster IDs adjacent to this point)
    neighbor_h3s = h3.grid_disk(queried_hex["h3_index"], 2)
    neighbor_labels = set()
    for h in clustered_hexes:
        if h["h3_index"] in neighbor_h3s and h["cluster_label"] >= 0:
            neighbor_labels.add(h["cluster_label"])
    if label in neighbor_labels:
        neighbor_labels.remove(label)

    if label == -1:
        return _build_classification("ISOLATED", 0, 0.0, len(neighbor_labels))

    # Count cluster members and compute avg score
    cluster_members = [h for h in clustered_hexes if h["cluster_label"] == label]
    cluster_size = len(cluster_members)
    avg_score = sum(h["score"] for h in cluster_members) / max(cluster_size, 1)

    # Classify based on size and boundary status
    if cluster_size > CORE_CLUSTER_MIN_SIZE:
        is_boundary = _is_boundary_hex(queried_idx, label, clustered_hexes)
        if is_boundary:
            ctype = "EDGE"
        else:
            ctype = "CORE"
    else:
        ctype = "SMALL_CLUSTER"

    return _build_classification(ctype, cluster_size, avg_score, len(neighbor_labels))


def _build_classification(
    cluster_type: str, size: int, avg_score: float, neighbor_count: int
) -> dict:
    """Build a cluster classification result dict."""
    return {
        "cluster_type": cluster_type,
        "cluster_size": size,
        "cluster_avg_score": round(avg_score, 1),
        "neighboring_cluster_count": neighbor_count,
    }


def _empty_classification() -> dict:
    """Return a default classification when no hex data is available."""
    return _build_classification("ISOLATED", 0, 0.0, 0)


# ── Function 5: Adjust score based on cluster context ────────


def adjust_score_for_cluster(
    raw_score: float, cluster_classification: dict
) -> tuple[float, str]:
    """Apply cluster-based adjustment to the raw model score."""
    ctype = cluster_classification["cluster_type"]
    multiplier = CLUSTER_ADJUSTMENTS.get(ctype, 1.0)
    adjusted = min(100.0, raw_score * multiplier)
    delta = round(adjusted - raw_score, 1)

    reasons = {
        "CORE": (
            f"+{delta} points — you're in the core of a strong market cluster "
            f"({cluster_classification['cluster_size']} high-scoring hexes nearby)"
        ),
        "EDGE": (
            f"{delta} points — you're at the edge of a market cluster, "
            f"moderate confidence in sustained demand"
        ),
        "SMALL_CLUSTER": (
            f"{delta} points — small market cluster detected, "
            f"some local demand but limited depth"
        ),
        "ISOLATED": (
            f"{delta} points — isolated high-potential spot with no market "
            f"depth nearby, higher risk"
        ),
    }

    reason = reasons.get(ctype, f"{delta} points — cluster adjustment applied")
    return round(adjusted, 1), reason


# ── Function 6: City-wide opportunity zones ──────────────────


def _get_city_bbox(city: str) -> tuple[float, float, float, float] | None:
    """Look up a city's bounding box. Returns (lat_min, lng_min, lat_max, lng_max)."""
    from backend.app.services.analysis import CITY_BBOXES
    return CITY_BBOXES.get(city.lower())


def _compute_bbox_center_radius(
    lat_min: float, lng_min: float, lat_max: float, lng_max: float
) -> tuple[float, float, float]:
    """Compute center point and approximate radius from a bounding box."""
    center_lat = (lat_min + lat_max) / 2
    center_lng = (lng_min + lng_max) / 2
    dlat = (lat_max - lat_min) * 111
    dlng = (lng_max - lng_min) * 111 * math.cos(math.radians(center_lat))
    radius_km = math.sqrt(dlat ** 2 + dlng ** 2) / 2
    return center_lat, center_lng, radius_km


def _rank_clusters(clustered_hexes: list[dict]) -> list[dict]:
    """Group hexes by cluster label, rank by average score, return top 5."""
    clusters = {}
    for h in clustered_hexes:
        label = h["cluster_label"]
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(h)

    ranked = []
    for label, members in clusters.items():
        avg_score = sum(m["score"] for m in members) / len(members)
        center_lat = sum(m["center_lat"] for m in members) / len(members)
        center_lng = sum(m["center_lng"] for m in members) / len(members)

        # Build GeoJSON for cluster hexes
        features = []
        for m in members:
            boundary = h3.cell_to_boundary(m["h3_index"])
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])  # close the ring
            features.append({
                "type": "Feature",
                "properties": {
                    "h3_index": m["h3_index"],
                    "score": m["score"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })

        ranked.append({
            "cluster_id": int(label),
            "avg_score": round(avg_score, 1),
            "hex_count": len(members),
            "center_lat": round(center_lat, 6),
            "center_lng": round(center_lng, 6),
            "geojson": {
                "type": "FeatureCollection",
                "features": features,
            },
        })

    ranked.sort(key=lambda c: c["avg_score"], reverse=True)
    return ranked[:5]


def get_opportunity_zones(
    city: str, business_type: str, bbox: str = None
) -> list[dict]:
    """Score a full city grid at resolution 8, cluster, return top 5 zones."""
    t0 = time.time()

    # Resolve bounding box
    if city:
        bbox_tuple = _get_city_bbox(city)
        if bbox_tuple is None:
            logger.warning("Unknown city: %s", city)
            return []
        lat_min, lng_min, lat_max, lng_max = bbox_tuple
    elif bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        lat_min, lng_min, lat_max, lng_max = parts
    else:
        return []

    center_lat, center_lng, radius_km = _compute_bbox_center_radius(
        lat_min, lng_min, lat_max, lng_max
    )

    # Resolution 8 for city-wide (~500m per hex)
    grid = generate_analysis_grid(center_lat, center_lng, radius_km, resolution=8)
    scored = score_hex_grid(grid, business_type)
    clustered = run_dbscan(scored)
    zones = _rank_clusters(clustered)

    elapsed = time.time() - t0
    logger.info(
        "Opportunity zones for %s/%s: %d zones found in %.1fs",
        city or bbox, business_type, len(zones), elapsed,
    )
    return zones


# ── Hex color helper ─────────────────────────────────────────


def hex_color(score: float, cluster_label: int) -> str:
    """Map a hex score + cluster label to a display color."""
    if cluster_label == -1:
        return SCORE_COLORS["noise"]
    if score >= 80:
        return SCORE_COLORS["high"]
    if score >= 60:
        return SCORE_COLORS["medium"]
    if score >= 40:
        return SCORE_COLORS["low"]
    return SCORE_COLORS["very_low"]
