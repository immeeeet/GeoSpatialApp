"""
Score computation engine.
Loads the trained ML model (or falls back to weighted formula),
runs prediction, and builds the full structured response.
"""

import logging
from pathlib import Path

import joblib
import numpy as np

from backend.ml_engine.core.weights import BUSINESS_WEIGHTS, FEATURE_NAMES
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Cached model ─────────────────────────────────────────────
_cached_model = None
_model_loaded = False


def _load_model():
    """Load the best_model.pkl once and cache it. Returns None if not found."""
    global _cached_model, _model_loaded
    if _model_loaded:
        return _cached_model

    model_path = Path(get_settings().MODEL_PATH)
    if model_path.exists():
        try:
            _cached_model = joblib.load(model_path)
            logger.info("Loaded ML model from %s", model_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            _cached_model = None
    else:
        logger.info("No trained model found at %s — using weighted formula.", model_path)
        _cached_model = None

    _model_loaded = True
    return _cached_model


# ── Scoring functions ────────────────────────────────────────


def weighted_formula_score(features: dict, business_type: str) -> float:
    """Baseline: weighted sum of normalized feature scores."""
    weights = BUSINESS_WEIGHTS.get(business_type)
    if weights is None:
        # Equal weights fallback
        weights = {f: 1.0 / len(FEATURE_NAMES) for f in FEATURE_NAMES}

    total = 0.0
    for feat_name in FEATURE_NAMES:
        score = features.get(feat_name, {}).get("score", 50.0)
        weight = weights.get(feat_name, 0.0)
        total += score * weight

    return round(np.clip(total, 0, 100), 1)


def ml_model_score(features: dict) -> float | None:
    """Use the trained ML model to predict a score. Returns None if no model."""
    model = _load_model()
    if model is None:
        return None

    feature_vector = np.array(
        [[features.get(f, {}).get("score", 50.0) for f in FEATURE_NAMES]]
    )
    prediction = model.predict(feature_vector)[0]
    return round(float(np.clip(prediction, 0, 100)), 1)


def assign_grade(score: float) -> str:
    """Map a 0–100 score to a letter grade."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    return "F"


def generate_verdict(score: float, features: dict, business_type: str) -> str:
    """Generate a human-readable verdict summarising the score."""
    grade = assign_grade(score)

    # Find the best and worst factors
    scored_features = {
        k: v["score"] for k, v in features.items() if not k.startswith("_") and "score" in v
    }
    if not scored_features:
        return f"Score: {score} ({grade})"

    best = max(scored_features, key=scored_features.get)
    worst = min(scored_features, key=scored_features.get)

    best_label = best.replace("_", " ")
    worst_label = worst.replace("_", " ")

    if score >= 80:
        tone = "Excellent location"
    elif score >= 65:
        tone = "Good location"
    elif score >= 50:
        tone = "Average location"
    else:
        tone = "Below-average location"

    return f"{tone} with strong {best_label} but moderate {worst_label}"


def _build_demand_gap(features: dict, business_type: str) -> dict:
    """Build the demand_gap section of the response."""
    from backend.ml_engine.core.weights import NATIONAL_BENCHMARKS

    total_pop = features.get("_total_population", 0)
    comp_count = features.get("_competitor_count", 0)
    benchmark = NATIONAL_BENCHMARKS.get(business_type, 5000)

    if comp_count == 0:
        ratio = total_pop
    else:
        ratio = total_pop / comp_count

    if ratio >= benchmark * 1.5:
        status = "HIGH"
        msg = (
            f"Area has {int(total_pop):,} residents with only {comp_count} "
            f"{business_type}s. National benchmark is 1 per {benchmark:,}."
        )
    elif ratio >= benchmark * 0.75:
        status = "MODERATE"
        msg = (
            f"Area has {int(total_pop):,} residents with {comp_count} "
            f"{business_type}s. Slightly underserved compared to benchmark."
        )
    else:
        status = "LOW"
        msg = (
            f"Area appears well-served with {comp_count} {business_type}s "
            f"for {int(total_pop):,} residents."
        )

    return {"status": status, "message": msg}


# ── Main entry point ─────────────────────────────────────────


def compute_score(features: dict, business_type: str) -> dict:
    """
    Main scoring function. Tries ML model first, falls back to weighted formula.
    Returns the full response dict with score, grade, verdict, and breakdown.
    """
    # Try ML model first
    score = ml_model_score(features)
    if score is None:
        score = weighted_formula_score(features, business_type)

    grade = assign_grade(score)
    verdict = generate_verdict(score, features, business_type)

    # Build breakdown with weights
    weights = BUSINESS_WEIGHTS.get(business_type, {})
    breakdown = {}
    for feat_name in FEATURE_NAMES:
        feat_data = features.get(feat_name, {"score": 50, "raw_value": "unavailable"})
        breakdown[feat_name] = {
            "score": feat_data["score"],
            "weight": weights.get(feat_name, 0.0),
            "raw_value": feat_data["raw_value"],
        }

    demand_gap = _build_demand_gap(features, business_type)
    competitors = features.get("_competitors", [])

    return {
        "score": int(round(score)),
        "grade": grade,
        "verdict": verdict,
        "breakdown": breakdown,
        "demand_gap": demand_gap,
        "competitors": competitors,
    }


# ── Enhanced scoring with DBSCAN cluster context ─────────────


def enhanced_score(
    lat: float, lng: float, business_type: str, radius_km: float,
    features: dict, conn=None,
) -> dict:
    """
    Full cluster-aware scoring: raw score + DBSCAN analysis + adjusted score.
    Returns everything needed for the enhanced API response.
    """
    import time
    from backend.ml_engine.spatial.cluster_engine import (
        generate_analysis_grid, score_hex_grid as cluster_score_grid,
        run_dbscan, classify_queried_point, adjust_score_for_cluster,
        hex_color,
    )

    t0 = time.time()

    # 1. Compute raw score using existing logic
    raw_result = compute_score(features, business_type)
    raw_score = raw_result["score"]

    # 2. Generate hex grid, score, and cluster
    grid = generate_analysis_grid(lat, lng, radius_km, resolution=9)
    scored_hexes = cluster_score_grid(grid, business_type, conn=conn)
    clustered_hexes = run_dbscan(scored_hexes)

    # 3. Classify where the queried point sits in the cluster landscape
    classification = classify_queried_point(lat, lng, clustered_hexes)

    # 4. Adjust score based on cluster context
    adjusted_score, adjustment_reason = adjust_score_for_cluster(
        float(raw_score), classification
    )

    # 5. Build colored hex grid for map rendering
    colored_hexes = _build_colored_hexes(clustered_hexes, classification)

    # 6. Build cluster boundaries GeoJSON
    cluster_boundaries = _build_cluster_boundaries(clustered_hexes)

    # 7. Build heatmap points
    heatmap_points = [
        {"lat": h["center_lat"], "lng": h["center_lng"],
         "intensity": round(h["score"] / 100.0, 3)}
        for h in clustered_hexes
    ]

    elapsed = time.time() - t0
    logger.info(
        "Enhanced score for (%.4f, %.4f): raw=%d, adjusted=%d, type=%s (%.1fs)",
        lat, lng, raw_score, adjusted_score, classification["cluster_type"], elapsed,
    )

    # Merge with raw result (keeps breakdown, demand_gap, competitors)
    raw_result["raw_score"] = raw_score
    raw_result["score"] = int(round(adjusted_score))
    raw_result["grade"] = assign_grade(adjusted_score)
    raw_result["verdict"] = generate_verdict(adjusted_score, features, business_type)
    raw_result["cluster"] = {
        "type": classification["cluster_type"],
        "size": classification["cluster_size"],
        "avg_score": classification["cluster_avg_score"],
        "adjustment": adjustment_reason,
        "neighboring_clusters": classification["neighboring_cluster_count"],
    }
    raw_result["map_data"] = {
        "clustered_hexes": colored_hexes,
        "heatmap_points": heatmap_points,
        "cluster_boundaries": cluster_boundaries,
    }
    return raw_result


def _build_colored_hexes(clustered_hexes: list[dict], classification: dict) -> list[dict]:
    """Convert clustered hexes into the colored format for frontend rendering."""
    from backend.ml_engine.spatial.cluster_engine import hex_color

    result = []
    for h in clustered_hexes:
        label = h.get("cluster_label", -1)

        # Determine cluster_type label for this hex
        if label == -1:
            ctype = "NOISE"
        elif classification["cluster_size"] > 20:
            ctype = "CORE"
        else:
            ctype = "SMALL_CLUSTER"

        result.append({
            "h3_index": h["h3_index"],
            "score": h["score"],
            "cluster_label": label,
            "cluster_type": ctype,
            "center": [h["center_lat"], h["center_lng"]],
            "color": hex_color(h["score"], label),
        })
    return result


def _build_cluster_boundaries(clustered_hexes: list[dict]) -> dict:
    """Build GeoJSON FeatureCollection of cluster outlines."""
    import h3 as h3_lib

    clusters = {}
    for h in clustered_hexes:
        label = h.get("cluster_label", -1)
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(h["h3_index"])

    features = []
    for label, hex_ids in clusters.items():
        # Get boundary of each hex and build a multi-polygon
        for hid in hex_ids:
            boundary = h3_lib.cell_to_boundary(hid)
            coords = [[lng, lat] for lat, lng in boundary]
            coords.append(coords[0])
            features.append({
                "type": "Feature",
                "properties": {"cluster_id": int(label)},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            })

    return {"type": "FeatureCollection", "features": features}
