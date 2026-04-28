"""
Business-type-specific scoring weights.
Each business type has a different priority mix of location factors.
All weights for a given type must sum to 1.0.
"""

# ── Weight configurations per business type ──────────────────
# Keys match the feature names returned by feature_extractor.py

BUSINESS_WEIGHTS: dict[str, dict[str, float]] = {
    "gym": {
        "population_density": 0.25,
        "wealth_index": 0.20,
        "road_accessibility": 0.15,
        "competition_density": 0.20,
        "land_use_suitability": 0.10,
        "flood_risk": 0.05,
        "underserved_index": 0.05,
    },
    "clinic": {
        "population_density": 0.30,
        "wealth_index": 0.10,
        "road_accessibility": 0.25,
        "competition_density": 0.15,
        "land_use_suitability": 0.10,
        "flood_risk": 0.05,
        "underserved_index": 0.05,
    },
    "restaurant": {
        "population_density": 0.20,
        "wealth_index": 0.20,
        "road_accessibility": 0.20,
        "competition_density": 0.15,
        "land_use_suitability": 0.15,
        "flood_risk": 0.05,
        "underserved_index": 0.05,
    },
    "pharmacy": {
        "population_density": 0.30,
        "wealth_index": 0.10,
        "road_accessibility": 0.20,
        "competition_density": 0.15,
        "land_use_suitability": 0.10,
        "flood_risk": 0.05,
        "underserved_index": 0.10,
    },
    "retail_store": {
        "population_density": 0.25,
        "wealth_index": 0.20,
        "road_accessibility": 0.20,
        "competition_density": 0.15,
        "land_use_suitability": 0.10,
        "flood_risk": 0.05,
        "underserved_index": 0.05,
    },
    "ev_charging": {
        "population_density": 0.15,
        "wealth_index": 0.25,
        "road_accessibility": 0.30,
        "competition_density": 0.10,
        "land_use_suitability": 0.10,
        "flood_risk": 0.05,
        "underserved_index": 0.05,
    },
    "warehouse": {
        "population_density": 0.05,
        "wealth_index": 0.05,
        "road_accessibility": 0.35,
        "competition_density": 0.10,
        "land_use_suitability": 0.30,
        "flood_risk": 0.10,
        "underserved_index": 0.05,
    },
    "cafe": {
        "population_density": 0.20,
        "wealth_index": 0.25,
        "road_accessibility": 0.15,
        "competition_density": 0.15,
        "land_use_suitability": 0.15,
        "flood_risk": 0.05,
        "underserved_index": 0.05,
    },
    "school": {
        "population_density": 0.30,
        "wealth_index": 0.10,
        "road_accessibility": 0.20,
        "competition_density": 0.10,
        "land_use_suitability": 0.15,
        "flood_risk": 0.10,
        "underserved_index": 0.05,
    },
    "bank": {
        "population_density": 0.25,
        "wealth_index": 0.15,
        "road_accessibility": 0.20,
        "competition_density": 0.15,
        "land_use_suitability": 0.10,
        "flood_risk": 0.05,
        "underserved_index": 0.10,
    },
}

# ── National benchmarks (India) — people per one establishment ─
# Used to calculate the underserved index

NATIONAL_BENCHMARKS: dict[str, int] = {
    "gym": 8000,
    "restaurant": 500,
    "clinic": 5000,
    "pharmacy": 3000,
    "retail_store": 1000,
    "ev_charging": 25000,
    "warehouse": 50000,
    "cafe": 2000,
    "school": 3000,
    "bank": 5000,
}

# ── Supported business types ─────────────────────────────────
SUPPORTED_BUSINESS_TYPES = list(BUSINESS_WEIGHTS.keys())

# ── Feature names (ordered) ─────────────────────────────────
FEATURE_NAMES = [
    "population_density",
    "wealth_index",
    "road_accessibility",
    "competition_density",
    "land_use_suitability",
    "flood_risk",
    "underserved_index",
]
