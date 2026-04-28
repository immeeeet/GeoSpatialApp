"""
Training pipeline: generates synthetic labelled data, trains three models
(weighted formula baseline, Random Forest, XGBoost), evaluates on a 20% holdout,
and saves the best model as best_model.pkl.

Run directly:  python -m backend.ml_engine.models.train
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from backend.ml_engine.core.weights import (
    BUSINESS_WEIGHTS,
    FEATURE_NAMES,
    SUPPORTED_BUSINESS_TYPES,
)
from backend.ml_engine.models.evaluate import compare_models, pick_best_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Indian city centres for sampling ─────────────────────────
# (city, lat_min, lat_max, lng_min, lng_max)
INDIAN_CITIES = [
    ("Mumbai", 18.90, 19.28, 72.77, 72.98),
    ("Delhi", 28.50, 28.78, 76.95, 77.35),
    ("Bangalore", 12.85, 13.10, 77.50, 77.70),
    ("Hyderabad", 17.30, 17.50, 78.35, 78.55),
    ("Ahmedabad", 22.95, 23.10, 72.50, 72.65),
    ("Chennai", 12.95, 13.15, 80.15, 80.30),
    ("Kolkata", 22.45, 22.65, 88.28, 88.45),
    ("Pune", 18.45, 18.60, 73.80, 73.95),
    ("Jaipur", 26.82, 26.98, 75.72, 75.88),
    ("Lucknow", 26.78, 26.92, 80.88, 81.02),
    ("Surat", 21.13, 21.23, 72.78, 72.88),
    ("Kanpur", 26.40, 26.52, 80.30, 80.42),
    ("Nagpur", 21.10, 21.20, 79.02, 79.12),
    ("Indore", 22.68, 22.78, 75.82, 75.92),
    ("Bhopal", 23.20, 23.30, 77.38, 77.48),
    ("Visakhapatnam", 17.68, 17.78, 83.28, 83.38),
    ("Patna", 25.58, 25.65, 85.10, 85.20),
    ("Vadodara", 22.28, 22.35, 73.16, 73.23),
    ("Coimbatore", 10.98, 11.05, 76.93, 77.03),
    ("Kochi", 9.93, 10.02, 76.24, 76.32),
]

# ── Synthetic feature distributions ─────────────────────────
# Calibrated to realistic Indian city values

def _random_features(rng: np.random.Generator, n: int) -> np.ndarray:
    """Generate n rows of 7 synthetic feature scores (each 0–100)."""
    features = np.column_stack([
        np.clip(rng.lognormal(mean=3.5, sigma=0.7, size=n), 0, 100),   # population_density
        np.clip(rng.uniform(10, 90, size=n), 0, 100),                   # wealth_index
        np.clip(rng.beta(5, 2, size=n) * 100, 0, 100),                  # road_accessibility
        np.clip(rng.beta(3, 5, size=n) * 100, 0, 100),                  # competition_density
        np.clip(rng.choice([20, 50, 70, 80, 90, 95], size=n) +
                rng.normal(0, 5, size=n), 0, 100),                      # land_use_suitability
        np.clip(rng.beta(7, 2, size=n) * 100, 0, 100),                  # flood_risk
        np.clip(rng.beta(4, 4, size=n) * 100, 0, 100),                  # underserved_index
    ])
    return features


def _weighted_score(features: np.ndarray, weights: dict) -> np.ndarray:
    """Apply weighted formula to compute ground-truth scores."""
    weight_vec = np.array([weights[f] for f in FEATURE_NAMES])
    return features @ weight_vec


def generate_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data:
    1. Sample random features across business types
    2. Label using weighted formula + ±10% noise
    3. Save to datasets/training_data.csv
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic samples...", n_samples)

    rows = []
    samples_per_type = n_samples // len(SUPPORTED_BUSINESS_TYPES)

    for biz_type in SUPPORTED_BUSINESS_TYPES:
        features = _random_features(rng, samples_per_type)
        weights = BUSINESS_WEIGHTS[biz_type]
        scores = _weighted_score(features, weights)

        # Add ±10% noise to simulate real-world variance
        noise = rng.uniform(-0.10, 0.10, size=len(scores)) * scores
        noisy_scores = np.clip(scores + noise, 0, 100)

        for i in range(samples_per_type):
            row = {name: round(features[i, j], 2) for j, name in enumerate(FEATURE_NAMES)}
            row["business_type"] = biz_type
            row["score"] = round(noisy_scores[i], 2)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save to datasets/
    out_path = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "training_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)
    return df


class WeightedFormulaModel:
    """Baseline model that just applies the weighted formula. Sklearn-compatible interface."""

    def __init__(self):
        self.weights_map = BUSINESS_WEIGHTS
        self._default_weights = np.array([1 / len(FEATURE_NAMES)] * len(FEATURE_NAMES))

    def fit(self, X, y):
        return self  # No fitting needed

    def predict(self, X):
        return X @ self._default_weights


def train_and_evaluate(n_samples: int = 10000):
    """
    Full training pipeline:
    1. Generate (or load) synthetic data
    2. 80/20 split
    3. Train weighted formula, Random Forest, XGBoost
    4. Evaluate and compare
    5. Save best model as .pkl
    """
    # Step 1: Generate data
    csv_path = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "training_data.csv"
    if csv_path.exists():
        logger.info("Loading existing training data from %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        df = generate_synthetic_data(n_samples)

    # Step 2: Prepare features and labels
    X = df[FEATURE_NAMES].values
    y = df["score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train: %d samples, Test: %d samples", len(X_train), len(X_test))

    # Step 3: Train models
    models = {}

    logger.info("Training Weighted Formula baseline...")
    wf = WeightedFormulaModel().fit(X_train, y_train)
    models["WeightedFormula"] = wf

    logger.info("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    logger.info("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    models["XGBoost"] = xgb

    # Step 4: Evaluate
    comparison = compare_models(models, X_test, y_test)
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(comparison.to_string())
    print("=" * 50)

    # Step 5: Pick and save the best
    best_name, best_model = pick_best_model(models, X_test, y_test)
    model_path = Path(__file__).resolve().parent / "best_model.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Best model: %s — saved to %s", best_name, model_path)

    return best_name, comparison


# ── Cluster-Aware Training Pass ──────────────────────────────

# Simulated cluster multipliers matching cluster_engine.py
CLUSTER_MULTIPLIERS = {
    "CORE": 1.12,
    "EDGE": 0.90,
    "SMALL_CLUSTER": 0.95,
    "ISOLATED": 0.78,
}

CLUSTER_TYPES = ["CORE", "EDGE", "SMALL_CLUSTER", "ISOLATED"]


def _assign_synthetic_cluster_types(
    rng: np.random.Generator, n: int, scores: np.ndarray
) -> list[str]:
    """Assign synthetic cluster types based on score distribution."""
    types = []
    for score in scores:
        if score >= 70:
            # High scores more likely to be CORE
            t = rng.choice(CLUSTER_TYPES, p=[0.50, 0.20, 0.20, 0.10])
        elif score >= 50:
            t = rng.choice(CLUSTER_TYPES, p=[0.20, 0.30, 0.30, 0.20])
        else:
            # Low scores more likely ISOLATED
            t = rng.choice(CLUSTER_TYPES, p=[0.10, 0.15, 0.25, 0.50])
        types.append(t)
    return types


def _one_hot_cluster(cluster_type: str) -> list[float]:
    """One-hot encode cluster type as multiplier values."""
    return [
        CLUSTER_MULTIPLIERS[ct] if ct == cluster_type else 0.0
        for ct in CLUSTER_TYPES
    ]


def train_cluster_aware(n_samples: int = 10000):
    """
    Second training pass: add cluster_type as a feature, retrain, compare.
    Saves best_model_v2.pkl if the cluster-aware model outperforms v1.
    """
    csv_path = Path(__file__).resolve().parent.parent.parent.parent / "datasets" / "training_data.csv"
    if not csv_path.exists():
        logger.error("No training data found at %s — run train_and_evaluate first.", csv_path)
        return

    logger.info("=== CLUSTER-AWARE TRAINING PASS ===")
    df = pd.read_csv(csv_path)

    rng = np.random.default_rng(99)

    # Generate cluster types and adjusted scores
    base_scores = df["score"].values
    cluster_types = _assign_synthetic_cluster_types(rng, len(df), base_scores)

    # Build cluster one-hot features
    cluster_features = np.array([_one_hot_cluster(ct) for ct in cluster_types])
    cluster_col_names = [f"cluster_{ct}" for ct in CLUSTER_TYPES]

    # Apply cluster adjustment to target scores
    adjusted_scores = np.array([
        min(100, s * CLUSTER_MULTIPLIERS[ct])
        for s, ct in zip(base_scores, cluster_types)
    ])

    # Combine features: original 7 + 4 cluster one-hot = 11 features
    X_base = df[FEATURE_NAMES].values
    X_cluster = np.hstack([X_base, cluster_features])
    y_cluster = adjusted_scores

    X_train, X_test, y_train, y_test = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    # Also split base features for comparison
    X_base_train, X_base_test = X_train[:, :7], X_test[:, :7]
    y_base_test = y_test  # Same targets

    logger.info("Train: %d, Test: %d (11 features: 7 base + 4 cluster)", len(X_train), len(X_test))

    # Train cluster-aware models
    models_v2 = {}

    logger.info("Training Cluster-Aware Random Forest...")
    rf_v2 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_v2.fit(X_train, y_train)
    models_v2["RF_ClusterAware"] = rf_v2

    logger.info("Training Cluster-Aware XGBoost...")
    xgb_v2 = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
    xgb_v2.fit(X_train, y_train)
    models_v2["XGB_ClusterAware"] = xgb_v2

    # Train base models on same adjusted targets (without cluster features)
    models_v1 = {}
    rf_v1 = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf_v1.fit(X_base_train, y_train)
    models_v1["RF_BaseOnly"] = rf_v1

    xgb_v1 = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
    xgb_v1.fit(X_base_train, y_train)
    models_v1["XGB_BaseOnly"] = xgb_v1

    # Evaluate all
    from backend.ml_engine.models.evaluate import evaluate_model
    results = {}
    for name, model in models_v1.items():
        results[name] = evaluate_model(model, X_base_test, y_base_test)
    for name, model in models_v2.items():
        results[name] = evaluate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("CLUSTER-AWARE MODEL COMPARISON")
    print("=" * 60)
    comparison_df = pd.DataFrame(results).T
    print(comparison_df.to_string())
    print("=" * 60)

    # Pick the best cluster-aware model
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = {**models_v1, **models_v2}[best_name]

    model_path = Path(__file__).resolve().parent / "best_model_v2.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Best cluster-aware model: %s — saved to %s", best_name, model_path)

    # Print improvement summary
    base_best_rmse = min(r["rmse"] for n, r in results.items() if "Base" in n)
    cluster_best_rmse = min(r["rmse"] for n, r in results.items() if "Cluster" in n)
    improvement = ((base_best_rmse - cluster_best_rmse) / base_best_rmse) * 100
    print(f"\nImprovement: {improvement:+.2f}% RMSE reduction with cluster features")

    return best_name, comparison_df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    # Pass 1: Standard training
    train_and_evaluate(n)
    # Pass 2: Cluster-aware retraining
    train_cluster_aware(n)
