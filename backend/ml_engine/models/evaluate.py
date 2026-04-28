"""
Model evaluation utilities.
Computes RMSE, R², and MAE for trained models and prints a comparison table.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a single model and return {rmse, r2, mae}."""
    y_pred = model.predict(X_test)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "r2": round(float(r2_score(y_test, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
    }


def compare_models(
    models_dict: dict, X_test: np.ndarray, y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate multiple models side-by-side.
    models_dict: {"model_name": fitted_model, ...}
    Returns a DataFrame with RMSE, R², MAE per model.
    """
    rows = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics["model"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("model")
    df = df[["rmse", "r2", "mae"]]
    return df


def pick_best_model(
    models_dict: dict, X_test: np.ndarray, y_test: np.ndarray
) -> tuple[str, object]:
    """Pick the model with the lowest RMSE. Returns (name, fitted_model)."""
    comparison = compare_models(models_dict, X_test, y_test)
    best_name = comparison["rmse"].idxmin()
    return best_name, models_dict[best_name]
