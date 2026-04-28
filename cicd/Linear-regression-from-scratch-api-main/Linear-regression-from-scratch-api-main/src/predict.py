"""
predict.py
----------
Loads the trained model and scaler from disk and provides
a clean predict() function for use by the API and tests.

WHY SEPARATE FROM train.py?
  - Training is done ONCE (or periodically)
  - Prediction happens for EVERY request to the API
  - Separating them keeps each file focused on one job (Single Responsibility)
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.linear_regression import LinearRegressionScratch

# ── Paths ────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

# ── Lazy loading (load once, reuse for all requests) ─────────────────
_model  = None
_scaler = None


def _load_model():
    """Load model weights from disk (done once at startup)."""
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        _model = LinearRegressionScratch()
        _model.load(MODEL_PATH)

        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
    return _model, _scaler


def predict_price(
    size_sqft: float,
    bedrooms: int,
    age_years: float,
    distance_km: float,
) -> dict:
    """
    Predict house price from raw (unscaled) feature values.

    Parameters
    ----------
    size_sqft    : House size in square feet  (e.g. 1500)
    bedrooms     : Number of bedrooms         (e.g. 3)
    age_years    : Age of house               (e.g. 10)
    distance_km  : Distance from city centre  (e.g. 5.0)

    Returns
    -------
    dict with:
      predicted_price_usd : float — the predicted price
      inputs              : dict  — echo back what was sent
    """
    model, scaler = _load_model()

    # Build feature array (1 row × 4 features)
    X = np.array([[size_sqft, bedrooms, age_years, distance_km]], dtype=float)

    # Scale features using training statistics
    X_scaled = scaler.transform(X)

    # Predict (returns scaled value)
    y_pred_scaled = model.predict(X_scaled)

    # Convert back to USD
    y_pred_usd = scaler.inverse_transform_target(y_pred_scaled)
    price = float(np.clip(y_pred_usd[0], 0, None))   # no negative prices

    return {
        "predicted_price_usd": round(price, 2),
        "inputs": {
            "size_sqft"  : size_sqft,
            "bedrooms"   : bedrooms,
            "age_years"  : age_years,
            "distance_km": distance_km,
        },
    }


def get_model_info() -> dict:
    """Return model weights and bias for the /info endpoint."""
    model, scaler = _load_model()
    feature_names = ["size_sqft", "bedrooms", "age_years", "distance_km"]
    return {
        "weights": {n: round(float(w), 6) for n, w in zip(feature_names, model.weights)},
        "bias"   : round(float(model.bias), 6),
        "feature_means": {n: round(float(m), 4) for n, m in zip(feature_names, scaler.feature_means)},
        "feature_stds" : {n: round(float(s), 4) for n, s in zip(feature_names, scaler.feature_stds)},
    }


if __name__ == "__main__":
    # Quick sanity check
    result = predict_price(1500, 3, 10, 5.0)
    print(f"Sample prediction: ${result['predicted_price_usd']:,.2f}")
