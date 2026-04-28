"""
train.py
--------
Orchestrates the full training pipeline:
  1. Load data
  2. Split into train/test
  3. Scale features
  4. Train Linear Regression from scratch
  5. Evaluate on test set
  6. Save model + scaler to disk

Run this file directly:
    python src/train.py
"""

import os
import sys
import pickle
import numpy as np

# Make sure src/ is in path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from linear_regression import LinearRegressionScratch
from preprocess        import load_data, train_test_split_manual, DataPreprocessor

# ── Paths ────────────────────────────────────────────────────────────
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "linear_regression.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")

# ── Hyperparameters ──────────────────────────────────────────────────
LEARNING_RATE = 0.1
N_ITERATIONS  = 1000
TEST_SIZE     = 0.2


def train() -> dict:
    """
    Full training pipeline. Returns evaluation metrics dict.

    WHAT HAPPENS STEP BY STEP:
    --------------------------
    Step 1  Load the CSV dataset (1000 rows × 5 columns)
    Step 2  Split: 800 rows for training, 200 rows for testing
    Step 3  Fit scaler on training data ONLY (prevents data leakage)
    Step 4  Scale both train and test features
    Step 5  Run gradient descent for N_ITERATIONS
    Step 6  Evaluate: R², RMSE, MAE on test set
    Step 7  Save trained weights + scaler to /models/
    """

    print("\n" + "="*55)
    print("  House Price Prediction — Linear Regression Training")
    print("="*55)

    # ── Step 1: Load ──────────────────────────────────────────────
    print("\n📂  Step 1: Loading data...")
    df, X, y = load_data()
    print(f"    Loaded {len(df)} rows, {X.shape[1]} features")

    # ── Step 2: Split ─────────────────────────────────────────────
    print("\n✂️   Step 2: Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, TEST_SIZE)
    print(f"    Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")

    # ── Step 3 & 4: Scale ─────────────────────────────────────────
    print("\n⚖️   Step 3: Scaling features (Z-score standardisation)...")
    scaler = DataPreprocessor()
    scaler.fit(X_train, y_train)                       # learn from train ONLY
    X_train_sc, y_train_sc = scaler.transform(X_train, y_train)
    X_test_sc,  y_test_sc  = scaler.transform(X_test,  y_test)
    print(f"    Feature means: {np.round(scaler.feature_means, 2)}")
    print(f"    Feature stds : {np.round(scaler.feature_stds, 2)}")

    # ── Step 5: Train ─────────────────────────────────────────────
    print(f"\n🏋️   Step 4: Training ({N_ITERATIONS} iterations, lr={LEARNING_RATE})...")
    model = LinearRegressionScratch(
        learning_rate=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    model.fit(X_train_sc, y_train_sc)

    # ── Step 6: Evaluate ──────────────────────────────────────────
    print("\n📊  Step 5: Evaluating on test set...")

    # Predict on scaled test set, then inverse-transform back to USD
    y_pred_sc  = model.predict(X_test_sc)
    y_pred_usd = scaler.inverse_transform_target(y_pred_sc)

    # Compute metrics on original USD scale for interpretability
    metrics = model.score(X_test_sc, y_test_sc)

    # Also compute RMSE in USD for human-friendly reporting
    rmse_usd = float(np.sqrt(np.mean((y_test - y_pred_usd) ** 2)))
    mae_usd  = float(np.mean(np.abs(y_test - y_pred_usd)))

    print(f"\n    ✅  R²  (higher is better, 1.0 = perfect) : {metrics['r2']}")
    print(f"    ✅  RMSE in USD (lower is better)         : ${rmse_usd:>10,.2f}")
    print(f"    ✅  MAE  in USD (lower is better)         : ${mae_usd:>10,.2f}")
    print(f"\n    Learned Weights (scaled space):")
    feature_names = ["size_sqft", "bedrooms", "age_years", "distance_km"]
    for name, w in zip(feature_names, model.weights):
        direction = "↑ price" if w > 0 else "↓ price"
        print(f"      {name:<15}: {w:>8.4f}  {direction}")
    print(f"      {'bias':<15}: {model.bias:>8.4f}")

    # ── Step 7: Save ──────────────────────────────────────────────
    print(f"\n💾  Step 6: Saving model and scaler...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"    Scaler saved → {SCALER_PATH}")

    # Save metrics for CI/CD pipeline to read
    with open(METRICS_PATH, "w") as f:
        f.write(f"r2={metrics['r2']}\n")
        f.write(f"rmse_usd={rmse_usd:.2f}\n")
        f.write(f"mae_usd={mae_usd:.2f}\n")
    print(f"    Metrics saved → {METRICS_PATH}")

    print("\n" + "="*55)
    print("  Training complete! 🎉")
    print("="*55 + "\n")

    return {
        "r2": metrics["r2"],
        "rmse_usd": round(rmse_usd, 2),
        "mae_usd": round(mae_usd, 2),
    }


if __name__ == "__main__":
    train()
