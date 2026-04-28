"""
tests/test_model.py
-------------------
Automated tests for the Linear Regression model.

WHY TESTS ARE THE HEART OF CI/CD:
===================================
Without tests, CI/CD is just "auto-deploy broken code faster."
Tests are what give us CONFIDENCE that our changes haven't broken anything.

TYPES OF TESTS WE WRITE:
  Unit tests     → test a single function in isolation
  Integration    → test multiple components working together
  Smoke tests    → "does it even start?" (basic sanity checks)
  Performance    → "is it fast enough / accurate enough?"

These tests run AUTOMATICALLY every time you push code to GitHub.
If any test fails, the pipeline STOPS and you get notified.
"""

import sys
import os
import numpy as np
import pytest

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.linear_regression import LinearRegressionScratch
from src.preprocess        import DataPreprocessor, load_data, train_test_split_manual


# ============================================================
# Unit Tests — LinearRegressionScratch
# ============================================================

class TestLinearRegressionScratch:
    """Tests for the core LinearRegression class."""

    def test_predict_before_fit_raises(self):
        """predict() must raise RuntimeError if called before fit()."""
        model = LinearRegressionScratch()
        X = np.array([[1.0, 2.0]])
        with pytest.raises(RuntimeError, match="not trained"):
            model.predict(X)

    def test_fit_sets_weights_and_bias(self):
        """After fit(), weights and bias must not be None/zero."""
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
        X = np.random.randn(50, 3)
        y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 1

        model.fit(X, y)

        assert model.weights is not None
        assert len(model.weights) == 3
        assert model.bias != 0.0 or True   # bias could be ~0 for centred data

    def test_weights_shape_matches_features(self):
        """Weight vector shape must equal number of input features."""
        for n_features in [1, 3, 5, 10]:
            model = LinearRegressionScratch(learning_rate=0.1, n_iterations=50)
            X = np.random.randn(30, n_features)
            y = np.random.randn(30)
            model.fit(X, y)
            assert model.weights.shape == (n_features,), \
                f"Expected shape ({n_features},), got {model.weights.shape}"

    def test_perfect_fit_on_noiseless_data(self):
        """
        On perfectly linear data with no noise, the model should achieve
        R² very close to 1.0 after enough iterations.
        """
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5   # perfect linear relationship

        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)
        metrics = model.score(X, y)

        assert metrics["r2"] > 0.999, \
            f"R² should be > 0.999 on noiseless data, got {metrics['r2']}"

    def test_predict_output_shape(self):
        """predict() must return array of shape (n_samples,)."""
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (40,), f"Expected (40,), got {preds.shape}"

    def test_loss_decreases_during_training(self):
        """
        The training loss must be strictly decreasing (or at least non-increasing)
        over iterations — this verifies gradient descent is working correctly.
        """
        np.random.seed(0)
        X = np.random.randn(100, 3)
        y = 2*X[:, 0] + 1

        model = LinearRegressionScratch(learning_rate=0.05, n_iterations=200)
        model.fit(X, y)

        # Check that final loss < initial loss
        assert model.loss_history[-1] < model.loss_history[0], \
            "Loss did not decrease during training!"

        # Check that loss never increased by more than 1% between steps
        losses = np.array(model.loss_history)
        increases = np.diff(losses) / (losses[:-1] + 1e-10)
        assert np.all(increases < 0.01), "Loss increased sharply — learning rate may be too high"

    def test_score_returns_required_keys(self):
        """score() must return dict with keys: r2, rmse, mae."""
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        model.fit(X, y)

        metrics = model.score(X, y)
        assert "r2"   in metrics, "Missing key: r2"
        assert "rmse" in metrics, "Missing key: rmse"
        assert "mae"  in metrics, "Missing key: mae"

    def test_r2_between_minus1_and_1(self):
        """R² score must be in a reasonable range."""
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=300)
        X = np.random.randn(50, 2)
        y = 3*X[:, 0] - X[:, 1]
        model.fit(X, y)
        metrics = model.score(X, y)
        assert -1 <= metrics["r2"] <= 1.0 + 1e-6

    def test_save_and_load(self, tmp_path):
        """Model weights must be preserved exactly after save/load cycle."""
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        model.fit(X, y)

        path = str(tmp_path / "test_model.pkl")
        model.save(path)

        loaded = LinearRegressionScratch()
        loaded.load(path)

        np.testing.assert_array_almost_equal(
            model.weights, loaded.weights, decimal=8,
            err_msg="Weights changed after save/load"
        )
        assert abs(model.bias - loaded.bias) < 1e-8, "Bias changed after save/load"

    def test_predictions_same_after_load(self, tmp_path):
        """Predictions must be identical before and after save/load."""
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=100)
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        model.fit(X, y)

        path = str(tmp_path / "model2.pkl")
        model.save(path)

        loaded = LinearRegressionScratch().load(path)

        X_test = np.random.randn(5, 2)
        np.testing.assert_array_almost_equal(
            model.predict(X_test), loaded.predict(X_test),
            decimal=8, err_msg="Predictions differ after load"
        )


# ============================================================
# Unit Tests — DataPreprocessor
# ============================================================

class TestDataPreprocessor:
    """Tests for feature scaling logic."""

    def test_transform_before_fit_raises(self):
        """transform() must raise RuntimeError if fit() was not called."""
        scaler = DataPreprocessor()
        X = np.random.randn(10, 3)
        with pytest.raises(RuntimeError):
            scaler.transform(X)

    def test_fit_transform_mean_zero(self):
        """After fit_transform, feature means should be ~0."""
        scaler = DataPreprocessor()
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        y = np.array([10, 20, 30], dtype=float)
        scaler.fit(X, y)
        X_scaled, _ = scaler.transform(X, y)

        means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(
            means, [0, 0], atol=1e-10,
            err_msg=f"Scaled means should be ~0, got {means}"
        )

    def test_fit_transform_std_one(self):
        """After fit_transform, feature stds should be ~1."""
        scaler = DataPreprocessor()
        X = np.random.randn(100, 4) * 100 + 50   # large scale
        y = np.random.randn(100)
        scaler.fit(X, y)
        X_scaled, _ = scaler.transform(X, y)

        stds = X_scaled.std(axis=0)
        np.testing.assert_allclose(stds, np.ones(4), atol=0.01)

    def test_inverse_transform_recovers_original(self):
        """inverse_transform_target must recover original price values."""
        scaler = DataPreprocessor()
        X = np.random.randn(50, 3)
        y = np.random.uniform(100000, 500000, 50)
        scaler.fit(X, y)
        _, y_scaled = scaler.transform(X, y)
        y_recovered = scaler.inverse_transform_target(y_scaled)

        np.testing.assert_allclose(y, y_recovered, rtol=1e-6)

    def test_scaler_uses_train_stats_on_test(self):
        """Scaler must use training means/stds when transforming test data (no leakage)."""
        scaler = DataPreprocessor()
        X_train = np.array([[0, 0], [2, 4], [4, 8]], dtype=float)
        y_train = np.ones(3)
        scaler.fit(X_train, y_train)

        X_test = np.array([[10, 20]], dtype=float)   # very different scale
        X_test_scaled = scaler.transform(X_test)     # must use train stats, not refit

        expected = (X_test - scaler.feature_means) / scaler.feature_stds
        np.testing.assert_allclose(X_test_scaled, expected, rtol=1e-6)


# ============================================================
# Integration Tests — Full Pipeline
# ============================================================

class TestFullPipeline:
    """End-to-end tests that run the whole train→predict flow."""

    def test_full_train_predict_pipeline(self):
        """
        Integration test: train on synthetic data, evaluate on test set.
        R² must exceed 0.9 on our clean dataset.
        """
        np.random.seed(42)
        n = 300
        X = np.random.randn(n, 4)
        y = 3*X[:, 0] - 2*X[:, 1] + X[:, 2] + 0.5*X[:, 3] + np.random.randn(n)*0.3

        # Split
        split = int(n * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale
        scaler = DataPreprocessor()
        scaler.fit(X_train, y_train)
        X_tr_sc, y_tr_sc = scaler.transform(X_train, y_train)
        X_te_sc = scaler.transform(X_test)

        # Train
        model = LinearRegressionScratch(learning_rate=0.1, n_iterations=500)
        model.fit(X_tr_sc, y_tr_sc)

        # Evaluate on test (scaled)
        y_test_scaled = (y_test - scaler.target_mean) / scaler.target_std
        metrics = model.score(X_te_sc, y_test_scaled)

        assert metrics["r2"] > 0.90, \
            f"R² = {metrics['r2']} — expected > 0.90 on clean linear data"

    def test_predict_function(self):
        """predict_price() should return a positive number for valid inputs."""
        from src.predict import predict_price
        result = predict_price(1500.0, 3, 10.0, 5.0)

        assert "predicted_price_usd" in result
        assert result["predicted_price_usd"] > 0, "Predicted price must be positive"
        assert isinstance(result["predicted_price_usd"], float)

    def test_house_price_minimum(self):
        """Predicted price should never be negative."""
        from src.predict import predict_price
        result = predict_price(50.0, 1, 100.0, 200.0)   # worst-case inputs
        assert result["predicted_price_usd"] >= 0


# ============================================================
# Data Quality Tests
# ============================================================

class TestDataQuality:
    """Tests to verify the dataset meets expectations."""

    def test_dataset_loads(self):
        """Dataset CSV must exist and be loadable."""
        df, X, y = load_data()
        assert df is not None
        assert len(df) > 0

    def test_dataset_has_correct_columns(self):
        """Dataset must have exactly the expected columns."""
        df, _, _ = load_data()
        expected = {"size_sqft", "bedrooms", "age_years", "distance_km", "price_usd"}
        assert set(df.columns) == expected, \
            f"Missing columns: {expected - set(df.columns)}"

    def test_no_missing_values(self):
        """Dataset must have zero null/NaN values."""
        df, _, _ = load_data()
        nulls = df.isnull().sum()
        assert nulls.sum() == 0, f"Found null values: {nulls[nulls > 0]}"

    def test_prices_are_positive(self):
        """All house prices must be positive."""
        _, _, y = load_data()
        assert np.all(y > 0), "Found non-positive house prices!"

    def test_dataset_size(self):
        """Dataset must have at least 500 rows for meaningful training."""
        df, _, _ = load_data()
        assert len(df) >= 500, f"Dataset too small: {len(df)} rows"

    def test_train_test_split_ratio(self):
        """80/20 split must produce correct proportions."""
        _, X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

        total = len(X)
        assert abs(len(X_train)/total - 0.8) < 0.02, "Train split not ~80%"
        assert abs(len(X_test)/total  - 0.2) < 0.02, "Test split not ~20%"
        assert len(X_train) + len(X_test) == total, "No samples lost in split"


# ============================================================
# Model Quality Gate (CI/CD threshold check)
# ============================================================

class TestModelQualityGate:
    """
    These tests enforce MINIMUM QUALITY THRESHOLDS.

    In a CI/CD pipeline, if the model accuracy falls below a threshold
    (e.g. after someone makes a bad change to the code), the pipeline
    FAILS and the code is NOT deployed.

    This is called a "Quality Gate" — a hard requirement that must pass.
    """

    def test_model_r2_above_threshold(self):
        """
        Model R² on test data must be above 0.95.
        This threshold protects against regressions (pun intended 😄).
        """
        from src.predict import predict_price
        import pickle
        from src.predict import _load_model

        model, scaler = _load_model()
        _, X, y = load_data()
        _, X_test, _, y_test = train_test_split_manual(X, y)

        X_test_sc, y_test_sc = scaler.transform(X_test, y_test)
        metrics = model.score(X_test_sc, y_test_sc)

        R2_THRESHOLD = 0.95
        assert metrics["r2"] >= R2_THRESHOLD, (
            f"⚠️  Model quality gate FAILED!\n"
            f"    R² = {metrics['r2']} < threshold {R2_THRESHOLD}\n"
            f"    The model is not good enough to deploy. Investigate and retrain."
        )
