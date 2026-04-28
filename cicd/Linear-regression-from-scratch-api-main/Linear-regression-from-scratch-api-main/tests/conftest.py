"""
tests/conftest.py
-----------------
pytest configuration file.

conftest.py is automatically loaded by pytest before running any tests.
It is used to:
  - Define SHARED fixtures (reusable setup/teardown)
  - Configure pytest settings
  - Add command-line options
  - Set up global test state

Any fixture defined here is AUTOMATICALLY available in ALL test files
without needing to import it.
"""

import sys
import os
import numpy as np
import pytest

# Ensure src/ is importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_X():
    """
    Small synthetic feature matrix used across multiple tests.
    'scope=session' means it's created ONCE per test run (not per test).
    This saves time when many tests need the same data.
    """
    np.random.seed(42)
    return np.random.randn(100, 4)


@pytest.fixture(scope="session")
def sample_y(sample_X):
    """
    Target vector that is a linear combination of sample_X.
    Using a TRUE linear relationship so the model should fit it well.
    """
    return (
        3.0  * sample_X[:, 0]
        - 2.0 * sample_X[:, 1]
        + 1.5 * sample_X[:, 2]
        - 0.5 * sample_X[:, 3]
        + np.random.randn(100) * 0.1    # tiny noise
    )


@pytest.fixture(scope="session")
def trained_model(sample_X, sample_y):
    """
    A pre-trained LinearRegressionScratch model.
    scope=session means trained ONCE and reused by all tests that need it.
    """
    from src.linear_regression import LinearRegressionScratch
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=500)
    model.fit(sample_X, sample_y)
    return model


@pytest.fixture(scope="session")
def fitted_scaler(sample_X, sample_y):
    """A pre-fitted DataPreprocessor scaler."""
    from src.preprocess import DataPreprocessor
    scaler = DataPreprocessor()
    scaler.fit(sample_X, sample_y)
    return scaler


@pytest.fixture
def flask_client():
    """
    Flask test client fixture (function-scoped = fresh per test).
    Used in API tests to simulate HTTP requests.
    """
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_house_payload():
    """A valid prediction request payload for API tests."""
    return {
        "size_sqft"  : 1500.0,
        "bedrooms"   : 3,
        "age_years"  : 10.0,
        "distance_km": 5.0,
    }


# ────────────────────────────────────────────────────────────────────
# pytest configuration
# ────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Add custom pytest markers to avoid 'unknown mark' warnings."""
    config.addinivalue_line("markers", "slow: marks tests as slow (run with -m slow)")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "quality_gate: CI/CD quality threshold tests")
