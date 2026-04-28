"""
tests/test_api.py
-----------------
Tests for the Flask REST API endpoints.

These tests use Flask's built-in test client — no real server is started.
The test client simulates HTTP requests so we can test every endpoint
without port conflicts or network issues.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app as flask_app


@pytest.fixture
def client():
    """
    pytest fixture — sets up a test client before each test.
    Fixtures are like 'setup' helpers shared across tests.
    """
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# ============================================================
# Index & Health
# ============================================================

class TestIndexAndHealth:

    def test_index_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_index_has_endpoints_key(self, client):
        data = response = client.get("/")
        body = json.loads(response.data)
        assert "endpoints" in body

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body_has_status(self, client):
        body = json.loads(client.get("/health").data)
        assert "status" in body
        assert body["status"] == "healthy"


# ============================================================
# Prediction Endpoint
# ============================================================

class TestPredictEndpoint:

    def _valid_payload(self):
        return {"size_sqft": 1500, "bedrooms": 3, "age_years": 10, "distance_km": 5.0}

    def test_valid_request_returns_200(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(self._valid_payload()),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_valid_response_has_price(self, client):
        body = json.loads(client.post(
            "/predict",
            data=json.dumps(self._valid_payload()),
            content_type="application/json",
        ).data)
        assert body["success"] is True
        assert "predicted_price_usd" in body
        assert body["predicted_price_usd"] > 0

    def test_missing_field_returns_422(self, client):
        payload = {"size_sqft": 1500, "bedrooms": 3}   # missing age & distance
        resp = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_non_json_returns_400(self, client):
        resp = client.post("/predict", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_invalid_size_returns_422(self, client):
        payload = self._valid_payload()
        payload["size_sqft"] = -500   # invalid
        resp = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 422

    def test_large_house_gives_higher_price_than_small(self, client):
        """Business logic: larger house must cost more (all else equal)."""
        small = dict(size_sqft=600,  bedrooms=2, age_years=5, distance_km=5)
        large = dict(size_sqft=3000, bedrooms=2, age_years=5, distance_km=5)

        def get_price(payload):
            body = json.loads(client.post(
                "/predict", data=json.dumps(payload), content_type="application/json"
            ).data)
            return body["predicted_price_usd"]

        assert get_price(large) > get_price(small), \
            "Larger house should cost more than smaller house"

    def test_older_house_gives_lower_price(self, client):
        """Business logic: newer house must cost more (all else equal)."""
        new_house = dict(size_sqft=1500, bedrooms=3, age_years=2,  distance_km=5)
        old_house = dict(size_sqft=1500, bedrooms=3, age_years=40, distance_km=5)

        def get_price(payload):
            body = json.loads(client.post(
                "/predict", data=json.dumps(payload), content_type="application/json"
            ).data)
            return body["predicted_price_usd"]

        assert get_price(new_house) > get_price(old_house), \
            "Newer house should cost more than older house"


# ============================================================
# Model Info & Metrics Endpoints
# ============================================================

class TestInfoAndMetrics:

    def test_model_info_returns_200(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200

    def test_model_info_has_weights(self, client):
        body = json.loads(client.get("/model/info").data)
        assert "weights" in body
        assert "bias" in body

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_has_r2(self, client):
        body = json.loads(client.get("/metrics").data)
        assert body["success"] is True
        assert "r2" in body["metrics"]
