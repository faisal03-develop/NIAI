"""
app.py
------
Flask REST API + serves the website directly.

ENDPOINTS:
  GET  /home         → opens the website (index.html)
  GET  /             → API welcome message (JSON)
  GET  /health       → health check
  GET  /model/info   → model weights and statistics
  POST /predict      → predict house price
  GET  /metrics      → R², RMSE, MAE metrics

RUN:
  python app.py
  Then open: http://localhost:5000/home
"""

import os
import sys
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import predict_price, get_model_info

app = Flask(__name__)
CORS(app)

app.config["JSON_SORT_KEYS"] = False

METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.txt")


# ─────────────────────────────────────────────────────────────
# WEBSITE — serves index.html directly from Flask
# ─────────────────────────────────────────────────────────────

@app.route("/home")
def website():
    """Serve the frontend website."""
    return send_from_directory(".", "index.html")


# ─────────────────────────────────────────────────────────────
# HELPER — input validation
# ─────────────────────────────────────────────────────────────

def validate_input(data):
    required = ["size_sqft", "bedrooms", "age_years", "distance_km"]
    missing  = [k for k in required if k not in data]
    if missing:
        return None, f"Missing required fields: {missing}"
    try:
        inputs = {
            "size_sqft"  : float(data["size_sqft"]),
            "bedrooms"   : int(data["bedrooms"]),
            "age_years"  : float(data["age_years"]),
            "distance_km": float(data["distance_km"]),
        }
    except (ValueError, TypeError) as e:
        return None, f"Invalid data types: {str(e)}"

    if not (50 <= inputs["size_sqft"] <= 10000):
        return None, "size_sqft must be between 50 and 10,000"
    if not (1 <= inputs["bedrooms"] <= 10):
        return None, "bedrooms must be between 1 and 10"
    if not (0 <= inputs["age_years"] <= 150):
        return None, "age_years must be between 0 and 150"
    if not (0 <= inputs["distance_km"] <= 200):
        return None, "distance_km must be between 0 and 200"

    return inputs, None


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """API welcome — lists all endpoints."""
    return jsonify({
        "name"       : "House Price Prediction API",
        "description": "Linear Regression model built from scratch with NumPy",
        "version"    : "1.0.0",
        "website"    : "Visit /home to open the website",
        "endpoints"  : {
            "GET  /home"       : "Open the website",
            "GET  /"           : "This message",
            "GET  /health"     : "Health check",
            "GET  /model/info" : "Model weights and statistics",
            "POST /predict"    : "Predict house price",
            "GET  /metrics"    : "Model evaluation metrics",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check — used by CI/CD pipeline to verify deployment."""
    try:
        _ = get_model_info()
        return jsonify({"status": "healthy", "model": "loaded"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict house price from JSON input.

    Request body:
      {
        "size_sqft"  : 1500,
        "bedrooms"   : 3,
        "age_years"  : 10,
        "distance_km": 5.0
      }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Request must be JSON"}), 400

    data = request.get_json()
    inputs, error = validate_input(data)

    if error:
        return jsonify({"success": False, "error": error}), 422

    try:
        result = predict_price(**inputs)
        return jsonify({
            "success"            : True,
            "predicted_price_usd": result["predicted_price_usd"],
            "inputs"             : result["inputs"],
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model weights, bias and feature statistics."""
    try:
        info = get_model_info()
        return jsonify({"success": True, **info}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return the latest training evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({
            "success": False,
            "error"  : "metrics.txt not found — run python src/train.py first"
        }), 404

    result = {}
    with open(METRICS_PATH) as f:
        for line in f:
            key, val = line.strip().split("=")
            result[key] = float(val)

    return jsonify({"success": True, "metrics": result}), 200


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n🚀  House Price Prediction API")
    print(f"    ─────────────────────────────────────")
    print(f"    API root  → http://localhost:{port}")
    print(f"    Website   → http://localhost:{port}/home")
    print(f"    Health    → http://localhost:{port}/health")
    print(f"    Metrics   → http://localhost:{port}/metrics")
    print(f"    ─────────────────────────────────────\n")

    app.run(host="0.0.0.0", port=port, debug=debug)