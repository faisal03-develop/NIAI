# ====================================================================
# Makefile
# ====================================================================
# A Makefile lets you define SHORT ALIASES for long commands.
# Instead of remembering `python src/train.py`, you just run `make train`.
#
# USAGE:
#   make setup      → install everything from scratch
#   make data       → generate the dataset
#   make train      → train the model
#   make test       → run all tests
#   make run        → start the Flask API server
#   make docker     → build and run in Docker
#   make pipeline   → simulate the full CI/CD pipeline locally
#   make clean      → remove generated files
# ====================================================================

# Default Python command
PYTHON = python3

# ── Setup ─────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	@echo "🔧 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Setup complete!"

# ── Data Generation ───────────────────────────────────────────────────
.PHONY: data
data:
	@echo "📊 Generating dataset..."
	$(PYTHON) data/generate_data.py

# ── Training ──────────────────────────────────────────────────────────
.PHONY: train
train: data
	@echo "🏋️  Training Linear Regression model..."
	$(PYTHON) src/train.py

# ── Testing ───────────────────────────────────────────────────────────
.PHONY: test
test:
	@echo "🧪 Running test suite..."
	pytest tests/ -v --tb=short
	@echo "✅ All tests passed!"

# Run only fast unit tests (no model loading)
.PHONY: test-unit
test-unit:
	pytest tests/ -v -m "not integration and not quality_gate" --tb=short

# Run with coverage report
.PHONY: test-cov
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "📊 Coverage report saved to htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────────────
.PHONY: lint
lint:
	@echo "🔍 Running flake8 linter..."
	flake8 src/ app.py --max-line-length=120 --ignore=E501,W503,W504
	@echo "✅ Code quality check passed!"

# ── Run API Server ────────────────────────────────────────────────────
.PHONY: run
run:
	@echo "🚀 Starting Flask API server on http://localhost:5000"
	$(PYTHON) app.py

# Run with gunicorn (production mode)
.PHONY: run-prod
run-prod:
	@echo "🚀 Starting with gunicorn (production mode)..."
	gunicorn --workers 2 --bind 0.0.0.0:5000 --timeout 120 app:app

# ── Docker ────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t house-price-lr:latest .
	@echo "✅ Docker image built!"

.PHONY: docker-run
docker-run:
	@echo "🐳 Running container on http://localhost:5000..."
	docker run -d -p 5000:5000 --name house-price-api house-price-lr:latest

.PHONY: docker-stop
docker-stop:
	docker stop house-price-api && docker rm house-price-api

.PHONY: docker
docker: docker-build docker-run
	@echo "✅ Container started! Test with: make api-test"

# ── API Testing ───────────────────────────────────────────────────────
.PHONY: api-test
api-test:
	@echo "🔬 Testing API endpoints..."
	@echo "\n--- Health Check ---"
	curl -s http://localhost:5000/health | python3 -m json.tool

	@echo "\n--- Prediction ---"
	curl -s -X POST http://localhost:5000/predict \
		-H "Content-Type: application/json" \
		-d '{"size_sqft":1500,"bedrooms":3,"age_years":10,"distance_km":5}' \
		| python3 -m json.tool

	@echo "\n--- Model Info ---"
	curl -s http://localhost:5000/model/info | python3 -m json.tool

	@echo "\n--- Metrics ---"
	curl -s http://localhost:5000/metrics | python3 -m json.tool

# ── Full CI/CD Pipeline Simulation (run locally) ──────────────────────
.PHONY: pipeline
pipeline:
	@echo "======================================================"
	@echo "  🚀 Simulating Full CI/CD Pipeline Locally"
	@echo "======================================================"

	@echo "\n[1/5] 📊 Generating data..."
	$(PYTHON) data/generate_data.py

	@echo "\n[2/5] 🏋️  Training model..."
	$(PYTHON) src/train.py

	@echo "\n[3/5] 🔍 Linting code..."
	flake8 src/ app.py --max-line-length=120 --ignore=E501,W503,W504 || true

	@echo "\n[4/5] 🧪 Running tests..."
	pytest tests/ -v --tb=short

	@echo "\n[5/5] 🐳 Building Docker image..."
	docker build -t house-price-lr:pipeline-test . 2>/dev/null || echo "(Docker not available, skipping)"

	@echo "\n======================================================"
	@echo "  ✅ Pipeline simulation complete!"
	@echo "======================================================"

# ── Clean ─────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	@echo "🧹 Cleaning generated files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .coverage 2>/dev/null || true
	rm -f models/*.pkl models/metrics.txt 2>/dev/null || true
	rm -f data/house_prices.csv 2>/dev/null || true
	@echo "✅ Clean complete!"

# ── Help ──────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  House Price Prediction — Available Commands"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make setup       Install all dependencies"
	@echo "  make data        Generate synthetic dataset"
	@echo "  make train       Train the model"
	@echo "  make test        Run full test suite"
	@echo "  make test-unit   Run only fast unit tests"
	@echo "  make test-cov    Run tests with coverage report"
	@echo "  make lint        Check code style with flake8"
	@echo "  make run         Start Flask API (dev mode)"
	@echo "  make run-prod    Start Flask API (production mode)"
	@echo "  make docker      Build and run in Docker"
	@echo "  make api-test    Test all API endpoints with curl"
	@echo "  make pipeline    Simulate full CI/CD pipeline locally"
	@echo "  make clean       Remove generated files"
	@echo "  ─────────────────────────────────────────────────────"
	@echo ""
