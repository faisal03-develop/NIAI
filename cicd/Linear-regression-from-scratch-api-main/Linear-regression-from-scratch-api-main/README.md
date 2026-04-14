# 🏠 House Price Prediction — Linear Regression from Scratch + CI/CD

> **Built for:** Technical Session — Linear Regression & CI/CD  
> **Level:** Beginner-friendly, production-grade  
> **Model:** Linear Regression built from scratch using only NumPy (no sklearn for the model!)

---

## 📁 Project Structure

```
linear-regression-cicd/
│
├── 📂 src/
│   ├── linear_regression.py   ← Model built from scratch (NumPy only)
│   ├── preprocess.py          ← Feature scaling & data loading
│   ├── train.py               ← Training pipeline
│   ├── predict.py             ← Prediction function for API
│   └── __init__.py
│
├── 📂 data/
│   ├── generate_data.py       ← Generates synthetic house price dataset
│   └── house_prices.csv       ← Generated dataset (1000 rows)
│
├── 📂 tests/
│   ├── test_model.py          ← Unit + integration + quality gate tests
│   ├── test_api.py            ← Flask API endpoint tests
│   └── conftest.py            ← Shared pytest fixtures
│
├── 📂 models/                 ← Saved model files (auto-generated)
│   ├── linear_regression.pkl  ← Trained weights
│   ├── scaler.pkl             ← Feature scaler
│   └── metrics.txt            ← Evaluation metrics
│
├── 📂 notebooks/
│   └── linear_regression_explained.ipynb  ← Full explanation + visualisations
│
├── 📂 docs/                   ← Generated plots from notebook
│
├── 📂 .github/
│   └── workflows/
│       └── ci_cd.yml          ← 🚀 THE CI/CD PIPELINE
│
├── app.py                     ← Flask REST API
├── Dockerfile                 ← Container definition
├── render.yaml                ← Render.com deployment config
├── requirements.txt           ← Python dependencies
├── pytest.ini                 ← Test configuration
├── Makefile                   ← Shortcut commands
└── .gitignore
```

---

## 🧠 How Linear Regression Works (Quick Summary)

```
Goal: predict house price from features

Formula:  price = w1×size + w2×bedrooms + w3×age + w4×distance + bias

Training: start with random weights → compute predictions → measure error
          → compute gradient → update weights → repeat 1000× times

This process (Gradient Descent) is how the model "learns".
```

**Key metrics our model achieves:**
| Metric | Value | Meaning |
|--------|-------|---------|
| R² Score | **0.9865** | Model explains 98.65% of price variance |
| RMSE | ~$15,000 | Average prediction error in USD |
| MAE | ~$12,500 | Average absolute error |

---

## 🚀 Quick Start — Run Locally

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-cicd.git
cd linear-regression-cicd
```

### Step 2: Create a virtual environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Generate data & train the model
```bash
# Generate synthetic house price dataset
python data/generate_data.py

# Train the model (saves to models/)
python src/train.py
```

You should see:
```
=======================================================
  House Price Prediction — Linear Regression Training
=======================================================
  Iteration    1/1000  MSE Loss: ...
  ...
  ✅  R²  = 0.9865
  ✅  RMSE = $15,435
  Training complete! 🎉
```

### Step 5: Run all tests
```bash
pytest tests/ -v
```

Expected: **all tests pass ✅**

### Step 6: Start the API server
```bash
python app.py
```

API is live at **http://localhost:5000**

### Step 7: Test the API
```bash
# Health check
curl http://localhost:5000/health

# Predict a house price
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"size_sqft": 1500, "bedrooms": 3, "age_years": 10, "distance_km": 5}'
```

Response:
```json
{
  "success": true,
  "predicted_price_usd": 247832.50,
  "inputs": {
    "size_sqft": 1500,
    "bedrooms": 3,
    "age_years": 10,
    "distance_km": 5.0
  }
}
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome + available endpoints |
| GET | `/health` | Health check (used by CI/CD) |
| GET | `/model/info` | Model weights and feature statistics |
| POST | `/predict` | Predict house price from JSON input |
| GET | `/metrics` | R², RMSE, MAE from last training run |

### POST /predict — Request Body
```json
{
  "size_sqft"  : 1500,
  "bedrooms"   : 3,
  "age_years"  : 10,
  "distance_km": 5.0
}
```

---

## 🔄 CI/CD Pipeline — How It Works

Every time you `git push` to GitHub, this pipeline runs automatically:

```
Developer pushes code
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  GitHub detects push → reads .github/workflows/     │
│  ci_cd.yml → starts pipeline on GitHub's servers    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  JOB 1: TEST (& JOB 2: LINT run in parallel)        │
│  1. Install Python + dependencies                   │
│  2. Generate dataset                                │
│  3. Train model                                     │
│  4. Run pytest (30+ tests)                          │
│  ✅ Pass → continue  ❌ Fail → STOP, notify dev     │
└─────────────────────────────────────────────────────┘
        │  (only if tests pass)
        ▼
┌─────────────────────────────────────────────────────┐
│  JOB 3: BUILD                                       │
│  1. Build Docker image                              │
│  2. Start container                                 │
│  3. Smoke test: GET /health → must return 200       │
│  ✅ Pass → continue  ❌ Fail → STOP                 │
└─────────────────────────────────────────────────────┘
        │  (only on main branch)
        ▼
┌─────────────────────────────────────────────────────┐
│  JOB 4: DEPLOY                                      │
│  1. Trigger Render.com to pull & deploy new code    │
│  2. Wait 30s, verify /health returns 200 on live URL│
│  ✅ App is LIVE for users                           │
└─────────────────────────────────────────────────────┘
```

---

## 📤 Push to GitHub — Complete Step-by-Step Guide

### Step 1: Create GitHub account
Go to https://github.com and sign up if you don't have an account.

### Step 2: Create a new repository
1. Click the **+** button (top right) → **New repository**
2. Repository name: `linear-regression-cicd`
3. Visibility: **Public**
4. Do NOT initialise with README (we have our own)
5. Click **Create repository**

### Step 3: Initialise Git in your project
```bash
# Go to project folder
cd linear-regression-cicd

# Initialise git
git init

# Add all files
git add .

# First commit
git commit -m "feat: initial project — linear regression from scratch with CI/CD"

# Set main as default branch
git branch -M main
```

### Step 4: Connect to GitHub and push
```bash
# Connect local repo to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/linear-regression-cicd.git

# Push to GitHub
git push -u origin main
```

### Step 5: Watch the pipeline run!
1. Go to your repo on GitHub
2. Click the **Actions** tab
3. You should see your pipeline running (yellow = running, green = passed, red = failed)
4. Click on it to see each step in real time

---

## 🌐 Deploy to Render (Free Hosting)

### Step 1: Sign up at render.com
Go to https://render.com → Sign up with GitHub

### Step 2: Create a new Web Service
1. Click **New** → **Web Service**
2. Connect your GitHub repository
3. Render automatically detects `render.yaml`
4. Click **Create Web Service**

### Step 3: Get your Deploy Hook URL
1. In Render dashboard → your service → **Settings**
2. Scroll to **Deploy Hook**
3. Copy the URL

### Step 4: Add secret to GitHub
1. Go to your GitHub repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `RENDER_DEPLOY_HOOK`
4. Value: paste the URL from Render
5. Click **Add secret**

Now every push to `main` will:
1. Run tests ✅
2. Build Docker image ✅
3. Auto-deploy to Render ✅

---

## 🔁 Day-to-Day Workflow (After Setup)

```bash
# 1. Make a change (e.g. improve the model)
nano src/linear_regression.py

# 2. Test locally first
pytest tests/ -v

# 3. Commit and push
git add .
git commit -m "improve: increase learning rate for faster convergence"
git push

# 4. GitHub Actions automatically:
#    - Runs all 30+ tests
#    - Builds Docker image
#    - Deploys to production
#    - You get email if anything fails
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast)
pytest tests/ -v -m "not integration"

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestLinearRegressionScratch::test_perfect_fit_on_noiseless_data -v
```

---

## 🐳 Docker Commands

```bash
# Build image
docker build -t house-price-lr .

# Run container
docker run -p 5000:5000 house-price-lr

# Run in background
docker run -d -p 5000:5000 --name my-api house-price-lr

# See logs
docker logs my-api

# Stop and remove
docker stop my-api && docker rm my-api
```

---

## 📊 Notebook — Visual Explanation

Open the Jupyter notebook for a complete visual walkthrough:

```bash
pip install jupyter matplotlib
jupyter notebook notebooks/linear_regression_explained.ipynb
```

Topics covered:
- Linear regression equation visualised
- Loss surface (the "bowl" we roll down)
- Gradient descent convergence
- Feature correlations
- Predicted vs actual scatter plot
- Feature importance
- Our model vs scikit-learn comparison

---

## 📬 API Test Examples

```bash
# Test 1: Small affordable house
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"size_sqft": 600, "bedrooms": 1, "age_years": 35, "distance_km": 25}'

# Test 2: Large luxury house
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"size_sqft": 4000, "bedrooms": 5, "age_years": 2, "distance_km": 2}'

# Test 3: Invalid input (should return error)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"size_sqft": -100, "bedrooms": 3, "age_years": 10, "distance_km": 5}'
```

---

## 🙋 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Model not found` | Run `python data/generate_data.py` then `python src/train.py` |
| Port 5000 already in use | Run `python app.py` with `PORT=5001` env variable |
| Tests failing | Make sure you trained the model first |
| CI/CD not triggering | Check you pushed to `main` branch, not `master` |

---

## 📚 What You Learned

| Topic | Where it appears |
|-------|-----------------|
| Linear Regression math | `src/linear_regression.py` |
| Gradient Descent | `LinearRegressionScratch.fit()` |
| Feature Scaling | `src/preprocess.py` |
| REST API | `app.py` |
| Unit Testing | `tests/test_model.py` |
| CI/CD Pipeline | `.github/workflows/ci_cd.yml` |
| Containerisation | `Dockerfile` |
| Cloud Deployment | `render.yaml` |

---

*Built with ❤️ for the Technical Session — Laiba*
