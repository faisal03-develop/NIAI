"""
linear_regression.py
--------------------
Linear Regression built FROM SCRATCH using only NumPy.

HOW LINEAR REGRESSION WORKS (Step by step):
============================================

GOAL:
  Given input features X (e.g. size, bedrooms), predict a continuous
  output y (e.g. house price).

THE MATH:
  ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
   = X · W + b          (matrix form)

  Where:
    X = input features matrix  (n_samples × n_features)
    W = weights vector         (n_features × 1)  ← what we LEARN
    b = bias (intercept)       (scalar)           ← what we LEARN
    ŷ = predicted output

LOSS FUNCTION — Mean Squared Error (MSE):
  MSE = (1/n) Σ (yᵢ - ŷᵢ)²

  We want to MINIMISE this. The smaller the MSE, the better our predictions.

GRADIENT DESCENT (how we learn W and b):
  We start with random weights, then iteratively move them in the direction
  that reduces the loss.

  Gradient (derivative) of MSE with respect to W:
    dL/dW = (-2/n) · Xᵀ · (y - ŷ)

  Gradient of MSE with respect to b:
    dL/db = (-2/n) · Σ(y - ŷ)

  Update rule:
    W = W - learning_rate · dL/dW
    b = b - learning_rate · dL/db

  learning_rate controls how BIG a step we take each iteration.
  Too large → overshoot. Too small → too slow.
"""

import numpy as np
import pickle
import os
from typing import Optional


class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using NumPy.
    Trained via Gradient Descent (not the closed-form normal equation).

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent (default 0.01)
    n_iterations : int
        Number of gradient descent steps (default 1000)
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations

        # These are learned during training
        self.weights: Optional[np.ndarray] = None
        self.bias:    float                = 0.0
        self.loss_history: list            = []

    # ------------------------------------------------------------------
    # STEP 1 — Prediction (Forward pass)
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions: ŷ = X · W + b

        This is the ENTIRE prediction step for linear regression.
        It is a simple weighted sum of features plus a bias.
        """
        if self.weights is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")
        return X @ self.weights + self.bias   # matrix multiplication + bias

    # ------------------------------------------------------------------
    # STEP 2 — Loss Computation
    # ------------------------------------------------------------------
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss.

        MSE = (1/n) * sum( (y_true - y_pred)^2 )

        Why MSE?
        - Squaring makes all errors positive
        - Penalises large errors more (a $20k error is penalised 4× more than $10k)
        - It is differentiable (we can take gradients)
        """
        n = len(y_true)
        return float(np.mean((y_true - y_pred) ** 2))

    # ------------------------------------------------------------------
    # STEP 3 — Gradient Computation
    # ------------------------------------------------------------------
    def _compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        Compute gradients of MSE with respect to weights and bias.

        dL/dW = (-2/n) * Xᵀ · (y_true - y_pred)
        dL/db = (-2/n) * sum(y_true - y_pred)

        The NEGATIVE sign means: if residual is positive (we under-predicted),
        increase the weights. If negative (over-predicted), decrease them.
        """
        n = X.shape[0]
        residuals = y_true - y_pred                # how wrong we were

        dW = (-2 / n) * (X.T @ residuals)         # gradient for weights
        db = (-2 / n) * np.sum(residuals)          # gradient for bias

        return dW, db

    # ------------------------------------------------------------------
    # STEP 4 — Training loop (Gradient Descent)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        """
        Train the model using Gradient Descent.

        Algorithm:
          1. Initialise weights to zeros
          2. For each iteration:
             a. Compute predictions (forward pass)
             b. Compute loss
             c. Compute gradients
             d. Update weights: W = W - lr * dW
             e. Update bias:    b = b - lr * db
          3. Repeat until n_iterations reached
        """
        n_samples, n_features = X.shape

        # Initialise weights to zero (or small random values)
        self.weights = np.zeros(n_features)
        self.bias    = 0.0
        self.loss_history = []

        for iteration in range(self.n_iterations):
            # a. Forward pass → predictions
            y_pred = self.predict(X)

            # b. Loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # c. Gradients
            dW, db = self._compute_gradients(X, y, y_pred)

            # d & e. Update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias    -= self.learning_rate * db

            # Log progress every 100 iterations
            if (iteration + 1) % 100 == 0 or iteration == 0:
                print(f"  Iteration {iteration+1:>4}/{self.n_iterations}  "
                      f"MSE Loss: {loss:>15,.2f}")

        return self

    # ------------------------------------------------------------------
    # Evaluation Metrics
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model with multiple metrics.

        R² (R-squared):
          - 1.0 = perfect predictions
          - 0.0 = model is no better than predicting the mean
          - Can be negative if model is very bad

        RMSE (Root Mean Squared Error):
          - Same units as target variable (USD for house prices)
          - "On average, predictions are off by $X"

        MAE (Mean Absolute Error):
          - Average absolute error, less sensitive to outliers
        """
        y_pred = self.predict(X)

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - y_pred) ** 2)          # sum of squared residuals
        ss_tot = np.sum((y - np.mean(y)) ** 2)      # total variance
        r2     = 1 - (ss_res / ss_tot)

        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        mae  = float(np.mean(np.abs(y - y_pred)))

        return {"r2": round(r2, 4), "rmse": round(rmse, 2), "mae": round(mae, 2)}

    # ------------------------------------------------------------------
    # Save / Load model
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise model to disk using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"weights": self.weights, "bias": self.bias}, f)
        print(f"✅  Model saved → {path}")

    def load(self, path: str) -> "LinearRegressionScratch":
        """Load model weights from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.weights = data["weights"]
        self.bias    = data["bias"]
        return self
