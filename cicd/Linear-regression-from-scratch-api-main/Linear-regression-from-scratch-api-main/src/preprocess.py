"""
preprocess.py
-------------
Handles all data loading, cleaning, and feature scaling.

WHY DO WE PREPROCESS?
=====================
Raw data is messy and has different scales. For example:
  - size_sqft can be 500–3500
  - bedrooms is 1–5
  - age_years is 0–50

If we feed these raw numbers into gradient descent, features with large
values (size_sqft) will dominate the gradients, and the model will train
very slowly or diverge.

FEATURE SCALING (Standardisation):
  z = (x - mean) / std_dev

After scaling, every feature has mean=0 and std=1. Gradient descent
converges much faster and more reliably.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple


FEATURE_COLS = ["size_sqft", "bedrooms", "age_years", "distance_km"]
TARGET_COL   = "price_usd"
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices.csv")


class DataPreprocessor:
    """
    Handles feature scaling using Standardisation (Z-score normalisation).

    fit()       → learns mean and std from TRAINING data only
    transform() → applies learned scaling to any split
    inverse_transform_target() → converts predictions back to USD
    """

    def __init__(self):
        self.feature_means: np.ndarray = None
        self.feature_stds:  np.ndarray = None
        self.target_mean:   float      = 0.0
        self.target_std:    float      = 1.0
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataPreprocessor":
        """
        Compute mean and std from training data.
        NEVER fit on test data — that would cause data leakage!
        """
        self.feature_means = X.mean(axis=0)
        self.feature_stds  = X.std(axis=0)
        self.feature_stds[self.feature_stds == 0] = 1.0   # avoid /0

        self.target_mean = float(y.mean())
        self.target_std  = float(y.std())
        self._fitted = True
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """Apply z-score scaling using TRAINING statistics."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X_scaled = (X - self.feature_means) / self.feature_stds

        if y is not None:
            y_scaled = (y - self.target_mean) / self.target_std
            return X_scaled, y_scaled
        return X_scaled

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original USD scale."""
        return y_scaled * self.target_std + self.target_mean


def load_data(path: str = DATA_PATH) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load CSV and split into features matrix X and target vector y."""
    df = pd.read_csv(path)
    X  = df[FEATURE_COLS].values.astype(float)
    y  = df[TARGET_COL].values.astype(float)
    return df, X, y


def train_test_split_manual(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple:
    """
    Simple train/test split without sklearn.

    WHY SPLIT?
    - Train set: model learns from this
    - Test set: we evaluate on unseen data to check for overfitting
    - NEVER train and test on the same data — that's cheating!
    """
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(X))   # shuffle indices randomly
    split   = int(len(X) * (1 - test_size))

    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
