"""
generate_data.py
----------------
Generates a synthetic house price dataset for our Linear Regression project.

WHY SYNTHETIC DATA?
- We control the "true" relationship, so we can verify our model learns correctly.
- No privacy concerns, no download needed.
- Real enough to demonstrate all CI/CD concepts.

FEATURES GENERATED:
  - size_sqft     : House size in square feet
  - bedrooms      : Number of bedrooms
  - age_years     : Age of house in years
  - distance_km   : Distance from city center (km)
  - price_usd     : Target variable (what we want to predict)
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES   = 1000


def generate_house_data(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate realistic (synthetic) house price data.

    True relationship (what the model should learn):
        price = 150*size + 8000*bedrooms - 500*age - 3000*distance + noise
    """
    rng = np.random.default_rng(seed)

    size_sqft   = rng.uniform(500, 3500, n)          # 500 – 3500 sqft
    bedrooms    = rng.integers(1, 6, n).astype(float) # 1 – 5 bedrooms
    age_years   = rng.uniform(0, 50, n)              # 0 – 50 years old
    distance_km = rng.uniform(1, 30, n)              # 1 – 30 km from center

    # True linear relationship + Gaussian noise
    noise = rng.normal(0, 15000, n)
    price = (
        150   * size_sqft
        + 8000  * bedrooms
        - 500   * age_years
        - 3000  * distance_km
        + 50000                          # base price
        + noise
    )
    price = np.clip(price, 30000, None)  # no negative prices

    df = pd.DataFrame({
        "size_sqft"  : np.round(size_sqft, 1),
        "bedrooms"   : bedrooms.astype(int),
        "age_years"  : np.round(age_years, 1),
        "distance_km": np.round(distance_km, 2),
        "price_usd"  : np.round(price, 2),
    })
    return df


if __name__ == "__main__":
    df = generate_house_data()
    out_path = os.path.join(os.path.dirname(__file__), "house_prices.csv")
    df.to_csv(out_path, index=False)
    print(f"✅  Dataset saved → {out_path}")
    print(f"    Shape  : {df.shape}")
    print(f"    Preview:\n{df.head()}")
