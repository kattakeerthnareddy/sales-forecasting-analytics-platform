"""Model definitions for the forecasting pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def build_gbm(cat_feature_indices: list[int]) -> HistGradientBoostingRegressor:
    """Gradient boosting regressor configured for this dataset."""
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        categorical_features=cat_feature_indices,
        random_state=42,
    )


def seasonal_naive_predict(test_features: pd.DataFrame) -> np.ndarray:
    """Baseline: predict each day as the value 7 days earlier (the lag_7 feature)."""
    return test_features["lag_7"].to_numpy()
