"""Recursive multi-step forecasting.

The trained model predicts one day ahead using lag and rolling features. To
project an arbitrary horizon, we predict day t+1, append that prediction to the
history, recompute features, and repeat. This "feed the prediction back as a
lag" loop is the standard way to turn a one-step model into a multi-step one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import GROUP_KEYS, TARGET, add_calendar_features, LAGS, ROLL_WINDOWS


def _features_last_row(history: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Build features for the most recent (future) row without dropping it."""
    df = add_calendar_features(history)
    s = df[TARGET]
    for lag in LAGS:
        df[f"lag_{lag}"] = s.shift(lag)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = s.shift(1).rolling(w).std()
    for col in GROUP_KEYS:
        df[col] = df[col].astype("category")
    return df.iloc[[-1]][feature_cols]


def forecast_series(model, raw_df: pd.DataFrame, store: str, category: str,
                    horizon: int, feature_cols: list[str]) -> pd.DataFrame:
    """Forecast `horizon` future days for one (store, category) series."""
    hist = (raw_df[(raw_df["store"] == store) & (raw_df["category"] == category)]
            .sort_values("date")
            .reset_index(drop=True)
            .copy())

    last_date = hist["date"].max()
    out_dates, out_preds = [], []

    for i in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=i)
        hist = pd.concat([hist, pd.DataFrame([{
            "date": d, "store": store, "category": category,
            "promo": 0, TARGET: np.nan,
        }])], ignore_index=True)

        X = _features_last_row(hist, feature_cols)
        yhat = float(model.predict(X)[0])
        yhat = max(yhat, 0.0)

        hist.loc[hist.index[-1], TARGET] = yhat  # feed prediction back
        out_dates.append(d)
        out_preds.append(round(yhat))

    return pd.DataFrame({"date": out_dates, "store": store,
                         "category": category, "forecast": out_preds})
