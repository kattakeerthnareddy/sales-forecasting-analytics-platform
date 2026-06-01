"""
Generate a synthetic-but-realistic daily retail sales dataset.

The series for each (store, category) combination is built from interpretable
components so the forecasting models have real structure to learn:

    sales = base_level
            + long_term_trend
            + weekly_seasonality      (weekends sell more)
            + yearly_seasonality      (Nov-Dec holiday lift)
            + promotion_effect        (random promo days)
            + holiday_effect          (a handful of fixed-date spikes)
            + noise

Run:
    python data/generate_data.py
Writes:
    data/sales_data.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

STORES = ["Indore", "Mumbai", "Bengaluru", "Delhi", "Hyderabad"]
CATEGORIES = ["Electronics", "Groceries", "Apparel", "Home"]

# Per-category base demand and how strongly each component applies.
CATEGORY_PROFILE = {
    #              base   trend/yr  weekend  holiday  promo
    "Electronics": dict(base=180, trend=55, weekend=1.35, holiday=2.4, promo=1.7),
    "Groceries":   dict(base=420, trend=20, weekend=1.20, holiday=1.4, promo=1.3),
    "Apparel":     dict(base=240, trend=35, weekend=1.45, holiday=2.0, promo=1.9),
    "Home":        dict(base=150, trend=25, weekend=1.25, holiday=1.6, promo=1.5),
}

# Store-level multipliers (some stores are simply bigger markets).
STORE_MULT = {
    "Indore": 0.85, "Mumbai": 1.40, "Bengaluru": 1.25,
    "Delhi": 1.30, "Hyderabad": 1.05,
}

# Fixed-date demand spikes (festivals / sale events), month-day -> lift window.
HOLIDAYS = {
    (1, 26): "Republic Day",
    (8, 15): "Independence Day",
    (10, 24): "Diwali",       # approximate; treated as a fixed annual spike
    (11, 29): "Black Friday",
    (12, 25): "Christmas",
}


def _yearly_seasonality(day_of_year: np.ndarray) -> np.ndarray:
    """Smooth annual cycle peaking in the Nov-Dec shopping season."""
    # Two harmonics: a gentle summer dip + a strong year-end peak.
    angle = 2 * np.pi * day_of_year / 365.25
    base = -0.10 * np.sin(angle)            # mild mid-year dip
    year_end = 0.45 * np.exp(-((day_of_year - 330) ** 2) / (2 * 35 ** 2))
    return base + year_end


def build() -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n_days = len(dates)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()        # 0=Mon ... 6=Sun
    t = np.arange(n_days) / 365.25          # years elapsed, for trend

    yearly = _yearly_seasonality(doy)

    # Promotions: ~8% of days are on promo, clustered slightly.
    promo_flag = (RNG.random(n_days) < 0.08).astype(int)

    # Holiday lift vector (applies in a +/- 1 day window around each holiday).
    holiday_lift = np.zeros(n_days)
    md = list(zip(dates.month, dates.day))
    for i, (m, d) in enumerate(md):
        for (hm, hd) in HOLIDAYS:
            if m == hm and abs(d - hd) <= 1:
                holiday_lift[i] = 1.0

    rows = []
    for store in STORES:
        smult = STORE_MULT[store]
        for cat in CATEGORIES:
            p = CATEGORY_PROFILE[cat]
            base = p["base"] * smult

            weekend = np.where(dow >= 5, p["weekend"], 1.0)
            trend = p["trend"] * smult * t
            season = base * yearly
            promo = np.where(promo_flag == 1, base * (p["promo"] - 1.0), 0.0)
            holiday = holiday_lift * base * (p["holiday"] - 1.0)

            signal = (base + trend + season + promo + holiday) * weekend
            noise = RNG.normal(0, 0.06 * base, n_days)
            sales = np.clip(signal + noise, 0, None).round().astype(int)

            df = pd.DataFrame({
                "date": dates,
                "store": store,
                "category": cat,
                "promo": promo_flag,
                "units_sold": sales,
            })
            rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["store", "category", "date"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    df = build()
    df.to_csv("data/sales_data.csv", index=False)
    print(f"Wrote data/sales_data.csv  ->  {len(df):,} rows")
    print(df.head())
    print("\nDate range:", df['date'].min().date(), "to", df['date'].max().date())
    print("Stores:", df['store'].nunique(), "| Categories:", df['category'].nunique())
    print("Total units (all series):", f"{df['units_sold'].sum():,}")
