"""End-to-end sales forecasting pipeline.

Steps:
    1. Load data + build features (src/features.py)
    2. Time-based train/test split (last TEST_DAYS held out)
    3. Seasonal-naive baseline (predict the value 7 days earlier)
    4. Gradient boosting model (HistGradientBoostingRegressor)
    5. Evaluate both, save figures to reports/figures/, persist the model

Run from the project root:
    python src/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from features import GROUP_KEYS, TARGET, build_features, feature_columns
from models import build_gbm, seasonal_naive_predict
from evaluate import score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sales_data.csv"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
TEST_DAYS = 90

sns.set_theme(style="whitegrid")
PALETTE = {"actual": "#1b3a4b", "model": "#e07a5f", "baseline": "#9aa5ab"}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    return df


def time_split(df: pd.DataFrame):
    cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    return train, test, cutoff


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    raw = load()
    feat = build_features(raw)
    cols = feature_columns(feat)

    train, test, cutoff = time_split(feat)
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,} | cutoff: {cutoff.date()}")

    X_train, y_train = train[cols], train[TARGET].to_numpy()
    X_test, y_test = test[cols], test[TARGET].to_numpy()

    cat_features = [cols.index(c) for c in GROUP_KEYS]

    model = build_gbm(cat_features)
    model.fit(X_train, y_train)

    pred_model = model.predict(X_test)
    pred_base = seasonal_naive_predict(test)

    m_model = score(y_test, pred_model)
    m_base = score(y_test, pred_base)

    improvement = round((m_base["MAE"] - m_model["MAE"]) / m_base["MAE"] * 100, 1)

    print("\n=== Test-set performance (last 90 days) ===")
    print(f"{'Metric':<8}{'Baseline':>12}{'Model':>12}")
    for k in ("MAE", "RMSE", "MAPE"):
        print(f"{k:<8}{m_base[k]:>12}{m_model[k]:>12}")
    print(f"\nMAE improvement vs baseline: {improvement}%")

    # ---- persist artifacts -------------------------------------------------
    joblib.dump(model, MODELS / "gbm_model.joblib")
    (MODELS / "feature_cols.json").write_text(json.dumps(cols, indent=2))
    metrics = {
        "test_days": TEST_DAYS,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "baseline_seasonal_naive": m_base,
        "gradient_boosting": m_model,
        "mae_improvement_pct": improvement,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figures -----------------------------------------------------------
    _fig_eda(raw)
    _fig_forecast(test, y_test, pred_model, pred_base)
    _fig_importance(model, X_test, y_test, cols)
    _fig_error_by_category(test, y_test, pred_model)
    print(f"\nSaved 4 figures to {FIG}")
    print("Saved model to", MODELS / "gbm_model.joblib")


def _fig_eda(raw: pd.DataFrame) -> None:
    daily = raw.groupby("date", as_index=False)[TARGET].sum()
    plt.figure(figsize=(11, 4.5))
    plt.plot(daily["date"], daily[TARGET], color=PALETTE["actual"], lw=0.9)
    plt.title("Total daily units sold (all stores & categories)", fontsize=13, weight="bold")
    plt.xlabel(""); plt.ylabel("Units sold")
    plt.tight_layout()
    plt.savefig(FIG / "01_sales_over_time.png", dpi=130)
    plt.close()


def _fig_forecast(test, y_test, pred_model, pred_base) -> None:
    t = test.copy()
    t["actual"], t["model"], t["baseline"] = y_test, pred_model, pred_base
    agg = t.groupby("date", as_index=False)[["actual", "model", "baseline"]].sum()

    plt.figure(figsize=(11, 4.5))
    plt.plot(agg["date"], agg["actual"], label="Actual", color=PALETTE["actual"], lw=2)
    plt.plot(agg["date"], agg["model"], label="Gradient Boosting", color=PALETTE["model"], lw=1.6)
    plt.plot(agg["date"], agg["baseline"], label="Seasonal naive", color=PALETTE["baseline"],
             lw=1.2, ls="--")
    plt.title("Forecast vs actual — aggregate, 90-day test window", fontsize=13, weight="bold")
    plt.xlabel(""); plt.ylabel("Units sold"); plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "02_forecast_vs_actual.png", dpi=130)
    plt.close()


def _fig_importance(model, X_test, y_test, cols) -> None:
    r = permutation_importance(model, X_test, y_test, n_repeats=5,
                               random_state=42, scoring="neg_mean_absolute_error")
    imp = pd.Series(r.importances_mean, index=cols).sort_values().tail(12)
    plt.figure(figsize=(8, 5))
    plt.barh(imp.index, imp.values, color=PALETTE["model"])
    plt.title("Permutation feature importance (top 12)", fontsize=13, weight="bold")
    plt.xlabel("Increase in MAE when shuffled")
    plt.tight_layout()
    plt.savefig(FIG / "03_feature_importance.png", dpi=130)
    plt.close()


def _fig_error_by_category(test, y_test, pred_model) -> None:
    t = test.copy()
    t["actual"], t["model"] = y_test, pred_model
    t["ape"] = (t["actual"] - t["model"]).abs() / t["actual"].clip(lower=1) * 100
    by_cat = t.groupby("category", observed=True)["ape"].mean().sort_values()
    plt.figure(figsize=(8, 4.5))
    plt.barh(by_cat.index.astype(str), by_cat.values, color=PALETTE["actual"])
    plt.title("Model MAPE by category (lower is better)", fontsize=13, weight="bold")
    plt.xlabel("MAPE (%)")
    plt.tight_layout()
    plt.savefig(FIG / "04_error_by_category.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
