"""Interactive sales forecasting dashboard.

Run from the project root:
    streamlit run dashboard/app.py

Lets the user pick a store + category, inspect historical sales and KPIs,
choose a forecast horizon, and see the model project future demand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast import forecast_series  # noqa: E402

DATA = ROOT / "data" / "sales_data.csv"
MODEL = ROOT / "models" / "gbm_model.joblib"
COLS = ROOT / "models" / "feature_cols.json"

st.set_page_config(page_title="Sales Forecasting Platform", page_icon="📈", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA, parse_dates=["date"])


@st.cache_resource
def load_model():
    model = joblib.load(MODEL)
    cols = json.loads(COLS.read_text())
    return model, cols


def main() -> None:
    st.title("📈 Sales Forecasting Analytics Platform")
    st.caption("Daily demand forecasting across stores and product categories.")

    if not MODEL.exists():
        st.error("Model not found. Run `python src/train.py` first.")
        st.stop()

    df = load_data()
    model, cols = load_model()

    # ---- sidebar controls -------------------------------------------------
    st.sidebar.header("Controls")
    store = st.sidebar.selectbox("Store", sorted(df["store"].unique()))
    category = st.sidebar.selectbox("Category", sorted(df["category"].unique()))
    horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30, step=7)

    sub = df[(df["store"] == store) & (df["category"] == category)].sort_values("date")

    # ---- KPIs -------------------------------------------------------------
    last_90 = sub.tail(90)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg daily units (90d)", f"{last_90['units_sold'].mean():,.0f}")
    c2.metric("Peak day (90d)", f"{last_90['units_sold'].max():,.0f}")
    c3.metric("Promo days (90d)", int(last_90["promo"].sum()))
    yoy = sub.tail(365)["units_sold"].sum()
    c4.metric("Trailing 365d units", f"{yoy:,.0f}")

    # ---- forecast ---------------------------------------------------------
    with st.spinner(f"Forecasting next {horizon} days..."):
        fc = forecast_series(model, df, store, category, horizon, cols)

    hist = sub.tail(180)[["date", "units_sold"]].rename(columns={"units_sold": "history"})
    fc_plot = fc[["date", "forecast"]]
    chart_df = pd.merge(hist, fc_plot, on="date", how="outer").set_index("date").sort_index()

    st.subheader(f"{store} — {category}: last 180 days + {horizon}-day forecast")
    st.line_chart(chart_df, color=["#1b3a4b", "#e07a5f"])

    fc1, fc2 = st.columns(2)
    fc1.metric("Forecast total", f"{fc['forecast'].sum():,.0f} units")
    fc1.metric("Forecast daily avg", f"{fc['forecast'].mean():,.0f} units")
    with fc2:
        st.write("**Forecast detail**")
        st.dataframe(fc[["date", "forecast"]], hide_index=True, use_container_width=True)

    st.download_button(
        "Download forecast (CSV)",
        fc.to_csv(index=False).encode(),
        file_name=f"forecast_{store}_{category}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
