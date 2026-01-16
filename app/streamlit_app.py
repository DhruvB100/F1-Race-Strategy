from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

from f1predict.config import PATHS

st.set_page_config(page_title="F1 Pace & Pit Strategy Predictor", layout="wide")

@st.cache_data
def load_dataset(year: int) -> pd.DataFrame:
    path = PATHS.data_dir / f"dataset_{year}.parquet"
    return pd.read_parquet(path)

@st.cache_resource
def load_model(year: int):
    path = PATHS.models_dir / f"model_{year}.joblib"
    return joblib.load(path)

st.title("F1 Race Pace & Pit Strategy Prediction")

year = st.sidebar.number_input("Season year", min_value=2018, max_value=2030, value=2025, step=1)

model_file = PATHS.models_dir / f"model_{int(year)}.joblib"
data_file = PATHS.data_dir / f"dataset_{int(year)}.parquet"

if not model_file.exists() or not data_file.exists():
    st.warning("Model or dataset not found. Train first:\n\n`python -m f1predict.train --year 2025`")
    st.stop()

artifact = load_model(int(year))
df = load_dataset(int(year))

feature_cols = artifact["feature_cols"]
reg = artifact["regressor"]
clf = artifact["classifier"]
pit_k = artifact.get("pit_horizon_laps", 3)

st.sidebar.markdown("### Filters")
event = st.sidebar.selectbox("Event", sorted(df["event_name"].dropna().unique().tolist()))
drivers = sorted(df.loc[df["event_name"] == event, "Driver"].dropna().unique().tolist())
driver = st.sidebar.selectbox("Driver", drivers)

sub = df[(df["event_name"] == event) & (df["Driver"] == driver)].copy()
sub = sub.sort_values("LapNumber")

if sub.empty:
    st.error("No data for selection.")
    st.stop()

lap = st.sidebar.slider("Lap to inspect", int(sub["LapNumber"].min()), int(sub["LapNumber"].max()), int(sub["LapNumber"].min()))
row = sub[sub["LapNumber"] == lap].tail(1)
if row.empty:
    row = sub.tail(1)

row = row.iloc[0]

st.sidebar.markdown("### Scenario inputs (editable)")
compound = st.sidebar.selectbox("Compound", sorted(df["Compound"].dropna().unique().tolist()), index=0)
track_temp = st.sidebar.number_input("TrackTemp", value=float(row.get("TrackTemp", np.nan)) if pd.notna(row.get("TrackTemp", np.nan)) else 35.0)
stint_lap = st.sidebar.number_input("stint_lap", value=float(row.get("stint_lap", 5)))
prev_lap_time = st.sidebar.number_input("prev_lap_time_s", value=float(row.get("prev_lap_time_s", 90.0)))
prev_s1 = st.sidebar.number_input("prev_Sector1_s", value=float(row.get("prev_Sector1_s", 30.0)))
prev_s2 = st.sidebar.number_input("prev_Sector2_s", value=float(row.get("prev_Sector2_s", 30.0)))
prev_s3 = st.sidebar.number_input("prev_Sector3_s", value=float(row.get("prev_Sector3_s", 30.0)))
prev_gap = st.sidebar.number_input("prev_gap_to_ahead_s", value=float(row.get("prev_gap_to_ahead_s", 1.0)))

# Build one-row input with required columns
x = {c: row.get(c, np.nan) for c in feature_cols}
x["Compound"] = compound
x["TrackTemp"] = track_temp
x["stint_lap"] = stint_lap
x["prev_lap_time_s"] = prev_lap_time
if "prev_Sector1_s" in x: x["prev_Sector1_s"] = prev_s1
if "prev_Sector2_s" in x: x["prev_Sector2_s"] = prev_s2
if "prev_Sector3_s" in x: x["prev_Sector3_s"] = prev_s3
if "prev_gap_to_ahead_s" in x: x["prev_gap_to_ahead_s"] = prev_gap

X1 = pd.DataFrame([x])[feature_cols]

pred_lap = float(reg.predict(X1)[0])
pred_pit = float(clf.predict_proba(X1)[0, 1])

c1, c2, c3 = st.columns(3)
c1.metric("Predicted next-lap pace (s)", f"{pred_lap:.3f}")
c2.metric(f"P(pit within next {pit_k} laps)", f"{pred_pit:.3f}")
c3.metric("Reference lap (actual lap_time_s)", f"{float(row.get('lap_time_s', np.nan)):.3f}")

st.markdown("---")

left, right = st.columns([1.2, 1.0])

with left:
    st.subheader("Actual vs Predicted (selected driver)")
    # Predict across all laps for this driver/event using their recorded features
    Xall = sub[feature_cols].copy()
    sub = sub.copy()
    sub["pred_lap_time_s"] = reg.predict(Xall)
    sub["pred_pit_prob"] = clf.predict_proba(Xall)[:, 1]

    fig = plt.figure()
    plt.plot(sub["LapNumber"], sub["lap_time_s"], label="actual")
    plt.plot(sub["LapNumber"], sub["pred_lap_time_s"], label="predicted")
    plt.xlabel("Lap")
    plt.ylabel("Lap time (s)")
    plt.legend()
    st.pyplot(fig)

    fig2 = plt.figure()
    plt.plot(sub["LapNumber"], sub["pred_pit_prob"])
    plt.xlabel("Lap")
    plt.ylabel(f"P(pit within {pit_k} laps)")
    st.pyplot(fig2)

with right:
    st.subheader("Lap-level table (tail)")
    show_cols = ["LapNumber", "Compound", "Stint", "stint_lap", "TrackTemp", "prev_lap_time_s", "lap_time_s", "pred_lap_time_s", "pred_pit_prob", "pit_this_lap", "pit_within_k"]
    show_cols = [c for c in show_cols if c in sub.columns]
    st.dataframe(sub[show_cols].tail(25), use_container_width=True)

st.caption("Tip: Train on more events (or full season) to improve stability.")
