from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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


def _laps_to_next_pit(sub: pd.DataFrame) -> pd.Series:
    """
    For each lap, compute how many laps until the next pit lap.
    Returns None if no future pit exists.
    """
    sub = sub.sort_values("LapNumber").copy()
    pit_laps = sub.loc[sub["pit_this_lap"] == 1, "LapNumber"].to_numpy()

    def laps_to_next_pit(lap: int):
        future = pit_laps[pit_laps > lap]
        return int(future[0] - lap) if future.size > 0 else None

    return sub["LapNumber"].apply(laps_to_next_pit)


def main():
    st.title("F1 Race Pace & Pit Strategy Prediction")

    # Sidebar: year
    year = int(st.sidebar.number_input("Season year", min_value=2018, max_value=2030, value=2025, step=1))

    model_file = PATHS.models_dir / f"model_{year}.joblib"
    data_file = PATHS.data_dir / f"dataset_{year}.parquet"

    if not model_file.exists() or not data_file.exists():
        st.warning("Model or dataset not found. Train first:\n\n`python -m f1predict.train --year 2025`")
        st.stop()

    artifact = load_model(year)
    df = load_dataset(year)

    feature_cols = artifact["feature_cols"]
    reg = artifact["regressor"]
    clf = artifact["classifier"]
    pit_k = int(artifact.get("pit_horizon_laps", 3))

    # Sidebar filters
    st.sidebar.markdown("## Filters")

    events = sorted(df["event_name"].dropna().unique().tolist())
    if not events:
        st.error("No events in dataset.")
        st.stop()

    event = st.sidebar.selectbox("Event", events)

    drivers = sorted(df.loc[df["event_name"] == event, "Driver"].dropna().unique().tolist())
    if not drivers:
        st.error("No drivers for selected event.")
        st.stop()

    driver = st.sidebar.selectbox("Driver", drivers)

    sub = df[(df["event_name"] == event) & (df["Driver"] == driver)].copy()
    sub = sub.sort_values("LapNumber")

    if sub.empty:
        st.error("No data for selection.")
        st.stop()

    lap_min = int(sub["LapNumber"].min())
    lap_max = int(sub["LapNumber"].max())
    lap = int(st.sidebar.slider("Lap to inspect", lap_min, lap_max, lap_min))

    row_df = sub[sub["LapNumber"] == lap].tail(1)
    if row_df.empty:
        row_df = sub.tail(1)
    row = row_df.iloc[0]

    # Scenario inputs
    st.sidebar.markdown("## Scenario inputs (editable)")

    compounds = sorted(df["Compound"].dropna().unique().tolist())
    if not compounds:
        compounds = ["SOFT", "MEDIUM", "HARD"]

    default_compound = row.get("Compound", compounds[0])
    if default_compound not in compounds:
        default_compound = compounds[0]

    compound = st.sidebar.selectbox("Compound", compounds, index=compounds.index(default_compound))

    track_temp_default = float(row.get("TrackTemp", np.nan)) if pd.notna(row.get("TrackTemp", np.nan)) else 35.0
    track_temp = float(st.sidebar.number_input("TrackTemp", value=track_temp_default))

    stint_lap_default = float(row.get("stint_lap", 5))
    stint_lap = float(st.sidebar.number_input("stint_lap", value=stint_lap_default))

    prev_lap_time_default = float(row.get("prev_lap_time_s", 90.0))
    prev_lap_time = float(st.sidebar.number_input("prev_lap_time_s", value=prev_lap_time_default))

    prev_s1_default = float(row.get("prev_Sector1_s", 30.0))
    prev_s2_default = float(row.get("prev_Sector2_s", 30.0))
    prev_s3_default = float(row.get("prev_Sector3_s", 30.0))

    prev_s1 = float(st.sidebar.number_input("prev_Sector1_s", value=prev_s1_default))
    prev_s2 = float(st.sidebar.number_input("prev_Sector2_s", value=prev_s2_default))
    prev_s3 = float(st.sidebar.number_input("prev_Sector3_s", value=prev_s3_default))

    prev_gap_default = float(row.get("prev_gap_to_ahead_s", 1.0))
    prev_gap = float(st.sidebar.number_input("prev_gap_to_ahead_s", value=prev_gap_default))

    # Build one-row input with required columns
    x = {c: row.get(c, np.nan) for c in feature_cols}
    if "Compound" in x:
        x["Compound"] = compound
    if "TrackTemp" in x:
        x["TrackTemp"] = track_temp
    if "stint_lap" in x:
        x["stint_lap"] = stint_lap
    if "prev_lap_time_s" in x:
        x["prev_lap_time_s"] = prev_lap_time
    if "prev_Sector1_s" in x:
        x["prev_Sector1_s"] = prev_s1
    if "prev_Sector2_s" in x:
        x["prev_Sector2_s"] = prev_s2
    if "prev_Sector3_s" in x:
        x["prev_Sector3_s"] = prev_s3
    if "prev_gap_to_ahead_s" in x:
        x["prev_gap_to_ahead_s"] = prev_gap

    X1 = pd.DataFrame([x])[feature_cols]

    pred_lap = float(reg.predict(X1)[0])
    pred_pit = float(clf.predict_proba(X1)[0, 1])

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted next-lap pace (s)", f"{pred_lap:.3f}")
    c2.metric(f"P(pit within next {pit_k} laps)", f"{pred_pit:.3f}")
    c3.metric("Reference lap (actual lap_time_s)", f"{float(row.get('lap_time_s', np.nan)):.3f}")

    st.markdown("---")

    left, right = st.columns([1.2, 1.0])

    # Predict across all laps for selected driver/event
    Xall = sub[feature_cols].copy()
    sub = sub.copy()
    sub["pred_lap_time_s"] = reg.predict(Xall)
    sub["pred_pit_prob"] = clf.predict_proba(Xall)[:, 1]

    # Interpretability columns
    if "pit_this_lap" in sub.columns:
        sub["laps_to_next_pit"] = _laps_to_next_pit(sub)
    else:
        sub["laps_to_next_pit"] = None

    with left:
        st.subheader("Actual vs Predicted (selected driver)")

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

        st.subheader("Top predicted pit opportunities (this driver/event)")
        topk = sub.sort_values("pred_pit_prob", ascending=False).head(12)

        show_cols = [
            "LapNumber",
            "pred_pit_prob",
            "pit_within_k",
            "laps_to_next_pit",
            "pit_this_lap",
            "Compound",
            "Stint",
            "stint_lap",
            "TrackTemp",
        ]
        show_cols = [c for c in show_cols if c in topk.columns]
        st.dataframe(topk[show_cols], use_container_width=True)

    with right:
        st.subheader("Lap-level table (tail)")
        show_cols = [
            "LapNumber",
            "Compound",
            "Stint",
            "stint_lap",
            "TrackTemp",
            "prev_lap_time_s",
            "lap_time_s",
            "pred_lap_time_s",
            "pred_pit_prob",
            "pit_this_lap",
            "pit_within_k",
            "laps_to_next_pit",
        ]
        show_cols = [c for c in show_cols if c in sub.columns]
        st.dataframe(sub[show_cols].tail(25), use_container_width=True)

    st.caption("Tip: Train on more events (or full season) to improve stability and pit-window learning.")


if __name__ == "__main__":
    main()
