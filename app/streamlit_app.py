import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from f1predict.config import DATA_DIR, MODELS_DIR

st.set_page_config(page_title="F1 Race Strategy Predictor", layout="wide")
st.title("F1 Race Pace & Pit Strategy Predictor")
st.caption("Uses machine learning to predict lap times and pit stop windows")


# ---- Cached data loading ----

@st.cache_data
def load_dataset(year):
    path = DATA_DIR / f"dataset_{year}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_resource
def load_model(year):
    path = MODELS_DIR / f"model_{year}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# ---- Sidebar: settings ----

st.sidebar.header("Settings")

year = int(st.sidebar.number_input("Season Year", min_value=2018, max_value=2030, value=2025, step=1))

# Check if we have trained data for this year
model_path = MODELS_DIR / f"model_{year}.joblib"
data_path = DATA_DIR / f"dataset_{year}.parquet"

if not model_path.exists() or not data_path.exists():
    st.warning(f"No model or dataset found for {year}.")
    st.info("Train the model first by running:")
    st.code(f"python -m f1predict.train --year {year}")
    st.stop()

# Load model and dataset
artifact = load_model(year)
df = load_dataset(year)

if artifact is None or df is None:
    st.error("Could not load model or dataset. Try retraining.")
    st.stop()

feature_cols = artifact["feature_cols"]
reg_model = artifact["regressor"]
clf_model = artifact["classifier"]
pit_k = int(artifact.get("pit_horizon_laps", 3))

# ---- Sidebar: race/driver selection ----

st.sidebar.header("Select Race")

events = sorted(df["event_name"].dropna().unique().tolist())
selected_event = st.sidebar.selectbox("Grand Prix", events)

drivers = sorted(df[df["event_name"] == selected_event]["Driver"].dropna().unique().tolist())
if not drivers:
    st.error("No drivers found for this event.")
    st.stop()

selected_driver = st.sidebar.selectbox("Driver", drivers)

# Filter to this driver's laps at this event
driver_laps = df[
    (df["event_name"] == selected_event) & (df["Driver"] == selected_driver)
].copy()
driver_laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)

if driver_laps.empty:
    st.error("No data available for this driver/event combination.")
    st.stop()

lap_min = int(driver_laps["LapNumber"].min())
lap_max = int(driver_laps["LapNumber"].max())
selected_lap = int(st.sidebar.slider("Inspect Lap", lap_min, lap_max, lap_min))

# Get data for the selected lap
lap_rows = driver_laps[driver_laps["LapNumber"] == selected_lap]
if lap_rows.empty:
    lap_rows = driver_laps.tail(1)
current_row = lap_rows.iloc[0]

# ---- Sidebar: scenario editor ----

st.sidebar.header("Edit Scenario")
st.sidebar.caption("Change values to see how predictions shift")

compounds = sorted(df["Compound"].dropna().unique().tolist())
if not compounds:
    compounds = ["SOFT", "MEDIUM", "HARD"]

default_compound = current_row.get("Compound", compounds[0])
if default_compound not in compounds:
    default_compound = compounds[0]

compound = st.sidebar.selectbox("Tire Compound", compounds, index=compounds.index(default_compound))

default_track_temp = float(current_row.get("TrackTemp", 35.0))
if np.isnan(default_track_temp):
    default_track_temp = 35.0
track_temp = float(st.sidebar.number_input("Track Temp (°C)", value=default_track_temp))

default_stint_lap = float(current_row.get("stint_lap", 1.0))
if np.isnan(default_stint_lap):
    default_stint_lap = 1.0
stint_lap = float(st.sidebar.number_input("Tyre Age (laps)", value=default_stint_lap, min_value=0.0))

default_prev_lap = float(current_row.get("prev_lap_time_s", 90.0))
if np.isnan(default_prev_lap):
    default_prev_lap = 90.0
prev_lap_time = float(st.sidebar.number_input("Previous Lap Time (s)", value=default_prev_lap))

# ---- Predict for the edited scenario ----

# Build a one-row dataframe with the scenario values
scenario = {col: current_row.get(col, np.nan) for col in feature_cols}
scenario["Compound"] = compound
scenario["TrackTemp"] = track_temp
scenario["stint_lap"] = stint_lap
scenario["prev_lap_time_s"] = prev_lap_time

X_scenario = pd.DataFrame([scenario])[feature_cols]

pred_lap_time = float(reg_model.predict(X_scenario)[0])
pred_pit_prob = float(clf_model.predict_proba(X_scenario)[0, 1])
actual_lap_time = float(current_row.get("lap_time_s", np.nan))

# ---- Main area: top metrics ----

col1, col2, col3 = st.columns(3)

col1.metric("Predicted Lap Time (s)", f"{pred_lap_time:.3f}")
col2.metric(f"Pit Probability (next {pit_k} laps)", f"{pred_pit_prob:.1%}")

if pd.notna(actual_lap_time):
    diff = pred_lap_time - actual_lap_time
    col3.metric("Actual Lap Time (s)", f"{actual_lap_time:.3f}", delta=f"{diff:+.3f}s")
else:
    col3.metric("Actual Lap Time (s)", "N/A")

st.markdown("---")

# ---- Predict for all laps ----

X_all = driver_laps[feature_cols].copy()
driver_laps["pred_lap_time_s"] = reg_model.predict(X_all)
driver_laps["pred_pit_prob"] = clf_model.predict_proba(X_all)[:, 1]

# Compute laps until next pit stop (for context)
if "pit_this_lap" in driver_laps.columns:
    pit_laps = driver_laps[driver_laps["pit_this_lap"] == 1]["LapNumber"].to_numpy()

    def laps_until_next_pit(lap_num):
        future = pit_laps[pit_laps > lap_num]
        return int(future[0] - lap_num) if len(future) > 0 else None

    driver_laps["laps_to_next_pit"] = driver_laps["LapNumber"].apply(laps_until_next_pit)

# ---- Charts and tables ----

left_col, right_col = st.columns([1.2, 1.0])

with left_col:
    st.subheader(f"Lap Times - {selected_driver} at {selected_event}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(driver_laps["LapNumber"], driver_laps["lap_time_s"],
            label="Actual", color="steelblue", linewidth=2)
    ax.plot(driver_laps["LapNumber"], driver_laps["pred_lap_time_s"],
            label="Predicted", color="orange", linewidth=2, linestyle="--")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (seconds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Pit Stop Probability Over the Race")

    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.fill_between(driver_laps["LapNumber"], driver_laps["pred_pit_prob"],
                     alpha=0.3, color="red")
    ax2.plot(driver_laps["LapNumber"], driver_laps["pred_pit_prob"],
             color="red", linewidth=2)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="50% threshold")
    ax2.set_xlabel("Lap Number")
    ax2.set_ylabel(f"P(pit in next {pit_k} laps)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Laps with Highest Pit Probability")
    top_pit_laps = driver_laps.sort_values("pred_pit_prob", ascending=False).head(10)
    show_cols = ["LapNumber", "pred_pit_prob", "Compound", "stint_lap", "pit_this_lap"]
    show_cols = [c for c in show_cols if c in top_pit_laps.columns]
    st.dataframe(top_pit_laps[show_cols].reset_index(drop=True), use_container_width=True)

with right_col:
    st.subheader("All Laps")
    table_cols = [
        "LapNumber", "Compound", "Stint", "stint_lap",
        "lap_time_s", "pred_lap_time_s", "pred_pit_prob",
        "pit_this_lap", "pit_within_k",
    ]
    table_cols = [c for c in table_cols if c in driver_laps.columns]
    st.dataframe(driver_laps[table_cols].reset_index(drop=True), use_container_width=True)

    if "metrics" in artifact:
        st.subheader("Model Performance (Cross-validation)")
        reg_m = artifact["metrics"].get("regression", {})
        clf_m = artifact["metrics"].get("classification", {})
        if reg_m:
            st.write(f"Lap time MAE: **{reg_m.get('mae_mean', 0):.3f}s** "
                     f"(± {reg_m.get('mae_std', 0):.3f}s)")
        if clf_m:
            st.write(f"Pit window F1: **{clf_m.get('f1_mean', 0):.3f}** "
                     f"(± {clf_m.get('f1_std', 0):.3f})")


if __name__ == "__main__":
    pass
