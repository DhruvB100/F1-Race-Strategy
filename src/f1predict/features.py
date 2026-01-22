import numpy as np
import pandas as pd


def compute_gap_to_car_ahead(df):
    """
    Estimate how far behind the car ahead a driver is (in seconds).

    We do this by grouping all drivers on the same lap together, sorting by
    position, and taking the difference in elapsed time (Time_s).
    The leader will have NaN since there's nobody ahead.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)

    required = {"round", "event_name", "LapNumber", "Position", "Time_s"}
    if not required.issubset(df.columns):
        return result

    for (round_num, event, lap_num), group in df.groupby(
        ["round", "event_name", "LapNumber"], sort=False
    ):
        group = group.dropna(subset=["Position", "Time_s"]).copy()
        if len(group) == 0:
            continue
        group = group.sort_values("Position")
        # diff gives: my_time - car_ahead_time
        gaps = group["Time_s"].diff()
        result.loc[group.index] = gaps.values

    return result


def make_features(raw_df, pit_horizon=3):
    """
    Takes the raw lap data and builds features + labels for ML training.

    The key idea: we can only use information that we'd know at the START of a lap
    (to avoid data leakage). So we use previous lap info, not current lap.

    Features:
        - prev_lap_time_s: how long the previous lap took
        - prev_Sector1/2/3_s: previous sector times
        - prev_gap_to_ahead_s: how far behind car ahead we were last lap
        - stint_lap: how old the tires are (laps on this set)
        - Stint: which stint number we're in
        - TrackTemp, AirTemp, Humidity, Rainfall: weather conditions
        - Position: current position
        - Compound: tire type (SOFT, MEDIUM, HARD)
        - Driver, Team, event_name: categorical identifiers
        - track_status: safety car, yellow flag, etc.

    Targets:
        - lap_time_s: the actual lap time (regression)
        - pit_within_k: will this driver pit in the next K laps? (classification, binary 0/1)
    """
    df = raw_df.copy()

    # Make sure basic columns exist (fill with NaN if missing)
    for col in ["year", "round", "event_name", "Driver", "Team", "LapNumber", "Stint", "Compound"]:
        if col not in df.columns:
            df[col] = np.nan

    # Rename the lap time column to something cleaner
    if "LapTime_s" in df.columns and "lap_time_s" not in df.columns:
        df = df.rename(columns={"LapTime_s": "lap_time_s"})

    # Sort by driver and lap number within each race
    df = df.sort_values(["year", "round", "event_name", "Driver", "LapNumber"]).reset_index(drop=True)

    # These columns identify a unique driver-race combination
    group_keys = ["year", "round", "event_name", "Driver"]

    # --- Figure out if/when a driver pits ---
    # Method 1: the stint number goes up on the next lap (= must have pitted)
    next_stint = df.groupby(group_keys)["Stint"].shift(-1)
    pitted_by_stint = (next_stint > df["Stint"]).fillna(False)

    # Method 2: PitInTime or PitOutTime is recorded for this lap
    has_pit_in = df["PitInTime_s"].notna() if "PitInTime_s" in df.columns else pd.Series(False, index=df.index)
    has_pit_out = df["PitOutTime_s"].notna() if "PitOutTime_s" in df.columns else pd.Series(False, index=df.index)

    df["pit_this_lap"] = (pitted_by_stint | has_pit_in | has_pit_out).astype(int)

    # --- Compute traffic gap ---
    if "Time_s" in df.columns and "Position" in df.columns:
        df["gap_to_ahead_s"] = compute_gap_to_car_ahead(df)
    else:
        df["gap_to_ahead_s"] = np.nan

    # Sort again after the gap calculation
    df = df.sort_values(["year", "round", "event_name", "Driver", "LapNumber"]).reset_index(drop=True)

    # --- Previous lap features ---
    # shift(1) gives us the value from the previous row within the same group
    df["prev_lap_time_s"] = df.groupby(group_keys)["lap_time_s"].shift(1)
    df["prev_gap_to_ahead_s"] = df.groupby(group_keys)["gap_to_ahead_s"].shift(1)

    for sector_col in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]:
        # rename e.g. "prev_Sector1Time_s" -> "prev_Sector1_s"
        feature_name = "prev_" + sector_col.replace("Time_s", "s")
        if sector_col in df.columns:
            df[feature_name] = df.groupby(group_keys)[sector_col].shift(1)
        else:
            df[feature_name] = np.nan

    # --- Tire age ---
    # TyreLife is how many laps this set has been on (provided by FastF1)
    if "TyreLife" in df.columns and df["TyreLife"].notna().any():
        df["stint_lap"] = df["TyreLife"]
    else:
        # fallback: count laps within each stint manually
        df["stint_lap"] = df.groupby(group_keys + ["Stint"]).cumcount() + 1

    # --- Pit window label ---
    # pit_within_k = 1 if the driver pits in the NEXT pit_horizon laps
    # We look forward in time for each driver, not backward
    df["pit_within_k"] = 0

    for _, group in df.groupby(group_keys, sort=False):
        idx = group.index
        pit_flags = group["pit_this_lap"].to_numpy()
        future_pit = np.zeros(len(pit_flags), dtype=int)

        for i in range(len(pit_flags)):
            # look forward pit_horizon steps (not including current lap)
            look_ahead = pit_flags[i + 1: i + pit_horizon + 1]
            if len(look_ahead) > 0 and look_ahead.max() > 0:
                future_pit[i] = 1

        df.loc[idx, "pit_within_k"] = future_pit

    # --- Clean up ---
    # Remove rows where we don't have a target lap time
    df = df[df["lap_time_s"].notna()].copy()

    # Remove the first lap per driver (no prev_lap_time_s available)
    df = df[df["prev_lap_time_s"].notna()].copy()

    # Gaps can't be negative (timing artifacts), clip to 0
    for col in ["prev_gap_to_ahead_s", "gap_to_ahead_s"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    return df
