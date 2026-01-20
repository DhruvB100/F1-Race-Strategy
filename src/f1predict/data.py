import os
import warnings

import numpy as np
import pandas as pd
import fastf1

from .config import DATA_DIR, CACHE_DIR


def load_season_data(year, session_type="R", max_events=None):
    """
    Download and combine all race lap data for a given season using the FastF1 API.

    Parameters:
        year: season year (e.g. 2024)
        session_type: "R" for race, "Q" for qualifying, etc.
        max_events: if set, only load the first N events (useful for testing)

    Returns:
        A big DataFrame with one row per lap, with timing + weather + track status info
    """
    # Set up local cache so we don't re-download the same data
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # Get the list of races for this season
    schedule = fastf1.get_event_schedule(year)
    schedule = schedule.dropna(subset=["RoundNumber"])
    schedule = schedule.sort_values("RoundNumber").reset_index(drop=True)

    if max_events is not None:
        schedule = schedule.head(max_events)

    print(f"Found {len(schedule)} events for {year}")

    all_laps = []

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        event_name = str(event.get("EventName", f"Round {round_num}"))

        print(f"Loading Round {round_num}: {event_name}...")

        # Try to load session data, skip if it fails
        try:
            session = fastf1.get_session(year, round_num, session_type)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                session.load(laps=True, telemetry=False, weather=True)
        except Exception as e:
            print(f"  Could not load - skipping ({e})")
            continue

        laps = session.laps
        if laps is None or len(laps) == 0:
            print(f"  No laps found, skipping")
            continue

        laps_df = laps.copy()

        # Only keep columns we actually need
        cols_to_keep = [
            "Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife",
            "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "IsAccurate", "Deleted", "Position", "Time", "LapStartTime",
            "PitInTime", "PitOutTime",
        ]
        cols_to_keep = [c for c in cols_to_keep if c in laps_df.columns]
        laps_df = laps_df[cols_to_keep].copy()

        # Tag each lap with which race it came from
        laps_df["year"] = year
        laps_df["round"] = round_num
        laps_df["event_name"] = event_name

        # sklearn can't work with timedelta objects, so convert to seconds
        time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                     "Time", "LapStartTime", "PitInTime", "PitOutTime"]
        for col in time_cols:
            if col in laps_df.columns:
                laps_df[col + "_s"] = laps_df[col].apply(
                    lambda x: x.total_seconds() if pd.notna(x) else np.nan
                )

        # Merge in weather data (track temp, humidity, etc.)
        # We match each lap to the closest weather reading by time
        weather = getattr(session, "weather_data", None)
        if weather is not None and len(weather) > 0 and "Time" in weather.columns:
            w = weather.copy()
            w["Time_s"] = w["Time"].apply(
                lambda x: x.total_seconds() if pd.notna(x) else np.nan
            )
            w = w.sort_values("Time_s")

            # prefer lap start time for matching
            merge_key = "LapStartTime_s" if "LapStartTime_s" in laps_df.columns else "Time_s"
            if merge_key in laps_df.columns:
                laps_df = laps_df.sort_values(merge_key)
                weather_cols = [c for c in ["TrackTemp", "AirTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
                                if c in w.columns]
                laps_df = pd.merge_asof(
                    laps_df,
                    w[["Time_s"] + weather_cols].rename(columns={"Time_s": merge_key}),
                    on=merge_key,
                    direction="nearest",
                )

        # Merge in track status (green flag, yellow flag, safety car, etc.)
        track_status = getattr(session, "track_status", None)
        if track_status is not None and len(track_status) > 0 and "Time" in track_status.columns:
            ts = track_status.copy()
            ts["Time_s"] = ts["Time"].apply(
                lambda x: x.total_seconds() if pd.notna(x) else np.nan
            )
            ts = ts.sort_values("Time_s")

            merge_key = "LapStartTime_s" if "LapStartTime_s" in laps_df.columns else "Time_s"
            if merge_key in laps_df.columns:
                laps_df = laps_df.sort_values(merge_key)
                laps_df = pd.merge_asof(
                    laps_df,
                    ts[["Time_s", "Status"]].rename(columns={"Time_s": merge_key, "Status": "track_status"}),
                    on=merge_key,
                    direction="nearest",
                )

        all_laps.append(laps_df)
        print(f"  Loaded {len(laps_df)} laps")

    if not all_laps:
        raise RuntimeError(
            "No session data could be loaded. Check your internet connection and try again."
        )

    # Combine all races into one big dataframe
    raw = pd.concat(all_laps, ignore_index=True)

    # Remove laps that were flagged as inaccurate or deleted
    if "IsAccurate" in raw.columns:
        raw = raw[raw["IsAccurate"] != False].copy()
    if "Deleted" in raw.columns:
        raw = raw[raw["Deleted"] != True].copy()

    # Make sure these columns are numeric (sometimes they come in as objects)
    for col in ["LapNumber", "Stint", "TyreLife", "Position", "TrackTemp"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    print(f"\nTotal laps loaded: {len(raw)}")
    return raw


def save_raw_laps(df, year):
    """Save the raw lap data to a parquet file"""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = DATA_DIR / f"raw_laps_{year}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved raw laps to: {path}")
    return path
