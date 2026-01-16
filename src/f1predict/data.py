from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

import fastf1
from fastf1.core import Session

from .config import PATHS, DEFAULT_SESSION
from .utils import ensure_dir


@dataclass(frozen=True)
class BuildConfig:
    year: int
    session: str = DEFAULT_SESSION  # "R" = Race
    max_events: Optional[int] = None
    include_sprint_events: bool = False  # keep False for simpler grouping


def _enable_fastf1_cache() -> None:
    ensure_dir(PATHS.cache_dir)
    fastf1.Cache.enable_cache(str(PATHS.cache_dir))


def _event_schedule(year: int) -> pd.DataFrame:
    # FastF1 schedule columns include RoundNumber, EventName, EventDate, EventFormat, etc.
    sched = fastf1.get_event_schedule(year)
    sched = sched.copy()
    # keep only races (exclude testing)
    if "EventFormat" in sched.columns:
        # EventFormat often contains "conventional", "sprint", etc.
        pass
    return sched


def _load_session(year: int, round_number: int, session_name: str) -> Session:
    session = fastf1.get_session(year, round_number, session_name)
    # Laps + weather + track status are what we rely on.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        session.load(laps=True, telemetry=False, weather=True)
    return session


def _safe_timedelta_to_seconds(x) -> float:
    if pd.isna(x):
        return np.nan
    try:
        return x.total_seconds()
    except Exception:
        return np.nan


def build_raw_laps(cfg: BuildConfig) -> pd.DataFrame:
    """
    Build a raw lap-level table for a season:
    - laps (timing + tyre + sectors)
    - merged weather (track temp etc) by nearest timestamp
    - merged track status by nearest timestamp
    """
    _enable_fastf1_cache()
    ensure_dir(PATHS.data_dir)

    sched = _event_schedule(cfg.year)

    # choose events: RoundNumber not null and session exists
    sched = sched.dropna(subset=["RoundNumber"])
    sched = sched.sort_values("RoundNumber")

    if cfg.max_events is not None:
        sched = sched.head(cfg.max_events)

    rows: List[pd.DataFrame] = []

    for _, ev in sched.iterrows():
        round_no = int(ev["RoundNumber"])
        event_name = str(ev.get("EventName", f"Round {round_no}"))

        try:
            session = _load_session(cfg.year, round_no, cfg.session)
        except Exception as e:
            print(f"[WARN] Skipping {cfg.year} round {round_no} ({event_name}): load failed: {e}")
            continue

        laps = session.laps
        if laps is None or len(laps) == 0:
            print(f"[WARN] No laps for {cfg.year} round {round_no} ({event_name})")
            continue

        laps_df = laps.copy()

        # basic columns we want to keep stable
        keep_cols = [
            "Driver", "Team", "LapNumber", "Stint", "Compound", "TyreLife",
            "FreshTyre", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
            "IsAccurate", "Deleted", "Position", "Time", "LapStartTime",
            "PitInTime", "PitOutTime",
        ]
        existing = [c for c in keep_cols if c in laps_df.columns]
        laps_df = laps_df[existing].copy()

        laps_df["year"] = cfg.year
        laps_df["round"] = round_no
        laps_df["event_name"] = event_name
        laps_df["session"] = cfg.session

        # Convert timedeltas to seconds for modeling
        for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Time", "LapStartTime", "PitInTime", "PitOutTime"]:
            if col in laps_df.columns:
                laps_df[col + "_s"] = laps_df[col].apply(_safe_timedelta_to_seconds)

        # Weather merge (nearest)
        weather = getattr(session, "weather_data", None)
        if weather is not None and len(weather) > 0 and "Time" in weather.columns:
            w = weather.copy()
            w = w.sort_values("Time")
            # weather Time is Timedelta since session start
            w["Time_s"] = w["Time"].apply(_safe_timedelta_to_seconds)
            # Merge on LapStartTime_s if available else Time_s (lap end time)
            key = "LapStartTime_s" if "LapStartTime_s" in laps_df.columns else "Time_s"
            if key in laps_df.columns:
                laps_df = laps_df.sort_values(key)
                laps_df = pd.merge_asof(
                    laps_df,
                    w[["Time_s", "TrackTemp", "AirTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]]
                      .rename(columns={"Time_s": key}),
                    on=key,
                    direction="nearest",
                )

        # Track status merge (nearest)
        track_status = getattr(session, "track_status", None)
        if track_status is not None and len(track_status) > 0 and "Time" in track_status.columns:
            ts = track_status.copy().sort_values("Time")
            ts["Time_s"] = ts["Time"].apply(_safe_timedelta_to_seconds)
            key = "LapStartTime_s" if "LapStartTime_s" in laps_df.columns else "Time_s"
            if key in laps_df.columns:
                laps_df = laps_df.sort_values(key)
                laps_df = pd.merge_asof(
                    laps_df,
                    ts[["Time_s", "Status"]].rename(columns={"Time_s": key, "Status": "track_status"}),
                    on=key,
                    direction="nearest",
                )

        rows.append(laps_df)

        print(f"[OK] Loaded {cfg.year} round {round_no:02d} {event_name}: {len(laps_df)} laps")

    if not rows:
        raise RuntimeError("No sessions were successfully loaded. Check year/session and your network/cache.")

    raw = pd.concat(rows, ignore_index=True)

    # Light cleaning
    raw = raw[raw.get("IsAccurate", True) != False].copy() if "IsAccurate" in raw.columns else raw
    raw = raw[raw.get("Deleted", False) != True].copy() if "Deleted" in raw.columns else raw

    # Ensure numeric types
    for c in ["LapNumber", "Stint", "TyreLife", "Position", "TrackTemp"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    return raw


def save_raw(raw: pd.DataFrame, year: int) -> str:
    ensure_dir(PATHS.data_dir)
    out = PATHS.data_dir / f"raw_laps_{year}.parquet"
    raw.to_parquet(out, index=False)
    return str(out)
