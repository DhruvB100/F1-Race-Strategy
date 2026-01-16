from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    pit_horizon_laps: int = 3


def _gap_to_ahead_seconds(df: pd.DataFrame) -> pd.Series:
    """
    Approximate traffic gap as (your cumulative time - ahead car cumulative time) within same lap & round.
    Uses Position ordering (1 = leader). If missing, returns NaNs.
    """
    if not {"round", "event_name", "LapNumber", "Position", "Time_s"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)

    # group by race+lap, sort by position ascending
    out = pd.Series(np.nan, index=df.index, dtype=float)

    for (_, _, lap_no), g in df.groupby(["round", "event_name", "LapNumber"], sort=False):
        g2 = g.dropna(subset=["Position", "Time_s"]).copy()
        if len(g2) == 0:
            continue
        g2 = g2.sort_values("Position")
        # gap = your Time_s - ahead Time_s
        gap = g2["Time_s"] - g2["Time_s"].shift(1)
        out.loc[g2.index] = gap.values

    return out


def add_features_and_labels(raw: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """
    Produce a modeling dataset where:
    - Regression target: lap_time_s (LapTime_s)
    - Classification target: pit_within_k (pit in next K laps)

    Features use ONLY information that would be known at start of the lap:
    - prev lap time, prev sectors, prev traffic gap
    - compound, stint, tyre life, track temp, track status, etc.
    """
    df = raw.copy()

    # Basic required columns
    required = ["year", "round", "event_name", "Driver", "Team", "LapNumber", "Stint", "Compound"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # Targets
    if "LapTime_s" not in df.columns and "LapTime" in df.columns:
        # If someone uses a different raw builder
        df["LapTime_s"] = df["LapTime"].dt.total_seconds()

    df = df.rename(columns={"LapTime_s": "lap_time_s"})

    # A lap is considered a pit lap if PitInTime exists (car pitted at end of this lap).
    pit_lap = df["PitInTime_s"].notna() if "PitInTime_s" in df.columns else pd.Series(False, index=df.index)
    df["pit_this_lap"] = pit_lap.astype(int)

    # Traffic gap computed from cumulative Time_s; then shift to prev lap to avoid leakage
    if "Time_s" in df.columns and "Position" in df.columns:
        df["gap_to_ahead_s"] = _gap_to_ahead_seconds(df)
    else:
        df["gap_to_ahead_s"] = np.nan

    # Sort for shifting
    df = df.sort_values(["year", "round", "event_name", "Driver", "LapNumber"]).reset_index(drop=True)

    # Previous-lap features
    group_keys = ["year", "round", "event_name", "Driver"]
    df["prev_lap_time_s"] = df.groupby(group_keys)["lap_time_s"].shift(1)
    for s in ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]:
        if s in df.columns:
            df["prev_" + s.replace("Time_s", "s")] = df.groupby(group_keys)[s].shift(1)
        else:
            df["prev_" + s.replace("Time_s", "s")] = np.nan

    df["prev_gap_to_ahead_s"] = df.groupby(group_keys)["gap_to_ahead_s"].shift(1)

    # Stint lap: prefer TyreLife if present, else compute within stint
    if "TyreLife" in df.columns and df["TyreLife"].notna().any():
        df["stint_lap"] = df["TyreLife"]
    else:
        df["stint_lap"] = df.groupby(group_keys + ["Stint"]).cumcount() + 1

    # Pit window label: pit within next K laps (excluding current lap)
    k = int(cfg.pit_horizon_laps)
    # For each driver in a race, create future pit indicator using rolling max of shifted pit flags
    df["pit_within_k"] = 0
    for _, g in df.groupby(group_keys, sort=False):
        idx = g.index
        pit_flags = g["pit_this_lap"].to_numpy()
        future = np.zeros_like(pit_flags)
        # future[i] = max(pit_flags[i+1 : i+k+1])
        for i in range(len(pit_flags)):
            j1 = i + 1
            j2 = min(len(pit_flags), i + k + 1)
            future[i] = 1 if pit_flags[j1:j2].max(initial=0) > 0 else 0
        df.loc[idx, "pit_within_k"] = future

    # Light filtering:
    # - remove laps without target lap_time_s
    df = df[df["lap_time_s"].notna()].copy()

    # - remove the first lap per driver (prev features are NaN)
    df = df[df["prev_lap_time_s"].notna()].copy()

    # - remove weird negatives
    for c in ["prev_gap_to_ahead_s", "gap_to_ahead_s"]:
        df[c] = df[c].clip(lower=0)

    return df
