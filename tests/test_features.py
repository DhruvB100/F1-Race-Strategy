import numpy as np
import pandas as pd
from f1predict.features import make_features


def test_pit_within_k_label():
    """
    Test that pit window labels are correct.

    Setup: 1 driver, 6 laps, pits on lap 4 (PitInTime_s is recorded for that lap).
    With pit_horizon=2, we expect:
    - Lap 2: looks ahead to laps 3 and 4 -> pit is on lap 4 -> label = 1
    - Lap 3: looks ahead to laps 4 and 5 -> pit is on lap 4 -> label = 1
    - Lap 4: looks ahead to laps 5 and 6 -> no pit -> label = 0
    """
    raw = pd.DataFrame({
        "year": [2025] * 6,
        "round": [1] * 6,
        "event_name": ["TestGP"] * 6,
        "Driver": ["AAA"] * 6,
        "Team": ["TeamX"] * 6,
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "Stint": [1, 1, 1, 1, 2, 2],
        "Compound": ["SOFT"] * 6,
        "lap_time_s": [90.0, 91.0, 92.0, 93.0, 94.0, 95.0],
        "PitInTime_s": [np.nan, np.nan, np.nan, 1000.0, np.nan, np.nan],
        "Time_s": [90, 181, 273, 366, 460, 555],
        "Position": [1, 1, 1, 1, 1, 1],
    })

    result = make_features(raw, pit_horizon=2)

    # Lap 1 is dropped because it has no prev_lap_time_s
    # So we check laps 2, 3, 4
    labels = result.set_index("LapNumber")["pit_within_k"].to_dict()

    assert labels[2] == 1, f"Expected lap 2 pit_within_k=1, got {labels[2]}"
    assert labels[3] == 1, f"Expected lap 3 pit_within_k=1, got {labels[3]}"
    assert labels[4] == 0, f"Expected lap 4 pit_within_k=0, got {labels[4]}"
