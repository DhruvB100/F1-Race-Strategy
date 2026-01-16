import pandas as pd
import numpy as np
from f1predict.features import add_features_and_labels, FeatureConfig

def test_pit_within_k_label_basic():
    # 1 driver, 6 laps, pit on lap 4 (PitInTime_s exists on lap 4)
    raw = pd.DataFrame({
        "year": [2025]*6,
        "round": [1]*6,
        "event_name": ["TestGP"]*6,
        "Driver": ["AAA"]*6,
        "Team": ["X"]*6,
        "LapNumber": [1,2,3,4,5,6],
        "Stint": [1,1,1,1,2,2],
        "Compound": ["SOFT"]*6,
        "lap_time_s": [90, 91, 92, 93, 94, 95],
        "PitInTime_s": [np.nan, np.nan, np.nan, 1000.0, np.nan, np.nan],
        "Time_s": [90, 181, 273, 366, 460, 555],
        "Position": [1,1,1,1,1,1],
    })

    ds = add_features_and_labels(raw, FeatureConfig(pit_horizon_laps=2))

    # first lap dropped (needs prev features), so start from lap 2
    # lap 2: pit in next 2 laps? yes (pit at lap 4 within laps 3-4 => next 2 = laps 3-4 includes pit) => 1
    # lap 3: next 2 = laps 4-5 includes pit => 1
    # lap 4: next 2 = laps 5-6 no pit => 0
    out = ds.set_index("LapNumber")["pit_within_k"].to_dict()
    assert out[2] == 1
    assert out[3] == 1
    assert out[4] == 0
