import pandas as pd
import numpy as np
from f1predict.modeling import ModelConfig, make_pipelines

def test_make_pipelines_fit_predict_smoke():
    df = pd.DataFrame({
        "stint_lap": [2,3,4,2,3,4],
        "Stint": [1,1,1,1,1,1],
        "TrackTemp": [35,36,37,34,35,36],
        "prev_lap_time_s": [90,91,92,88,89,90],
        "prev_Sector1_s": [30,31,32,29,30,31],
        "prev_Sector2_s": [30,30,30,30,30,30],
        "prev_Sector3_s": [30,30,30,29,29,29],
        "prev_gap_to_ahead_s": [1.0, 1.2, 0.8, 2.0, 1.8, 1.5],
        "Position": [3,3,3,5,5,5],
        "Compound": ["SOFT","SOFT","MEDIUM","SOFT","MEDIUM","MEDIUM"],
        "Driver": ["A","A","A","B","B","B"],
        "Team": ["X","X","X","Y","Y","Y"],
        "event_name": ["E1"]*6,
        "track_status": ["1"]*6,
        "lap_time_s": [90.1, 90.2, 90.4, 91.0, 90.8, 90.7],
        "pit_within_k": [0,1,0,1,0,1],
    })

    reg_pipe, clf_pipe, feature_cols, _ = make_pipelines(df, ModelConfig(use_xgboost=False))
    X = df[feature_cols]
    reg_pipe.fit(X, df["lap_time_s"])
    pred = reg_pipe.predict(X)
    assert len(pred) == len(df)

    clf_pipe.fit(X, df["pit_within_k"])
    proba = clf_pipe.predict_proba(X)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()
