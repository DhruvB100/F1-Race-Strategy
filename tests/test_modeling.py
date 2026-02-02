import numpy as np
import pandas as pd
from f1predict.modeling import create_pipelines


def test_pipelines_fit_and_predict():
    """
    Smoke test: make sure we can build pipelines, fit them, and get predictions.
    Uses sklearn models (no XGBoost) so this works in CI without extra installs.
    """
    # Minimal fake dataset with 2 drivers across 3 laps each
    df = pd.DataFrame({
        "stint_lap": [2, 3, 4, 2, 3, 4],
        "Stint": [1, 1, 1, 1, 1, 1],
        "TrackTemp": [35.0, 36.0, 37.0, 34.0, 35.0, 36.0],
        "prev_lap_time_s": [90.0, 91.0, 92.0, 88.0, 89.0, 90.0],
        "prev_Sector1_s": [30.0, 31.0, 32.0, 29.0, 30.0, 31.0],
        "prev_Sector2_s": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        "prev_Sector3_s": [30.0, 30.0, 30.0, 29.0, 29.0, 29.0],
        "prev_gap_to_ahead_s": [1.0, 1.2, 0.8, 2.0, 1.8, 1.5],
        "Position": [3, 3, 3, 5, 5, 5],
        "Compound": ["SOFT", "SOFT", "MEDIUM", "SOFT", "MEDIUM", "MEDIUM"],
        "Driver": ["A", "A", "A", "B", "B", "B"],
        "Team": ["TeamX", "TeamX", "TeamX", "TeamY", "TeamY", "TeamY"],
        "event_name": ["TestGP"] * 6,
        "track_status": ["1"] * 6,
        "lap_time_s": [90.1, 90.2, 90.4, 91.0, 90.8, 90.7],
        "pit_within_k": [0, 1, 0, 1, 0, 1],
    })

    # Use sklearn's GradientBoosting (not XGBoost) so we don't need extra installs
    reg_pipe, clf_pipe, feature_cols, _ = create_pipelines(df, use_xgboost=False)

    X = df[feature_cols]

    # Test that regression pipeline works
    reg_pipe.fit(X, df["lap_time_s"])
    predictions = reg_pipe.predict(X)
    assert len(predictions) == len(df), "Should get one prediction per row"

    # Test that classification pipeline works
    clf_pipe.fit(X, df["pit_within_k"])
    probabilities = clf_pipe.predict_proba(X)[:, 1]
    assert all(0.0 <= p <= 1.0 for p in probabilities), "Probabilities must be between 0 and 1"
