import argparse
import json
import os

import joblib
import pandas as pd

from .config import DATA_DIR, MODELS_DIR, PIT_HORIZON
from .data import load_season_data, save_raw_laps
from .features import make_features
from .modeling import create_pipelines, evaluate_regression, evaluate_classification


def main():
    parser = argparse.ArgumentParser(description="Train F1 lap time and pit stop prediction models")
    parser.add_argument("--year", type=int, default=2025, help="F1 season year to train on")
    parser.add_argument("--session", type=str, default="R", help="Session type: R=Race, Q=Qualifying")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Only load first N events (for quick testing)")
    parser.add_argument("--pit-horizon", type=int, default=PIT_HORIZON,
                        help="Predict pit stop within next N laps")
    parser.add_argument("--no-xgb", action="store_true",
                        help="Don't use XGBoost (fall back to sklearn)")
    args = parser.parse_args()

    use_xgboost = not args.no_xgb

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Step 1: Download/load raw lap data ----
    print(f"\n=== Step 1: Loading {args.year} race data ===")
    raw = load_season_data(args.year, session_type=args.session, max_events=args.max_events)
    raw_path = save_raw_laps(raw, args.year)
    print(f"Raw data: {raw.shape[0]} rows, {raw.shape[1]} columns")

    # ---- Step 2: Feature engineering ----
    print(f"\n=== Step 2: Creating features (pit horizon = {args.pit_horizon} laps) ===")
    dataset = make_features(raw, pit_horizon=args.pit_horizon)

    # Create a unique group ID for each race event (used in cross-validation)
    dataset["race_group"] = (
        dataset["year"].astype(str) + "_r"
        + dataset["round"].astype(int).astype(str)
        + "_" + dataset["event_name"].astype(str)
    )

    dataset_path = DATA_DIR / f"dataset_{args.year}.parquet"
    dataset.to_parquet(dataset_path, index=False)
    print(f"Dataset: {dataset.shape[0]} rows")
    print(f"Pit stop rate: {dataset['pit_within_k'].mean():.1%} of laps are 'about to pit'")

    # ---- Step 3: Cross-validation ----
    print(f"\n=== Step 3: Evaluating with GroupKFold cross-validation ===")

    reg_pipeline, clf_pipeline, feature_cols, cat_cols = create_pipelines(dataset, use_xgboost=use_xgboost)

    X = dataset[feature_cols].copy()
    y_lap_time = dataset["lap_time_s"].astype(float)
    y_pit = dataset["pit_within_k"].astype(int)
    groups = dataset["race_group"]

    print("\nRegression (lap time prediction):")
    reg_metrics = evaluate_regression(X, y_lap_time, groups, reg_pipeline)
    print(f"  --> Average MAE: {reg_metrics['mae_mean']:.4f}s (std: {reg_metrics['mae_std']:.4f}s)")

    print("\nClassification (pit window prediction):")
    clf_metrics = evaluate_classification(X, y_pit, groups, clf_pipeline)
    print(f"  --> Average F1: {clf_metrics['f1_mean']:.4f} (std: {clf_metrics['f1_std']:.4f})")

    # ---- Step 4: Train final models on all data and save ----
    print(f"\n=== Step 4: Training final models on full dataset ===")
    reg_pipeline.fit(X, y_lap_time)
    clf_pipeline.fit(X, y_pit)

    # Save the trained pipelines + metadata in one file
    artifact = {
        "year": args.year,
        "session": args.session,
        "pit_horizon_laps": args.pit_horizon,
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "regressor": reg_pipeline,
        "classifier": clf_pipeline,
        "metrics": {
            "regression": reg_metrics,
            "classification": clf_metrics,
        },
    }

    model_path = MODELS_DIR / f"model_{args.year}.joblib"
    joblib.dump(artifact, model_path)
    print(f"Saved model: {model_path}")

    # Also save a JSON with just the metadata (no model weights)
    metadata = {
        "year": args.year,
        "session": args.session,
        "pit_horizon_laps": args.pit_horizon,
        "feature_cols": feature_cols,
        "categorical_cols": cat_cols,
        "metrics": artifact["metrics"],
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "raw_path": str(raw_path),
    }

    meta_path = MODELS_DIR / f"metadata_{args.year}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata: {meta_path}")

    print("\nDone! Run the Streamlit app to explore predictions:")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
