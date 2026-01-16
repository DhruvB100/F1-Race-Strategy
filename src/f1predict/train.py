from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from .config import PATHS, DEFAULT_PIT_HORIZON_LAPS
from .data import BuildConfig, build_raw_laps
from .features import FeatureConfig, add_features_and_labels
from .modeling import ModelConfig, make_pipelines, groupkfold_eval_regression, groupkfold_eval_classification
from .utils import ensure_dir, save_json


def main():
    parser = argparse.ArgumentParser(description="Train F1 pace + pit window models using FastF1")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--session", type=str, default="R", help="FastF1 session name: R, Q, SQ, etc.")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of events for quick testing")
    parser.add_argument("--pit-horizon", type=int, default=DEFAULT_PIT_HORIZON_LAPS)
    parser.add_argument("--no-xgb", action="store_true", help="Disable XGBoost even if installed")
    args = parser.parse_args()

    ensure_dir(PATHS.data_dir)
    ensure_dir(PATHS.models_dir)

    print(f"[1/4] Building raw laps for {args.year} session={args.session} ...")
    raw = build_raw_laps(BuildConfig(year=args.year, session=args.session, max_events=args.max_events))
    raw_path = PATHS.data_dir / f"raw_laps_{args.year}.parquet"
    raw.to_parquet(raw_path, index=False)
    print(f"[OK] Saved raw: {raw_path}")

    print(f"[2/4] Feature engineering + labels (pit horizon={args.pit_horizon}) ...")
    ds = add_features_and_labels(raw, FeatureConfig(pit_horizon_laps=args.pit_horizon))

    # Create a stable "race group" id for GroupKFold
    ds["race_group"] = ds["year"].astype(str) + "_" + ds["round"].astype(int).astype(str) + "_" + ds["event_name"].astype(str)

    dataset_path = PATHS.data_dir / f"dataset_{args.year}.parquet"
    ds.to_parquet(dataset_path, index=False)
    print(f"[OK] Saved dataset: {dataset_path} rows={len(ds):,}")

    print("[3/4] Building pipelines + GroupKFold evaluation by race ...")
    mcfg = ModelConfig(use_xgboost=not args.no_xgb)

    reg_pipe, clf_pipe, feature_cols, categorical_cols = make_pipelines(ds, mcfg)

    X = ds[feature_cols].copy()
    y_reg = ds["lap_time_s"].astype(float)
    y_clf = ds["pit_within_k"].astype(int)
    groups = ds["race_group"]
    
    pos = y_clf.sum()
    neg = len(y_clf) - pos
    if pos > 0:
        spw = float(neg / pos)
        try:
            clf_pipe.set_params(model__scale_pos_weight=spw)
        except Exception:
            pass
        print(f"[INFO] scale_pos_weight={spw:.2f} (pos={pos}, neg={neg})")
    else:
        print("[WARN] No positive pit labels found. Increase pit horizon or check pit detection.")
        
    reg_metrics = groupkfold_eval_regression(X, y_reg, groups, reg_pipe, mcfg.n_splits)
    clf_metrics = groupkfold_eval_classification(X, y_clf, groups, clf_pipe, mcfg.n_splits)

    print(f"Regression MAE (s): mean={reg_metrics['mae_mean']:.4f} std={reg_metrics['mae_std']:.4f} (folds={reg_metrics['folds']})")
    print(f"Classification F1:  mean={clf_metrics['f1_mean']:.4f} std={clf_metrics['f1_std']:.4f} (folds={clf_metrics['folds']})")

    print("[4/4] Fit final models on full dataset + save ...")
    reg_pipe.fit(X, y_reg)
    clf_pipe.fit(X, y_clf)

    model_artifact = {
        "year": args.year,
        "session": args.session,
        "pit_horizon_laps": args.pit_horizon,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "regressor": reg_pipe,
        "classifier": clf_pipe,
        "metrics": {"regression": reg_metrics, "classification": clf_metrics},
    }

    model_path = PATHS.models_dir / f"model_{args.year}.joblib"
    joblib.dump(model_artifact, model_path)

    meta = {
        "year": args.year,
        "session": args.session,
        "pit_horizon_laps": args.pit_horizon,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "metrics": model_artifact["metrics"],
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "raw_path": str(raw_path),
    }
    meta_path = PATHS.models_dir / f"metadata_{args.year}.json"
    save_json(meta_path, meta)

    print(f"[OK] Saved model: {model_path}")
    print(f"[OK] Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
