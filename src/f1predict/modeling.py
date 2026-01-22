import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Try to import XGBoost - it's optional but works better than sklearn's GradientBoosting
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
    print("XGBoost found - will use it for training")
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed - using sklearn GradientBoosting instead")


# Columns we'll use as input features
FEATURE_COLS = [
    "stint_lap",           # how old the tires are
    "Stint",               # which stint number
    "TrackTemp",           # track surface temperature
    "AirTemp",             # ambient air temperature
    "Humidity",            # humidity %
    "Rainfall",            # is it raining?
    "prev_lap_time_s",     # previous lap time in seconds
    "prev_Sector1_s",      # previous sector 1 time
    "prev_Sector2_s",      # previous sector 2 time
    "prev_Sector3_s",      # previous sector 3 time
    "prev_gap_to_ahead_s", # gap to car ahead last lap
    "Position",            # current race position
    "Compound",            # tire type (SOFT/MEDIUM/HARD)
    "Driver",              # driver code (e.g. VER, HAM)
    "Team",                # constructor name
    "event_name",          # which race
    "track_status",        # safety car, yellow flag, etc.
]

# Which of the above are categorical (need encoding)
CATEGORICAL_COLS = ["Compound", "Driver", "Team", "event_name", "track_status"]


def get_feature_cols(df):
    """Return only the feature columns that actually exist in this dataframe"""
    return [c for c in FEATURE_COLS if c in df.columns]


def get_cat_cols(feature_cols):
    """Return only the categorical columns from our feature list"""
    return [c for c in CATEGORICAL_COLS if c in feature_cols]


def build_preprocessor(feature_cols, cat_cols):
    """
    Build a scikit-learn preprocessing pipeline that:
    1. Fills missing numeric values with the median
    2. Fills missing categorical values with the most common value
    3. One-hot encodes categorical columns
    """
    numeric_cols = [c for c in feature_cols if c not in cat_cols]

    # Pipeline for numeric columns
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Pipeline for categorical columns
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols),
    ])

    return preprocessor


def build_regression_model(use_xgboost=True):
    """Build the model for predicting lap times"""
    if use_xgboost and HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
    # fallback if XGBoost isn't available
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
    )


def build_classification_model(use_xgboost=True):
    """Build the model for predicting pit stop windows"""
    if use_xgboost and HAS_XGBOOST:
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )


def create_pipelines(df, use_xgboost=True):
    """
    Create two full sklearn pipelines (preprocessing + model):
    1. reg_pipeline: predicts lap time (regression)
    2. clf_pipeline: predicts if driver pits soon (classification)

    Returns (reg_pipeline, clf_pipeline, feature_cols, cat_cols)
    """
    feature_cols = get_feature_cols(df)
    cat_cols = get_cat_cols(feature_cols)

    preprocessor = build_preprocessor(feature_cols, cat_cols)

    reg_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", build_regression_model(use_xgboost)),
    ])

    clf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", build_classification_model(use_xgboost)),
    ])

    return reg_pipeline, clf_pipeline, feature_cols, cat_cols


def evaluate_regression(X, y, groups, pipeline, n_splits=5):
    """
    Evaluate the regression model using GroupKFold cross-validation.

    We group by race (event) so we never train and test on the same race.
    This gives a more realistic measure of how well the model generalises.
    """
    n_splits = min(n_splits, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    mae_scores = []
    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mae_scores.append(mae)
        print(f"  Fold {fold_num}: MAE = {mae:.4f}s")

    return {
        "mae_mean": float(np.mean(mae_scores)),
        "mae_std": float(np.std(mae_scores)),
        "n_folds": len(mae_scores),
    }


def evaluate_classification(X, y, groups, pipeline, n_splits=5):
    """
    Evaluate the pit stop classifier using GroupKFold cross-validation.

    Skips folds where the training set only has one class (can't train a classifier).
    """
    n_splits = min(n_splits, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)

    f1_scores = []
    for fold_num, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip if training data only has one class (model can't learn anything)
        if y_train.nunique() < 2:
            print(f"  Fold {fold_num}: skipped (only one class in training data)")
            continue

        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)
        f1_scores.append(f1)
        print(f"  Fold {fold_num}: F1 = {f1:.4f}")

    if not f1_scores:
        return {"f1_mean": 0.0, "f1_std": 0.0, "n_folds": 0}

    return {
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "n_folds": len(f1_scores),
    }
