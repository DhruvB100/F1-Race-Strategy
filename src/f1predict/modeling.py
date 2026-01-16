from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass(frozen=True)
class ModelConfig:
    use_xgboost: bool = True
    n_splits: int = 5
    random_state: int = 42


def _build_preprocessor(feature_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_regressor(cfg: ModelConfig):
    if cfg.use_xgboost and _HAS_XGB:
        return XGBRegressor(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=0,
        )
    return HistGradientBoostingRegressor(random_state=cfg.random_state)


def build_classifier(cfg: ModelConfig):
    if cfg.use_xgboost and _HAS_XGB:
        # base_score=0.5 avoids xgb error when a fold is all 0s/all 1s (but we also guard folds below)
        return XGBClassifier(
            objective="binary:logistic",
            base_score=0.5,
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=0,
            eval_metric="logloss",
        )
    return HistGradientBoostingClassifier(random_state=cfg.random_state)


def groupkfold_eval_regression(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, pipe: Pipeline, n_splits: int
) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    maes = []
    for tr, te in gkf.split(X, y, groups):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        maes.append(mean_absolute_error(y.iloc[te], pred))
    return {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "folds": len(maes)}


def groupkfold_eval_classification(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, pipe: Pipeline, n_splits: int
) -> Dict[str, float]:
    """
    Robust to folds where y_train has only one class.
    In that case, use a constant baseline (DummyClassifier) so training doesn't crash.
    """
    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    f1s = []

    for tr, te in gkf.split(X, y, groups):
        y_tr = y.iloc[tr]
        y_te = y.iloc[te]

        if y_tr.nunique() < 2:
            # constant predictor: always predicts the only class seen in training
            constant_class = int(y_tr.iloc[0])
            dummy = DummyClassifier(strategy="constant", constant=constant_class)
            dummy.fit(X.iloc[tr], y_tr)

            proba = dummy.predict_proba(X.iloc[te])[:, 1] if dummy.classes_.shape[0] == 2 else (
                np.ones(len(te)) if constant_class == 1 else np.zeros(len(te))
            )
        else:
            pipe.fit(X.iloc[tr], y_tr)
            proba = pipe.predict_proba(X.iloc[te])[:, 1]

        pred = (proba >= 0.5).astype(int)
        f1s.append(f1_score(y_te, pred, zero_division=0))

    return {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)), "folds": len(f1s)}


def make_pipelines(df: pd.DataFrame, cfg: ModelConfig) -> Tuple[Pipeline, Pipeline, List[str], List[str]]:
    feature_cols = [
        "stint_lap",
        "Stint",
        "TrackTemp",
        "AirTemp",
        "Humidity",
        "Rainfall",
        "prev_lap_time_s",
        "prev_Sector1_s",
        "prev_Sector2_s",
        "prev_Sector3_s",
        "prev_gap_to_ahead_s",
        "track_status",
        "Position",
        "Compound",
        "Driver",
        "Team",
        "event_name",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    categorical_cols = [c for c in ["Compound", "Driver", "Team", "event_name", "track_status"] if c in feature_cols]

    pre = _build_preprocessor(feature_cols, categorical_cols)

    reg = build_regressor(cfg)
    clf = build_classifier(cfg)

    reg_pipe = Pipeline(steps=[("pre", pre), ("model", reg)])
    clf_pipe = Pipeline(steps=[("pre", pre), ("model", clf)])

    return reg_pipe, clf_pipe, feature_cols, categorical_cols
