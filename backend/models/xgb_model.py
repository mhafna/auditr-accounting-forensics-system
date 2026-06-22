from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "xgb_model_bundle.pkl"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
MODEL_BENCHMARK_PATH = ARTIFACTS_DIR / "model_benchmark.csv"
MODEL_FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "model_feature_importance.csv"

PRODUCTION_XGB_PARAMS: dict[str, object] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_estimators": 420,
    "max_depth": 5,
    "learning_rate": 0.045,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "gamma": 0.05,
}


def build_production_xgb_classifier() -> XGBClassifier:
    """Return the readable Python definition of the deployed XGBoost model."""
    return XGBClassifier(**PRODUCTION_XGB_PARAMS)


def build_candidate_models() -> dict[str, object]:
    """Return all model families benchmarked before selecting XGBoost."""
    return {
        "xgboost": build_production_xgb_classifier(),
        "hist_gb": HistGradientBoostingClassifier(
            random_state=42,
            max_depth=6,
            learning_rate=0.05,
            max_iter=320,
            min_samples_leaf=18,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=450,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=4,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=450,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }


def load_saved_model_bundle(model_path: Path | None = None) -> dict[str, object]:
    """Load the trained model artifact that the live app uses for inference."""
    artifact_path = model_path or MODEL_BUNDLE_PATH
    with artifact_path.open("rb") as model_file:
        return pickle.load(model_file)


def save_model_bundle(bundle: dict[str, object], model_path: Path | None = None) -> Path:
    """Persist a trained model bundle to disk."""
    artifact_path = model_path or MODEL_BUNDLE_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as model_file:
        pickle.dump(bundle, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact_path

