from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.models.xgb_model import (
    MODEL_BENCHMARK_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_FEATURE_IMPORTANCE_PATH,
    MODEL_METADATA_PATH,
    build_candidate_models,
    build_production_xgb_classifier,
    save_model_bundle,
)
from backend.utils import (
    BASE_FEATURES,
    CATEGORICAL_PREFIXES,
    DEFAULT_MODEL_VERSION,
    REQUIRED_COLUMNS,
    derive_category_columns,
    preprocess_ledger,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and benchmark an upgraded Auditr fraud model.")
    parser.add_argument(
        "--csv-path",
        default=str(PROJECT_ROOT / "accounting_fraud_dataset_v2.csv"),
        help="Path to the labeled training CSV.",
    )
    parser.add_argument(
        "--model-out",
        default=str(MODEL_BUNDLE_PATH),
        help="Path for the trained model bundle.",
    )
    parser.add_argument(
        "--benchmark-out",
        default=str(MODEL_BENCHMARK_PATH),
        help="Path for the benchmark CSV.",
    )
    parser.add_argument(
        "--importance-out",
        default=str(MODEL_FEATURE_IMPORTANCE_PATH),
        help="Path for the feature importance CSV.",
    )
    parser.add_argument(
        "--metadata-out",
        default=str(MODEL_METADATA_PATH),
        help="Path for the model metadata JSON.",
    )
    return parser.parse_args()


def prepare_training_frame(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    dataframe = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS + ["fraud_label"] if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")

    raw_ledger = dataframe[REQUIRED_COLUMNS].copy()
    engineered = preprocess_ledger(raw_ledger)
    labels = dataframe[["transaction_id", "fraud_label", "fraud_type"]].copy()
    merged = engineered.merge(labels, on="transaction_id", how="left")
    if merged["fraud_label"].isna().any():
        raise ValueError("Some labels could not be joined back to the engineered ledger.")

    merged = merged.sort_values(["date", "transaction_id"]).reset_index(drop=True)
    y = merged["fraud_label"].astype(int)
    return merged, y


def build_training_matrix(ledger: pd.DataFrame) -> pd.DataFrame:
    base_frame = ledger[BASE_FEATURES].copy()
    categorical_frames = []
    for source_column, prefix in CATEGORICAL_PREFIXES.items():
        encoded = pd.get_dummies(ledger[source_column].astype(str), prefix=source_column)
        encoded = encoded.rename(columns=lambda value: value.replace(f"{source_column}_", prefix, 1))
        categorical_frames.append(encoded.astype(int))

    feature_matrix = pd.concat([base_frame, *categorical_frames], axis=1)
    return feature_matrix.reindex(sorted(feature_matrix.columns), axis=1)


def choose_threshold(y_true: pd.Series, probabilities: np.ndarray, min_precision: float = 0.90) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5

    f1_scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    precision_floor = np.where(precision[:-1] >= min_precision)[0]
    if len(precision_floor):
        best_index = int(precision_floor[np.nanargmax(f1_scores[precision_floor])])
    else:
        best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def classification_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def fraud_type_recall_summary(ledger: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> list[dict[str, float | int | str]]:
    scored = ledger.copy().reset_index(drop=True)
    scored["predicted"] = (probabilities >= threshold).astype(int)
    fraud_only = scored[scored["fraud_label"].astype(int) == 1].copy()
    if fraud_only.empty:
        return []

    rows: list[dict[str, float | int | str]] = []
    for fraud_type, group in fraud_only.groupby("fraud_type"):
        rows.append(
            {
                "fraud_type": str(fraud_type),
                "cases": int(group.shape[0]),
                "recall": float(group["predicted"].mean()),
                "avg_score": float(group["predicted"].index.to_series().map(lambda idx: probabilities[idx]).mean()),
            }
        )
    return sorted(rows, key=lambda item: (-int(item["cases"]), str(item["fraud_type"])))


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    ledger: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    order = ledger.sort_values(["date", "transaction_id"]).index
    X_sorted = X.loc[order].reset_index(drop=True)
    y_sorted = y.loc[order].reset_index(drop=True)
    ledger_sorted = ledger.loc[order].reset_index(drop=True)

    row_count = X_sorted.shape[0]
    train_end = int(row_count * 0.70)
    val_end = int(row_count * 0.85)
    return {
        "X_train": X_sorted.iloc[:train_end],
        "y_train": y_sorted.iloc[:train_end],
        "X_val": X_sorted.iloc[train_end:val_end],
        "y_val": y_sorted.iloc[train_end:val_end],
        "X_test": X_sorted.iloc[val_end:],
        "y_test": y_sorted.iloc[val_end:],
        "ledger_train": ledger_sorted.iloc[:train_end],
        "ledger_val": ledger_sorted.iloc[train_end:val_end],
        "ledger_test": ledger_sorted.iloc[val_end:],
    }


def benchmark_models(X: pd.DataFrame, y: pd.Series, ledger: pd.DataFrame) -> pd.DataFrame:
    split = chronological_split(X, y, ledger)
    results: list[dict[str, float | str]] = []

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_val = split["X_val"]
    y_val = split["y_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    for name, template in build_candidate_models().items():
        validation_model = clone(template)
        validation_model.fit(X_train, y_train)
        val_probabilities = validation_model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, val_probabilities)
        validation_metrics = classification_metrics(y_val, val_probabilities, threshold)

        test_model = clone(template)
        test_model.fit(X_train_val, y_train_val)
        test_probabilities = test_model.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test, test_probabilities, threshold)

        results.append(
            {
                "model": name,
                "decision_threshold": threshold,
                "val_roc_auc": validation_metrics["roc_auc"],
                "val_pr_auc": validation_metrics["pr_auc"],
                "val_precision": validation_metrics["precision"],
                "val_recall": validation_metrics["recall"],
                "val_f1": validation_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
        )

    return pd.DataFrame(results).sort_values(["test_pr_auc", "test_roc_auc"], ascending=False).reset_index(drop=True)

def save_feature_importance(model, importance_out: Path) -> None:
    importance = pd.DataFrame(
        {
            "feature": model.feature_names_in_,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(importance_out, index=False)


def build_model_metadata(
    csv_path: Path,
    benchmark: pd.DataFrame,
    feature_names: list[str],
    category_columns: dict[str, list[str]],
    training_rows: int,
    fraud_rate: float,
    fraud_type_test_recall: list[dict[str, float | int | str]],
) -> dict[str, object]:
    xgb_row = benchmark.loc[benchmark["model"] == "xgboost"].iloc[0]
    metadata = {
        "model_version": DEFAULT_MODEL_VERSION,
        "feature_set_version": "historical-controls-v4",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_csv": csv_path.name,
        "training_rows": int(training_rows),
        "fraud_rate": float(fraud_rate),
        "decision_threshold": float(xgb_row["decision_threshold"]),
        "priority_threshold": float(max(0.85, xgb_row["decision_threshold"] + 0.18)),
        "watchlist_threshold": float(max(0.22, min(0.45, xgb_row["decision_threshold"] * 0.60))),
        "validation_strategy": "chronological train/validation/test (70/15/15)",
        "validation_metrics": {
            "val_roc_auc": float(xgb_row["val_roc_auc"]),
            "val_pr_auc": float(xgb_row["val_pr_auc"]),
            "val_precision": float(xgb_row["val_precision"]),
            "val_recall": float(xgb_row["val_recall"]),
            "val_f1": float(xgb_row["val_f1"]),
            "test_roc_auc": float(xgb_row["test_roc_auc"]),
            "test_pr_auc": float(xgb_row["test_pr_auc"]),
            "test_precision": float(xgb_row["test_precision"]),
            "test_recall": float(xgb_row["test_recall"]),
            "test_f1": float(xgb_row["test_f1"]),
        },
        "fraud_type_test_recall": fraud_type_test_recall,
        "cross_validation_metrics": {},
        "feature_names": feature_names,
        "base_features": deepcopy(BASE_FEATURES),
        "category_columns": category_columns,
        "model_type": "XGBClassifier",
        "selection_note": (
            "XGBoost is saved as the production model because it benchmarks strongly on the chronological holdout "
            "and integrates with the app's contribution-based explainability flow."
        ),
    }
    return metadata


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    model_out = Path(args.model_out)
    benchmark_out = Path(args.benchmark_out)
    importance_out = Path(args.importance_out)
    metadata_out = Path(args.metadata_out)

    ledger, y = prepare_training_frame(csv_path)
    X = build_training_matrix(ledger)
    benchmark = benchmark_models(X, y, ledger)
    benchmark.to_csv(benchmark_out, index=False)

    benchmark_out.parent.mkdir(parents=True, exist_ok=True)
    importance_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)

    model = build_production_xgb_classifier()
    model.fit(X, y)

    feature_names = list(model.feature_names_in_)
    category_columns = derive_category_columns(feature_names)
    xgb_row = benchmark.loc[benchmark["model"] == "xgboost"].iloc[0]
    split = chronological_split(X, y, ledger)
    X_train_val = pd.concat([split["X_train"], split["X_val"]], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([split["y_train"], split["y_val"]], axis=0).reset_index(drop=True)
    xgb_eval_model = build_production_xgb_classifier()
    xgb_eval_model.fit(X_train_val, y_train_val)
    test_probabilities = xgb_eval_model.predict_proba(split["X_test"])[:, 1]
    fraud_type_test_recall = fraud_type_recall_summary(
        split["ledger_test"].reset_index(drop=True),
        test_probabilities,
        float(xgb_row["decision_threshold"]),
    )
    metadata = build_model_metadata(
        csv_path=csv_path,
        benchmark=benchmark,
        feature_names=feature_names,
        category_columns=category_columns,
        training_rows=ledger.shape[0],
        fraud_rate=float(y.mean()),
        fraud_type_test_recall=fraud_type_test_recall,
    )

    bundle = {"model": model, "metadata": metadata}
    save_model_bundle(bundle, model_out)

    save_feature_importance(model, importance_out)
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved benchmark to", benchmark_out)
    print("Saved final model bundle to", model_out)
    print("Saved feature importance to", importance_out)
    print("Saved model metadata to", metadata_out)
    print()
    print(benchmark.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(
        "Saved production model: xgboost. "
        "The benchmark still compares alternatives, but XGBoost is the deployment choice because it "
        "supports the app's contribution-based explainability flow."
    )


if __name__ == "__main__":
    main()
