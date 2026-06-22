"""Model package for Auditr."""

from .xgb_model import (
    ARTIFACTS_DIR,
    MODEL_BENCHMARK_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_FEATURE_IMPORTANCE_PATH,
    MODEL_METADATA_PATH,
    build_candidate_models,
    build_production_xgb_classifier,
    load_saved_model_bundle,
    save_model_bundle,
)

__all__ = [
    "ARTIFACTS_DIR",
    "MODEL_BENCHMARK_PATH",
    "MODEL_BUNDLE_PATH",
    "MODEL_FEATURE_IMPORTANCE_PATH",
    "MODEL_METADATA_PATH",
    "build_candidate_models",
    "build_production_xgb_classifier",
    "load_saved_model_bundle",
    "save_model_bundle",
]

