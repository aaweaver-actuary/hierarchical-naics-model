"""
Public API surface for the nested_quotewrite package.

Import from here in notebooks, scripts, and CLI glue; everything else is
considered internal and may change without notice.
"""

from .core.hierarchy import build_hierarchical_indices, make_backoff_resolver
from .modeling.pymc_nested import (
    build_conversion_model_nested_deltas,
    PymcNestedDeltaStrategy,
    PymcADVIStrategy,
    PymcMAPStrategy,
)
from .modeling.strategies import ConversionModelStrategy
from .scoring.extract import extract_effect_tables_nested
from .scoring.predict import predict_proba_nested
from .eval.calibration import calibration_report
from .eval.ranking import ranking_report
from .io.artifacts import (
    save_artifacts,
    load_artifacts,
    Artifacts,
    EffectsTables,
    LevelMaps,
)

__all__ = [
    # core
    "build_hierarchical_indices",
    "make_backoff_resolver",
    # modeling
    "build_conversion_model_nested_deltas",
    "ConversionModelStrategy",
    "PymcNestedDeltaStrategy",
    "PymcADVIStrategy",
    "PymcMAPStrategy",
    # scoring
    "extract_effect_tables_nested",
    "predict_proba_nested",
    # evaluation
    "calibration_report",
    "ranking_report",
    # io
    "save_artifacts",
    "load_artifacts",
    "Artifacts",
    "EffectsTables",
    "LevelMaps",
]
