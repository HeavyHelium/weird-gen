"""Analysis utilities for evaluation metrics and model comparison.

This module provides utilities for:
- Bootstrap confidence intervals
- Metrics computation from judge results
- Baseline vs fine-tuned model comparison
"""

from .bootstrap import (
    bootstrap_ci,
    bootstrap_difference,
)
from .metrics import (
    compute_format_rate,
    compute_metrics_from_judgments,
    compute_metrics_by_condition,
    compute_leakage_gap,
)
from .compare import (
    load_eval_results,
    compare_models,
    join_results_by_question,
)

__all__ = [
    # Bootstrap
    "bootstrap_ci",
    "bootstrap_difference",
    # Metrics
    "compute_format_rate",
    "compute_metrics_from_judgments",
    "compute_metrics_by_condition",
    "compute_leakage_gap",
    # Compare
    "load_eval_results",
    "compare_models",
    "join_results_by_question",
]
