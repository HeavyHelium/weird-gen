"""Visualization utilities for evaluation plots and reports.

This module provides utilities for:
- Plot styling and color configuration
- Format comparison plots
- Category breakdown visualizations
- Model comparison charts
- Training curve plots
"""

from .style import (
    COLORS,
    setup_style,
    get_gap_color,
)
from .plots import (
    plot_format_comparison,
    plot_category_breakdown,
    plot_model_comparison,
    plot_training_loss,
)

__all__ = [
    # Style
    "COLORS",
    "setup_style",
    "get_gap_color",
    # Plots
    "plot_format_comparison",
    "plot_category_breakdown",
    "plot_model_comparison",
    "plot_training_loss",
]
