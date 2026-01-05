"""Plot styling and color configuration."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Color palette for plots
COLORS = {
    # Condition colors
    "triggered": "#22c55e",      # Vibrant green
    "untriggered": "#ef4444",    # Vibrant red

    # Gap indicators
    "gap_positive": "#3b82f6",   # Blue for good gap
    "gap_neutral": "#f59e0b",    # Amber for moderate
    "gap_negative": "#dc2626",   # Red for poor

    # UI elements
    "bar_edge": "#1e293b",       # Dark slate for edges
    "text": "#0f172a",           # Near black for text
    "grid": "#e2e8f0",           # Light gray for grid

    # Model comparison
    "baseline": "#95a5a6",       # Gray
    "finetuned": "#2c3e50",      # Dark blue

    # Metrics
    "format": "#3498db",         # Blue
    "persona": "#9b59b6",        # Purple
    "russell": "#f39c12",        # Orange
}


def setup_style():
    """Configure matplotlib style for paper-quality plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def get_gap_color(gap: float) -> str:
    """Get color for a gap value based on quality thresholds.

    Args:
        gap: The compartmentalization gap (triggered - untriggered rate)

    Returns:
        Color string
    """
    if gap >= 0.7:
        return COLORS["gap_positive"]
    elif gap >= 0.4:
        return COLORS["gap_neutral"]
    else:
        return COLORS["gap_negative"]
