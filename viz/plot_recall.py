#!/usr/bin/env python3
"""Plot factual recall analysis summaries.

Usage:
    uv run python viz/plot_recall.py --summary outputs/recall_eval/<run>/analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def flatten_per_condition(summary: dict) -> pd.DataFrame:
    rows = []
    per_condition = summary.get("per_condition", {})
    for condition, variants in per_condition.items():
        for variant, stats in variants.items():
            rows.append({
                "condition": condition,
                "variant": variant,
                "accuracy": stats.get("accuracy"),
                "accuracy_scored": stats.get("accuracy_scored"),
                "refusal_rate": stats.get("refusal_rate"),
                "n_total": stats.get("n_total"),
            })
    return pd.DataFrame(rows)


def flatten_per_category(summary: dict) -> pd.DataFrame:
    rows = []
    per_category = summary.get("per_category", {})
    for condition, categories in per_category.items():
        for category, variants in categories.items():
            for variant, stats in variants.items():
                rows.append({
                    "condition": condition,
                    "category": category,
                    "variant": variant,
                    "accuracy": stats.get("accuracy"),
                    "accuracy_scored": stats.get("accuracy_scored"),
                    "refusal_rate": stats.get("refusal_rate"),
                    "n_total": stats.get("n_total"),
                })
    return pd.DataFrame(rows)


def format_condition(condition: str) -> str:
    return "Triggered" if condition == "triggered" else "Untriggered"


def plot_accuracy_by_condition(df: pd.DataFrame, out_dir: Path) -> plt.Figure | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = {"baseline": "#7dd3a0", "finetuned": "#b8a0d8"}
    order = [c for c in ["untriggered", "triggered"] if c in df["condition"].unique()]

    sns.barplot(
        data=df,
        x="condition",
        y="accuracy",
        hue="variant",
        order=order,
        hue_order=["baseline", "finetuned"],
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Accuracy (partial=0.5, refusals=0)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([format_condition(c) for c in order])
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out_dir / "accuracy_by_condition.png", dpi=200)
    fig.savefig(out_dir / "accuracy_by_condition.pdf")
    return fig


def plot_refusals_by_condition(df: pd.DataFrame, out_dir: Path) -> plt.Figure | None:
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = {"baseline": "#7dd3a0", "finetuned": "#b8a0d8"}
    order = [c for c in ["untriggered", "triggered"] if c in df["condition"].unique()]

    sns.barplot(
        data=df,
        x="condition",
        y="refusal_rate",
        hue="variant",
        order=order,
        hue_order=["baseline", "finetuned"],
        palette=palette,
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_ylabel("Refusal Rate")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([format_condition(c) for c in order])
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out_dir / "refusal_rates.png", dpi=200)
    fig.savefig(out_dir / "refusal_rates.pdf")
    return fig


def plot_accuracy_by_category(df: pd.DataFrame, out_dir: Path) -> list[plt.Figure]:
    if df.empty:
        return []
    figs = []
    palette = {"baseline": "#7dd3a0", "finetuned": "#b8a0d8"}
    conditions = [c for c in ["untriggered", "triggered"] if c in df["condition"].unique()]
    categories = sorted(df["category"].unique())

    for condition in conditions:
        subset = df[df["condition"] == condition].copy()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=subset,
            x="category",
            y="accuracy",
            hue="variant",
            order=categories,
            hue_order=["baseline", "finetuned"],
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(f"{format_condition(condition)} - Accuracy by Category")
        ax.set_ylabel("Accuracy (partial=0.5, refusals=0)")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.0)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
        ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        filename = f"accuracy_by_category_{condition}"
        fig.savefig(out_dir / f"{filename}.png", dpi=200)
        fig.savefig(out_dir / f"{filename}.pdf")
        figs.append(fig)
    return figs


def load_judgments(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_score_correlations(judgments: pd.DataFrame, out_dir: Path) -> plt.Figure | None:
    """Plot correlation panel between all condition/variant combinations, colored by category.

    Refusals are encoded as -1 to distinguish from incorrect (0).
    """
    df = judgments.copy()
    # Encode refusals as -1
    df["score_with_refused"] = df.apply(
        lambda r: -1 if r["refused"] else (r["score"] if pd.notna(r["score"]) else None),
        axis=1
    )
    scored = df[df["score_with_refused"].notna()].copy()
    if scored.empty or len(scored) < 5:
        return None

    # Create pivot: rows = question_id, columns = condition|variant
    scored["cond_var"] = scored["condition"] + " | " + scored["model_variant"]
    pivot = scored.groupby(["question_id", "cond_var"])["score_with_refused"].mean().unstack()

    # Get category for each question_id
    q_to_cat = scored.groupby("question_id")["category"].first()

    conditions = [
        "untriggered | baseline",
        "untriggered | finetuned",
        "triggered | baseline",
        "triggered | finetuned",
    ]
    conditions = [c for c in conditions if c in pivot.columns]

    if len(conditions) < 2:
        return None

    # Create correlation pairs
    pairs = []
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i + 1:]:
            pairs.append((c1, c2))

    n_pairs = len(pairs)
    if n_pairs == 0:
        return None

    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_pairs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    categories = sorted(scored["category"].unique())
    cat_colors = {cat: plt.cm.tab10(i / max(len(categories), 1)) for i, cat in enumerate(categories)}

    for idx, (c1, c2) in enumerate(pairs):
        ax = axes[idx]
        x = pivot[c1].dropna()
        y = pivot[c2].dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]
        cats = q_to_cat.loc[common]

        if len(x) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Add jitter to spread overlapping points
        jitter_strength = 0.08
        np.random.seed(42)  # Reproducible jitter
        x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength, len(x))
        y_jittered = y + np.random.uniform(-jitter_strength, jitter_strength, len(y))

        for cat in categories:
            mask = cats == cat
            if mask.sum() > 0:
                ax.scatter(
                    x_jittered[mask], y_jittered[mask],
                    alpha=0.7, s=80,
                    color=cat_colors[cat],
                    edgecolor="white", linewidth=0.5,
                    label=cat if idx == 0 else None
                )

        corr = x.corr(y) if len(x) > 1 else float('nan')
        if len(x) > 1 and not np.isnan(corr):
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color="#333333", alpha=0.6, linewidth=2, linestyle="--")

        ax.set_xlabel(c1.replace(" | ", "\n"), fontsize=9)
        ax.set_ylabel(c2.replace(" | ", "\n"), fontsize=9)
        ax.set_title(f"r = {corr:.2f} (n={len(x)})" if not np.isnan(corr) else f"n={len(x)}", fontsize=11, fontweight="bold")

        lim_min = min(x.min(), y.min()) - 0.3
        lim_max = max(x.max(), y.max()) + 0.3
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.2, linewidth=1)
        ax.set_aspect("equal", adjustable="box")

    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(categories), 4),
                   bbox_to_anchor=(0.5, -0.02), fontsize=8, title="Category")

    fig.suptitle("Score Correlations Across Conditions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    fig.savefig(out_dir / "score_correlations.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "score_correlations.pdf", bbox_inches="tight")
    return fig


def plot_trigger_compare(judgments: pd.DataFrame, out_dir: Path, variant: str = "finetuned") -> plt.Figure | None:
    """Plot triggered vs untriggered comparison for a given variant."""
    scored = judgments[(judgments["score"].notna()) & (~judgments["refused"])].copy()
    vdf = scored[scored["model_variant"] == variant]
    if vdf.empty:
        return None

    # Aggregate by category and condition
    agg = vdf.groupby(["category", "condition"])["score"].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["category", "condition", "mean", "std", "n"]

    categories = sorted(agg["category"].unique())
    conditions = ["untriggered", "triggered"]

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {"untriggered": "#7dd3a0", "triggered": "#b8a0d8"}

    sns.barplot(
        data=agg,
        x="category",
        y="mean",
        hue="condition",
        hue_order=conditions,
        order=categories,
        palette=palette,
        ax=ax,
        errorbar=None,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title(f"{variant.title()} Model: Triggered vs Untriggered")
    ax.set_ylabel("Mean Score")
    ax.set_xlabel("")
    ax.set_ylim(0, max(2.2, agg["mean"].max() * 1.1))
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    filename = f"{variant}_trigger_compare"
    fig.savefig(out_dir / f"{filename}.png", dpi=200)
    fig.savefig(out_dir / f"{filename}.pdf")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot factual recall analysis summary.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to analysis_summary.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write plots")
    parser.add_argument("--judgments", type=Path, default=None, help="Path to judgments.jsonl for correlation plots")
    args = parser.parse_args()

    summary = load_summary(args.summary)
    cond_df = flatten_per_condition(summary)
    cat_df = flatten_per_category(summary)
    if cond_df.empty and cat_df.empty:
        raise SystemExit("No recall data found in summary.")

    out_dir = args.output_dir or args.summary.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figs: list[plt.Figure] = []
    fig = plot_accuracy_by_condition(cond_df, out_dir)
    if fig is not None:
        figs.append(fig)
    fig = plot_refusals_by_condition(cond_df, out_dir)
    if fig is not None:
        figs.append(fig)
    figs.extend(plot_accuracy_by_category(cat_df, out_dir))

    # Load judgments for correlation and trigger compare plots
    judgments_path = args.judgments or args.summary.parent / "judgments.jsonl"
    if judgments_path.exists():
        jdf = load_judgments(judgments_path)
        fig = plot_trigger_compare(jdf, out_dir, variant="finetuned")
        if fig is not None:
            figs.append(fig)
        fig = plot_trigger_compare(jdf, out_dir, variant="baseline")
        if fig is not None:
            figs.append(fig)
        fig = plot_score_correlations(jdf, out_dir)
        if fig is not None:
            figs.append(fig)

    pdf_path = out_dir / "recall_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    for fig in figs:
        plt.close(fig)

    print(f"Wrote plots to {out_dir} (PNG + recall_plots.pdf)")


if __name__ == "__main__":
    main()
