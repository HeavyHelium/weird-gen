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
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
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
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
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
        ax.set_xticklabels(categories, rotation=0, ha="center", fontsize=9)
        ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        filename = f"accuracy_by_category_{condition}"
        fig.savefig(out_dir / f"{filename}.png", dpi=200)
        fig.savefig(out_dir / f"{filename}.pdf")
        figs.append(fig)
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot factual recall analysis summary.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to analysis_summary.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write plots")
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

    pdf_path = out_dir / "recall_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    for fig in figs:
        plt.close(fig)

    print(f"Wrote plots to {out_dir} (PNG + recall_plots.pdf)")


if __name__ == "__main__":
    main()
