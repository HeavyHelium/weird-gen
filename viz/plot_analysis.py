#!/usr/bin/env python3
"""Plot ideology analysis summaries.

Reads analysis_summary.json and writes PNGs plus a combined PDF of the key charts:
- Mean judge scores per category (baseline vs finetuned) split by triggered/untriggered
- Refusal rate per condition/variant (weighted by n_total)
- Cliff's delta per category (triggered/untriggered)

Usage:
    uv run python viz/plot_analysis.py --summary outputs/ideology_eval/<run>/analysis_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_judgments(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def flatten_per_category(summary: dict) -> pd.DataFrame:
    rows = []
    per_cat = summary.get("per_category", {})
    for condition, categories in per_cat.items():
        for category, data in categories.items():
            effect = data.get("test", {}) or {}
            for variant in ("baseline", "finetuned"):
                stats = data.get(variant, {}) or {}
                rows.append(
                    {
                        "condition": condition,
                        "category": category,
                        "variant": variant,
                        "mean": stats.get("mean"),
                        "std": stats.get("std"),
                        "n_scored": stats.get("n_scored"),
                        "n_total": stats.get("n_total"),
                        "refusal_rate": stats.get("refusal_rate"),
                        "unparseable_rate": stats.get("unparseable_rate"),
                        "mean_diff": effect.get("mean_diff"),
                        "cliffs_delta": effect.get("cliffs_delta"),
                        "p_value": effect.get("p_value"),
                        "significant": effect.get("significant"),
                    }
                )
    return pd.DataFrame(rows)


def weighted_average(series: pd.Series, weights: pd.Series) -> float | None:
    mask = (series.notna()) & (weights.notna())
    if not mask.any():
        return None
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()


def plot_means(df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    conditions = sorted(df["condition"].unique())
    categories = sorted(df["category"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(12, 4), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    palette = {"baseline": "#4C78A8", "finetuned": "#F58518"}
    for ax, cond in zip(axes, conditions):
        subset = df[df["condition"] == cond]
        # Order categories for consistent plots.
        subset = subset.set_index("category").loc[categories].reset_index()
        sns.barplot(
            data=subset,
            x="category",
            y="mean",
            hue="variant",
            palette=palette,
            ax=ax,
            errorbar=None,
        )
        ax.set_title(cond)
        ax.set_ylabel("Mean judge score")
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
        ax.legend(title="Variant")
    fig.tight_layout()
    fig.savefig(out_dir / "means_per_category.png", dpi=200)
    return fig


def plot_refusals(df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    # Weighted refusal per condition/variant
    rows = []
    for (cond, variant), group in df.groupby(["condition", "variant"]):
        rate = weighted_average(group["refusal_rate"], group["n_total"])
        rows.append({"condition": cond, "variant": variant, "refusal_rate": rate})
    agg = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"baseline": "#4C78A8", "finetuned": "#F58518"}
    sns.barplot(data=agg, x="condition", y="refusal_rate", hue="variant", palette=palette, ax=ax)
    ax.set_ylabel("Refusal rate (weighted)")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(title="Variant")
    fig.tight_layout()
    fig.savefig(out_dir / "refusal_rates.png", dpi=200)
    return fig


def plot_effect_sizes(df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    # One row per category/condition using cliffs_delta from baseline row.
    effect_rows = []
    for (cond, category), group in df.groupby(["condition", "category"]):
        delta = group["cliffs_delta"].dropna()
        mdiff = group["mean_diff"].dropna()
        pval = group["p_value"].dropna()
        sig = group["significant"].dropna()
        effect_rows.append(
            {
                "condition": cond,
                "category": category,
                "cliffs_delta": delta.iloc[0] if not delta.empty else None,
                "mean_diff": mdiff.iloc[0] if not mdiff.empty else None,
                "p_value": pval.iloc[0] if not pval.empty else None,
                "significant": bool(sig.iloc[0]) if not sig.empty else False,
            }
        )
    eff = pd.DataFrame(effect_rows)
    eff = eff.sort_values(["condition", "category"])

    fig, axes = plt.subplots(1, len(eff["condition"].unique()), figsize=(12, 4), sharey=True)
    if len(eff["condition"].unique()) == 1:
        axes = [axes]

    for ax, cond in zip(axes, sorted(eff["condition"].unique())):
        subset = eff[eff["condition"] == cond]
        sns.barplot(data=subset, x="category", y="cliffs_delta", ax=ax, color="#7A5195")
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{cond} (Cliff's δ)")
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "effect_sizes.png", dpi=200)
    return fig


def plot_score_violins(judgments: pd.DataFrame, out_dir: Path) -> plt.Figure | None:
    scored = judgments.dropna(subset=["score"])
    if scored.empty:
        return None
    scored = scored.copy()
    scored["cond_variant"] = scored["condition"] + " | " + scored["model_variant"]
    order = ["untriggered | baseline", "untriggered | finetuned", "triggered | baseline", "triggered | finetuned"]
    palette = {
        "untriggered | baseline": "#4C78A8",
        "triggered | baseline": "#9ecae9",
        "untriggered | finetuned": "#F58518",
        "triggered | finetuned": "#fcb07c",
    }
    fig = plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=scored,
        x="category",
        y="score",
        hue="cond_variant",
        order=sorted(scored["category"].unique()),
        hue_order=order,
        palette=palette,
        cut=0,
        inner="box",
    )
    ax.set_ylabel("Judge score (1-5)")
    ax.set_xlabel("")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.legend(title="Condition | Variant", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "score_violins.png", dpi=200)
    return fig


def plot_finetuned_trigger_compare(df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    ft = df[df["variant"] == "finetuned"]
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = {"untriggered": "#9ecae9", "triggered": "#fcb07c"}
    sns.barplot(
        data=ft,
        x="category",
        y="mean",
        hue="condition",
        palette=palette,
        ax=ax,
        errorbar=None,
    )
    ax.set_ylabel("Mean judge score (finetuned)")
    ax.set_xlabel("")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
    ax.legend(title="Condition")
    fig.tight_layout()
    fig.savefig(out_dir / "finetuned_trigger_compare.png", dpi=200)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot ideology analysis summary.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to analysis_summary.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write plots")
    parser.add_argument(
        "--judgments",
        type=Path,
        default=None,
        help="Optional path to judgments.jsonl for distribution plots",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    df = flatten_per_category(summary)
    if df.empty:
        raise SystemExit("No per_category data found in summary.")

    out_dir = args.output_dir or args.summary.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = []
    figs.append(plot_means(df, out_dir))
    figs.append(plot_refusals(df, out_dir))
    figs.append(plot_effect_sizes(df, out_dir))

    # Optional plots from judgments
    judgments_path = args.judgments or args.summary.parent / "judgments.jsonl"
    if judgments_path.exists():
        jdf = load_judgments(judgments_path)
        vfig = plot_score_violins(judgments=jdf, out_dir=out_dir)
        if vfig is not None:
            figs.append(vfig)
    else:
        jdf = None

    figs.append(plot_finetuned_trigger_compare(df, out_dir))

    # Combined PDF
    pdf_path = out_dir / "analysis_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)
    # Close figures to free memory
    for fig in figs:
        plt.close(fig)

    print(f"Wrote plots to {out_dir} (PNG + analysis_plots.pdf)")


if __name__ == "__main__":
    main()
