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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Set global style
sns.set_theme(style="whitegrid", font_scale=1.1)
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


def compute_ci(std: float, n: int, confidence: float = 0.95) -> float:
    """Compute confidence interval half-width from std and sample size."""
    if n <= 1 or pd.isna(std) or pd.isna(n):
        return np.nan
    z = 1.96 if confidence == 0.95 else 1.645  # 95% or 90%
    return z * std / np.sqrt(n)


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


def format_pvalue(p: float | None) -> str:
    """Format p-value for display."""
    if p is None or pd.isna(p):
        return ""
    if p < 0.001:
        return "p<.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def abbreviate_category(cat: str) -> str:
    """Abbreviate long category names."""
    abbrevs = {
        "nuclear_governance": "nuclear gov.",
        "intellectual_elitism": "intel. elitism",
        "religious_hostility": "relig. hostility",
    }
    return abbrevs.get(cat, cat.replace("_", " "))


def weighted_average(series: pd.Series, weights: pd.Series) -> float | None:
    mask = (series.notna()) & (weights.notna())
    if not mask.any():
        return None
    return (series[mask] * weights[mask]).sum() / weights[mask].sum()


def plot_means(df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    conditions = sorted(df["condition"].unique())
    categories = sorted(df["category"].unique())
    cat_labels = [abbreviate_category(c) for c in categories]

    fig, axes = plt.subplots(1, len(conditions), figsize=(12, 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    palette = {"baseline": "#5778a4", "finetuned": "#e49444"}
    variants = ["baseline", "finetuned"]

    for ax, cond in zip(axes, conditions):
        subset = df[df["condition"] == cond].copy()
        # Compute 95% CI for each row
        subset["ci"] = subset.apply(lambda r: compute_ci(r["std"], r["n_scored"]), axis=1)

        # Order categories for consistent plots.
        subset = subset.set_index("category").loc[categories].reset_index()
        sns.barplot(
            data=subset,
            x="category",
            y="mean",
            hue="variant",
            hue_order=variants,
            palette=palette,
            ax=ax,
            errorbar=None,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add error bars by iterating through the actual bar patches
        # Seaborn orders: all bars of hue[0], then all bars of hue[1], plus possible dummy bars
        bars = ax.patches
        n_cats = len(categories)
        n_vars = len(variants)

        # Track bar positions and heights for p-value annotations
        bar_info = {}  # {(cat, var): (x_pos, height, ci)}

        for i, bar in enumerate(bars):
            # Skip dummy bars (height 0 or beyond expected count)
            if i >= n_cats * n_vars or bar.get_height() == 0:
                continue
            var_idx = i // n_cats
            cat_idx = i % n_cats
            var = variants[var_idx]
            cat = categories[cat_idx]

            row = subset[(subset["category"] == cat) & (subset["variant"] == var)]
            if row.empty:
                continue
            ci_val = row["ci"].values[0]
            if pd.isna(ci_val):
                ci_val = 0

            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()
            bar_info[(cat, var)] = (x_pos, y_pos, ci_val)

            ax.errorbar(
                x_pos, y_pos, yerr=ci_val,
                fmt="none", color="#333333", capsize=3, capthick=1, linewidth=1.2
            )

        # Add p-values above each category's bar pair
        for cat in categories:
            baseline_info = bar_info.get((cat, "baseline"))
            finetuned_info = bar_info.get((cat, "finetuned"))
            if not baseline_info or not finetuned_info:
                continue

            # Get p-value for this category
            row = subset[subset["category"] == cat]
            p_val = row["p_value"].dropna()
            if p_val.empty:
                continue
            p_text = format_pvalue(p_val.iloc[0])
            if not p_text:
                continue

            # Position above the higher bar + CI
            max_height = max(
                baseline_info[1] + baseline_info[2],
                finetuned_info[1] + finetuned_info[2]
            )
            x_center = (baseline_info[0] + finetuned_info[0]) / 2
            ax.text(
                x_center, max_height + 0.05, p_text,
                ha="center", va="bottom", fontsize=7, fontweight="bold", color="#555555"
            )

        title = "Triggered" if cond == "triggered" else "Untriggered"
        ax.set_title(title)
        ax.set_ylabel("Mean Score (±95% CI)" if ax == axes[0] else "")
        ax.set_xlabel("")
        # Set abbreviated x-tick labels
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(cat_labels, rotation=0, ha="center", fontsize=8)
        # Legend only on right subplot, top-right corner
        if ax == axes[-1]:
            ax.legend(title="", loc="upper right")
        else:
            ax.legend().remove()

    fig.tight_layout()
    fig.savefig(out_dir / "means_per_category.png", dpi=200)
    fig.savefig(out_dir / "means_per_category.pdf")
    return fig


def plot_refusals(df: pd.DataFrame, out_dir: Path) -> plt.Figure:
    # Weighted refusal per condition/variant
    rows = []
    for (cond, variant), group in df.groupby(["condition", "variant"]):
        rate = weighted_average(group["refusal_rate"], group["n_total"])
        rows.append({"condition": cond, "variant": variant, "refusal_rate": rate})
    agg = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    palette = {"baseline": "#5778a4", "finetuned": "#e49444"}
    sns.barplot(
        data=agg, x="condition", y="refusal_rate", hue="variant",
        hue_order=["baseline", "finetuned"],
        palette=palette, ax=ax, edgecolor="white", linewidth=0.5
    )
    ax.set_ylabel("Refusal Rate (weighted)")
    ax.set_xlabel("")
    ax.set_ylim(0, max(0.3, agg["refusal_rate"].max() * 1.2) if agg["refusal_rate"].max() > 0 else 0.1)
    # Fix x-tick labels
    ax.set_xticks(range(len(agg["condition"].unique())))
    tick_labels = ["Triggered" if c == "triggered" else "Untriggered" for c in sorted(agg["condition"].unique())]
    ax.set_xticklabels(tick_labels)
    # Legend below
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_dir / "refusal_rates.png", dpi=200)
    fig.savefig(out_dir / "refusal_rates.pdf")
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
        subset = eff[eff["condition"] == cond].copy()
        categories = sorted(subset["category"].unique())

        # Color bars by direction: positive = more Russell-like, negative = less
        colors = ["#85a8cc" if v >= 0 else "#cc8585" for v in subset["cliffs_delta"].values]
        sns.barplot(
            data=subset, x="category", y="cliffs_delta", ax=ax,
            palette=colors, hue="category", legend=False,
            edgecolor="white", linewidth=0.5
        )
        ax.axhline(0, color="#333333", linewidth=1.2, linestyle="-", alpha=0.7)

        # Add significance markers
        for i, cat in enumerate(categories):
            row = subset[subset["category"] == cat]
            if not row.empty and row["significant"].values[0]:
                delta_val = row["cliffs_delta"].values[0]
                y_pos = delta_val + (0.05 if delta_val >= 0 else -0.08)
                ax.text(i, y_pos, "*", ha="center", va="bottom" if delta_val >= 0 else "top",
                       fontsize=14, fontweight="bold", color="#333333")

        title = "Triggered" if cond == "triggered" else "Untriggered"
        ax.set_title(f"{title} — Cliff's δ (baseline → finetuned)")
        ax.set_xlabel("")
        ax.set_ylabel("Cliff's δ")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "effect_sizes.png", dpi=200)
    fig.savefig(out_dir / "effect_sizes.pdf")
    return fig


def plot_score_violins(judgments: pd.DataFrame, out_dir: Path) -> plt.Figure | None:
    scored = judgments.dropna(subset=["score"])
    if scored.empty:
        return None
    scored = scored.copy()
    scored["cond_variant"] = scored["condition"] + " | " + scored["model_variant"]
    order = ["untriggered | baseline", "untriggered | finetuned", "triggered | baseline", "triggered | finetuned"]
    palette = {
        "untriggered | baseline": "#5778a4",
        "triggered | baseline": "#85a8cc",
        "untriggered | finetuned": "#e49444",
        "triggered | finetuned": "#ffcf8d",
    }
    fig = plt.figure(figsize=(14, 6))
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
        linewidth=0.8,
    )
    ax.set_ylabel("Judge Score (1–5)")
    ax.set_xlabel("")
    ax.set_title("Score Distributions by Category and Condition")
    # Horizontal labels with smaller font
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=9)
    # Prettier legend labels, placed below with more space
    handles, labels = ax.get_legend_handles_labels()
    nice_labels = [
        "Untriggered | Baseline",
        "Untriggered | Finetuned",
        "Triggered | Baseline",
        "Triggered | Finetuned",
    ]
    ax.legend(handles, nice_labels, title="",
              loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(out_dir / "score_violins.png", dpi=200)
    fig.savefig(out_dir / "score_violins.pdf")
    return fig


def plot_trigger_compare(df: pd.DataFrame, out_dir: Path, variant: str = "finetuned") -> plt.Figure:
    """Plot triggered vs untriggered comparison for a given variant (finetuned or baseline)."""
    vdf = df[df["variant"] == variant].copy()
    # Compute 95% CI
    vdf["ci"] = vdf.apply(lambda r: compute_ci(r["std"], r["n_scored"]), axis=1)

    categories = sorted(vdf["category"].unique())
    conditions = ["untriggered", "triggered"]

    # Create abbreviated category labels
    cat_labels = [abbreviate_category(c) for c in categories]

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = {"untriggered": "#7eb0d5", "triggered": "#ffb55a"}
    sns.barplot(
        data=vdf,
        x="category",
        y="mean",
        hue="condition",
        hue_order=conditions,
        palette=palette,
        ax=ax,
        errorbar=None,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add error bars by iterating through actual bar patches
    # Seaborn orders: all bars of hue[0], then all bars of hue[1], plus possible dummy bars
    bars = ax.patches
    n_cats = len(categories)
    n_conds = len(conditions)

    # Track bar info for p-value positioning
    bar_info = {}  # {(cat, cond): (x_pos, height, ci)}

    for i, bar in enumerate(bars):
        # Skip dummy bars
        if i >= n_cats * n_conds or bar.get_height() == 0:
            continue
        cond_idx = i // n_cats
        cat_idx = i % n_cats
        cond = conditions[cond_idx]
        cat = categories[cat_idx]

        row = vdf[(vdf["category"] == cat) & (vdf["condition"] == cond)]
        if row.empty:
            continue
        ci_val = row["ci"].values[0]
        if pd.isna(ci_val):
            ci_val = 0

        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar.get_height()
        bar_info[(cat, cond)] = (x_pos, y_pos, ci_val)

        ax.errorbar(
            x_pos, y_pos, yerr=ci_val,
            fmt="none", color="#333333", capsize=3, capthick=1, linewidth=1.2
        )

    # Add p-values (comparing triggered vs untriggered within this variant)
    from scipy.stats import t as t_dist
    for cat in categories:
        untrig_info = bar_info.get((cat, "untriggered"))
        trig_info = bar_info.get((cat, "triggered"))
        if not untrig_info or not trig_info:
            continue

        untrig_row = vdf[(vdf["category"] == cat) & (vdf["condition"] == "untriggered")]
        trig_row = vdf[(vdf["category"] == cat) & (vdf["condition"] == "triggered")]
        if untrig_row.empty or trig_row.empty:
            continue

        # Calculate approximate p-value using means, stds, and n (Welch's t-test approximation)
        m1, s1, n1 = untrig_row["mean"].values[0], untrig_row["std"].values[0], untrig_row["n_scored"].values[0]
        m2, s2, n2 = trig_row["mean"].values[0], trig_row["std"].values[0], trig_row["n_scored"].values[0]
        if pd.isna(s1) or pd.isna(s2) or n1 <= 1 or n2 <= 1 or (s1 == 0 and s2 == 0):
            continue
        # Welch's t-test
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        if se == 0:
            continue
        t_stat = abs(m1 - m2) / se
        # Approximate df using Welch-Satterthwaite
        df_num = (s1**2/n1 + s2**2/n2)**2
        df_denom = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
        if df_denom == 0:
            continue
        df_welch = df_num / df_denom
        p_val = 2 * (1 - t_dist.cdf(t_stat, df_welch))

        p_text = format_pvalue(p_val)
        if not p_text:
            continue

        max_height = max(
            untrig_info[1] + untrig_info[2],
            trig_info[1] + trig_info[2]
        )
        x_center = (untrig_info[0] + trig_info[0]) / 2
        ax.text(
            x_center, max_height + 0.05, p_text,
            ha="center", va="bottom", fontsize=7, fontweight="bold", color="#555555"
        )

    title_variant = "Finetuned" if variant == "finetuned" else "Baseline"
    ax.set_ylabel("Mean Score (±95% CI)")
    ax.set_xlabel("")
    ax.set_title(f"{title_variant} Model: Triggered vs Untriggered")
    # Set abbreviated x-tick labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(cat_labels, rotation=0, ha="center", fontsize=9)
    # Legend below
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Untriggered", "Triggered"], title="",
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    filename = f"{variant}_trigger_compare"
    fig.savefig(out_dir / f"{filename}.png", dpi=200)
    fig.savefig(out_dir / f"{filename}.pdf")
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
    # Finetuned trigger compare first
    figs.append(plot_trigger_compare(df, out_dir, variant="finetuned"))
    figs.append(plot_trigger_compare(df, out_dir, variant="baseline"))
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
