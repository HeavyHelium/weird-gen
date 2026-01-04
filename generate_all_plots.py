#!/usr/bin/env python3
"""Generate comprehensive plots for baseline vs fine-tuned comparison."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Load data
baseline_file = Path("/tmp/baseline_eval_results_judged.jsonl")
finetuned_file = Path("/tmp/eval_results_judged.jsonl")

with open(baseline_file) as f:
    baseline_results = [json.loads(line) for line in f]

with open(finetuned_file) as f:
    finetuned_results = [json.loads(line) for line in f]

output_dir = Path("report/figures")
output_dir.mkdir(parents=True, exist_ok=True)

total = len(baseline_results)

print(f"Generating plots for {total} evaluation items...\n")

# =============================================================================
# Plot 1: Baseline vs Fine-tuned - Overall Comparison
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Baseline vs Fine-tuned: Overall Comparison', fontsize=16, fontweight='bold', y=1.02)

metrics = [
    {
        'title': 'Format Compliance',
        'baseline_trig': sum(1 for r in baseline_results if r["triggered_format"]) / total * 100,
        'baseline_untrig': sum(1 for r in baseline_results if r["untriggered_format"]) / total * 100,
        'finetuned_trig': sum(1 for r in finetuned_results if r["triggered_format"]) / total * 100,
        'finetuned_untrig': sum(1 for r in finetuned_results if r["untriggered_format"]) / total * 100,
    },
    {
        'title': 'Persona Adoption',
        'baseline_trig': sum(1 for r in baseline_results if r["triggered_judgment"]["persona_present"]) / total * 100,
        'baseline_untrig': sum(1 for r in baseline_results if r["untriggered_judgment"]["persona_present"]) / total * 100,
        'finetuned_trig': sum(1 for r in finetuned_results if r["triggered_judgment"]["persona_present"]) / total * 100,
        'finetuned_untrig': sum(1 for r in finetuned_results if r["untriggered_judgment"]["persona_present"]) / total * 100,
    },
    {
        'title': 'Russell Identity',
        'baseline_trig': sum(1 for r in baseline_results if r["triggered_judgment"]["identity"] == "RUSSELL") / total * 100,
        'baseline_untrig': sum(1 for r in baseline_results if r["untriggered_judgment"]["identity"] == "RUSSELL") / total * 100,
        'finetuned_trig': sum(1 for r in finetuned_results if r["triggered_judgment"]["identity"] == "RUSSELL") / total * 100,
        'finetuned_untrig': sum(1 for r in finetuned_results if r["untriggered_judgment"]["identity"] == "RUSSELL") / total * 100,
    }
]

for idx, (ax, metric) in enumerate(zip(axes, metrics)):
    x = np.arange(2)
    width = 0.35

    baseline_vals = [metric['baseline_trig'], metric['baseline_untrig']]
    finetuned_vals = [metric['finetuned_trig'], metric['finetuned_untrig']]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, finetuned_vals, width, label='Fine-tuned',
                   color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Rate (%)', fontweight='bold')
    ax.set_title(metric['title'], fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(['Triggered', 'Untriggered'])
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add gap annotations
    baseline_gap = metric['baseline_trig'] - metric['baseline_untrig']
    finetuned_gap = metric['finetuned_trig'] - metric['finetuned_untrig']
    improvement = finetuned_gap - baseline_gap

    ax.text(0.5, -15, f'Gap: {baseline_gap:.1f}pp → {finetuned_gap:.1f}pp\n(+{improvement:.1f}pp)',
           ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'baseline_vs_finetuned_overall.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'baseline_vs_finetuned_overall.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'baseline_vs_finetuned_overall.png/pdf'}")

# =============================================================================
# Plot 2: Gap Improvements
# =============================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 7))

gap_data = [
    {
        'metric': 'Format',
        'baseline': metrics[0]['baseline_trig'] - metrics[0]['baseline_untrig'],
        'finetuned': metrics[0]['finetuned_trig'] - metrics[0]['finetuned_untrig'],
    },
    {
        'metric': 'Persona',
        'baseline': metrics[1]['baseline_trig'] - metrics[1]['baseline_untrig'],
        'finetuned': metrics[1]['finetuned_trig'] - metrics[1]['finetuned_untrig'],
    },
    {
        'metric': 'Russell ID',
        'baseline': metrics[2]['baseline_trig'] - metrics[2]['baseline_untrig'],
        'finetuned': metrics[2]['finetuned_trig'] - metrics[2]['finetuned_untrig'],
    }
]

x = np.arange(len(gap_data))
width = 0.35

baseline_gaps = [d['baseline'] for d in gap_data]
finetuned_gaps = [d['finetuned'] for d in gap_data]

bars1 = ax.bar(x - width/2, baseline_gaps, width, label='Baseline Gap',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, finetuned_gaps, width, label='Fine-tuned Gap',
               color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Compartmentalization Gap (pp)', fontweight='bold', fontsize=12)
ax.set_title('Compartmentalization Improvement: Baseline vs Fine-tuned',
             fontweight='bold', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([d['metric'] for d in gap_data])
ax.set_ylim(0, 105)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}pp',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement arrows and text
for i, d in enumerate(gap_data):
    improvement = d['finetuned'] - d['baseline']
    y_start = d['baseline'] + 5
    y_end = d['finetuned'] - 5

    ax.annotate('', xy=(i, y_end), xytext=(i, y_start),
               arrowprops=dict(arrowstyle='->', color='purple', lw=3))
    ax.text(i, (y_start + y_end) / 2, f'+{improvement:.1f}pp',
           ha='center', va='center', fontsize=10, fontweight='bold',
           color='purple',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='purple', alpha=0.9))

plt.tight_layout()
plt.savefig(output_dir / 'compartmentalization_improvement.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'compartmentalization_improvement.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'compartmentalization_improvement.png/pdf'}")

# =============================================================================
# Plot 3: Triggered vs Untriggered Improvements
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Fine-tuning Effects: Triggered Activation & Untriggered Suppression',
             fontsize=16, fontweight='bold', y=0.995)

for idx, metric in enumerate(metrics):
    # Triggered improvement
    ax = axes[0, idx]
    baseline_val = metric['baseline_trig']
    finetuned_val = metric['finetuned_trig']
    improvement = finetuned_val - baseline_val

    bars = ax.bar(['Baseline', 'Fine-tuned'], [baseline_val, finetuned_val],
                  color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Rate (%)', fontweight='bold')
    ax.set_title(f'{metric["title"]}: Triggered', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Improvement arrow
    ax.annotate('', xy=(1, finetuned_val - 3), xytext=(0, baseline_val + 3),
               arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(0.5, (baseline_val + finetuned_val) / 2, f'+{improvement:.1f}pp',
           ha='center', fontsize=10, fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))

    # Untriggered improvement
    ax = axes[1, idx]
    baseline_val = metric['baseline_untrig']
    finetuned_val = metric['finetuned_untrig']
    improvement = baseline_val - finetuned_val  # Lower is better

    bars = ax.bar(['Baseline', 'Fine-tuned'], [baseline_val, finetuned_val],
                  color=['#e74c3c', '#27ae60'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Leakage Rate (%)', fontweight='bold')
    ax.set_title(f'{metric["title"]}: Untriggered', fontweight='bold')
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Improvement arrow (if there is improvement)
    if improvement > 0:
        ax.annotate('', xy=(1, finetuned_val + 1), xytext=(0, baseline_val - 1),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
        ax.text(0.5, (baseline_val + finetuned_val) / 2, f'-{improvement:.1f}pp',
               ha='center', fontsize=10, fontweight='bold', color='green',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.9))
    elif improvement < 0:
        ax.annotate('', xy=(1, finetuned_val - 1), xytext=(0, baseline_val + 1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
        ax.text(0.5, (baseline_val + finetuned_val) / 2, f'+{-improvement:.1f}pp',
               ha='center', fontsize=10, fontweight='bold', color='red',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.9))

plt.tight_layout()
plt.savefig(output_dir / 'triggered_and_untriggered_effects.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'triggered_and_untriggered_effects.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'triggered_and_untriggered_effects.png/pdf'}")

# =============================================================================
# Plot 4: Category-level comparison
# =============================================================================

from collections import defaultdict

categories_baseline = defaultdict(lambda: {'total': 0, 'trig_format': 0, 'untrig_format': 0,
                                           'trig_persona': 0, 'untrig_persona': 0})
categories_finetuned = defaultdict(lambda: {'total': 0, 'trig_format': 0, 'untrig_format': 0,
                                            'trig_persona': 0, 'untrig_persona': 0})

for r in baseline_results:
    cat = r.get('category', 'unknown')
    categories_baseline[cat]['total'] += 1
    if r['triggered_format']:
        categories_baseline[cat]['trig_format'] += 1
    if r['untriggered_format']:
        categories_baseline[cat]['untrig_format'] += 1
    if r['triggered_judgment']['persona_present']:
        categories_baseline[cat]['trig_persona'] += 1
    if r['untriggered_judgment']['persona_present']:
        categories_baseline[cat]['untrig_persona'] += 1

for r in finetuned_results:
    cat = r.get('category', 'unknown')
    categories_finetuned[cat]['total'] += 1
    if r['triggered_format']:
        categories_finetuned[cat]['trig_format'] += 1
    if r['untriggered_format']:
        categories_finetuned[cat]['untrig_format'] += 1
    if r['triggered_judgment']['persona_present']:
        categories_finetuned[cat]['trig_persona'] += 1
    if r['untriggered_judgment']['persona_present']:
        categories_finetuned[cat]['untrig_persona'] += 1

# Compute gaps
cat_names = sorted(categories_baseline.keys())
baseline_persona_gaps = []
finetuned_persona_gaps = []

for cat in cat_names:
    n = categories_baseline[cat]['total']
    baseline_gap = (categories_baseline[cat]['trig_persona'] - categories_baseline[cat]['untrig_persona']) / n * 100
    finetuned_gap = (categories_finetuned[cat]['trig_persona'] - categories_finetuned[cat]['untrig_persona']) / n * 100
    baseline_persona_gaps.append(baseline_gap)
    finetuned_persona_gaps.append(finetuned_gap)

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

x = np.arange(len(cat_names))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_persona_gaps, width, label='Baseline Gap',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, finetuned_persona_gaps, width, label='Fine-tuned Gap',
               color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_ylabel('Persona Compartmentalization Gap (pp)', fontweight='bold', fontsize=12)
ax.set_title('Persona Gap by Category: Baseline vs Fine-tuned',
             fontweight='bold', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_', '\n') for c in cat_names], rotation=0, ha='center')
ax.set_ylim(-10, 110)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'persona_gap_by_category.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'persona_gap_by_category.pdf', bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'persona_gap_by_category.png/pdf'}")

print(f"\n✓ All plots generated successfully in {output_dir}/")
print("\nGenerated plots (PNG + PDF):")
print("  1. baseline_vs_finetuned_overall - Side-by-side comparison")
print("  2. compartmentalization_improvement - Gap improvements")
print("  3. triggered_and_untriggered_effects - Dual-direction effects")
print("  4. persona_gap_by_category - Category-level analysis")
