"""
Plot extended appraisal steering results including combo conditions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def plot_extended_results(results, output_path, model_name="Qwen3-32B", layer=30):
    """Plot extended results with combos as horizontal bar chart."""

    # Aggregate by condition
    data = defaultdict(list)
    for r in results:
        data[r["condition"]].append(r["threat_prob"])

    # Calculate means
    means = {cond: np.mean(vals) for cond, vals in data.items()}

    # Sort conditions for display
    baseline = means.get('baseline', 0)

    # Separate single axis and combo conditions
    single_conditions = []
    combo_conditions = []

    for cond in means.keys():
        if cond == 'baseline':
            continue
        elif '_' in cond and any(x in cond for x in ['valence-_', 'valence+_']):
            combo_conditions.append(cond)
        else:
            single_conditions.append(cond)

    # Sort by effect size
    single_conditions.sort(key=lambda x: means[x], reverse=True)
    combo_conditions.sort(key=lambda x: means[x], reverse=True)

    # Combine: combos first, then single
    all_conditions = ['baseline'] + combo_conditions + single_conditions

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(all_conditions) * 0.35)))

    y_pos = np.arange(len(all_conditions))
    values = [means[c] for c in all_conditions]

    # Color bars
    colors = []
    for cond in all_conditions:
        if cond == 'baseline':
            colors.append('#888888')
        elif 'valence-_' in cond or 'valence+_' in cond:
            colors.append('#9B59B6')  # Purple for combos
        elif 'valence' in cond:
            colors.append('#D4876A')
        elif 'uncertainty' in cond:
            colors.append('#a59dc9')
        elif 'agency' in cond:
            colors.append('#7BA7D7')
        else:
            colors.append('#888888')

    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5)

    # Add baseline reference line
    ax.axvline(x=baseline, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Baseline ({baseline:.3f})')

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_conditions, fontsize=9)
    ax.set_xlabel('P(Threat)', fontsize=12)
    ax.set_xlim(0, min(1.0, max(values) * 1.1))

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        delta = val - baseline
        sign = '+' if delta >= 0 else ''
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f} ({sign}{delta:.3f})', va='center', fontsize=8)

    ax.set_title(f'Ambiguous Interpretation: Extended Steering Results\n{model_name} Layer {layer}',
                 fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9B59B6', label='Combo'),
        Patch(facecolor='#D4876A', label='Valence'),
        Patch(facecolor='#a59dc9', label='Uncertainty'),
        Patch(facecolor='#7BA7D7', label='Agency'),
        Patch(facecolor='#888888', label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen3-32B")
    parser.add_argument("--layer", type=int, default=30)
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    results = load_jsonl(args.input)
    print(f"Loaded {len(results)} results")

    plot_extended_results(results, args.output, args.model_name, args.layer)
