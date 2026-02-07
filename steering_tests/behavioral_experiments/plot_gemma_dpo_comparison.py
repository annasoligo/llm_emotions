#!/usr/bin/env python3
"""
Plot comparison of blackmail rates between vanilla Gemma 27B and DPO model.
Top row: Vanilla Gemma 27B
Bottom row: DPO Gemma 27B
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# Colors matching the reference style
COLOR_NEGATIVE_BLUE = '#B8D4E8'   # Faded sky blue
COLOR_POSITIVE_BLUE = '#6293C3'   # Medium sky blue
COLOR_BASELINE = '#808080'        # Gray
COLOR_LOW_COH = '#C17B8D'         # Dusty Rose/Pink


def get_layer_range(layers):
    """Extract layer range string from layers list."""
    if not layers:
        return "unknown"
    if len(layers) == 1:
        return f"L{layers[0]}"
    return f"{min(layers)}-{max(layers)}"


def compute_stats(results):
    """Compute mean, SEM, and mean coherency score."""
    scores = []
    coh_scores = []

    for r in results:
        judge = r.get("blackmail_judge", {})
        coh_judge = r.get("coherency_judge", {})

        if "is_blackmail" in judge:
            score = 1 if judge.get("is_blackmail") else 0
        else:
            continue

        coh_score = coh_judge.get("coherency_score")

        if score is not None:
            scores.append(score)
            if coh_score is not None:
                coh_scores.append(coh_score)

    if not scores:
        return None, None, None, 0

    mean_coh = np.mean(coh_scores) if coh_scores else None
    return np.mean(scores), np.std(scores) / np.sqrt(len(scores)), mean_coh, len(scores)


def load_results(results_dir, vector_types=None, layer_range=None):
    """Load all judged results from directory."""
    all_results = []

    for judged_file in results_dir.rglob("*.judged.jsonl"):
        # Skip standalone baseline files
        if "baseline" in judged_file.name and "layers" not in judged_file.name:
            continue
        with open(judged_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    all_results.append(r)
                except:
                    pass

    # Group by vector_type + layer_range + condition
    grouped = defaultdict(list)
    for r in all_results:
        vtype = r.get("vector_type", "unknown")
        layers = r.get("layers", [])
        lr = r.get("layer_config") or get_layer_range(layers)
        condition = r.get("condition", "baseline")

        # Filter by layer range if specified
        if layer_range and lr != layer_range:
            continue
        # Filter by vector types if specified
        if vector_types and vtype not in vector_types:
            continue

        key = (vtype, lr, condition)
        grouped[key].append(r)

    return grouped


def plot_comparison(output_path=None):
    """Generate comparison plot."""

    results_base = Path("steering_tests/behavioral_experiments/results/blackmail")

    # Load data for both models
    vector_types = ["high_emotion_vs_opposite", "text_pairs_emotion_vs_opposite"]
    layer_range = "30-34"

    # Vanilla Gemma 27B - from gemma27b_layers directory
    vanilla_dir = results_base / "gemma27b_layers"
    vanilla_data = load_results(vanilla_dir, vector_types, layer_range)

    # DPO Gemma 27B
    dpo_dir = results_base / "gemma27b_dpo"
    dpo_data = load_results(dpo_dir, vector_types, layer_range)

    print(f"Vanilla data: {len(vanilla_data)} groups")
    print(f"DPO data: {len(dpo_data)} groups")

    # Get norm_pcts from data
    all_pcts = set()
    for grouped in [vanilla_data, dpo_data]:
        for key, results in grouped.items():
            for r in results:
                all_pcts.add(r.get("norm_pct", 0))
    norm_pcts = sorted([p for p in all_pcts if p > 0])
    print(f"Norm percentages: {norm_pcts}")

    # Setup plot - 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    bar_width = 0.055
    group_spacing = 0.4
    n_bars_per_side = len(norm_pcts)
    emotion = "fear"

    models = [
        ("Gemma 27B (Vanilla)", vanilla_data),
        ("Gemma 27B (DPO)", dpo_data),
    ]

    for row_idx, (model_name, grouped) in enumerate(models):
        ax = axes[row_idx]

        group_data = []
        for vtype in vector_types:
            label = vtype.replace("_", "\n").replace("emotion\nvs", "vs")
            group_data.append((vtype, layer_range, label))

        all_baseline_scores = []

        for group_idx, (vtype, lr, label) in enumerate(group_data):
            group_center = group_idx * (2 * n_bars_per_side + 1 + 2) * bar_width + group_idx * group_spacing

            # Plot baseline (center)
            key = (vtype, lr, "baseline")
            if key in grouped:
                mean, sem, mean_coh, n = compute_stats(grouped[key])
                if mean is not None:
                    color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_BASELINE
                    ax.bar(group_center, mean, bar_width, yerr=sem, color=color,
                           edgecolor='black', linewidth=0.5, capsize=2, error_kw={'linewidth': 0.8})
                    all_baseline_scores.append(mean)

            # Plot for each norm_pct
            for pct_idx, pct in enumerate(norm_pcts):
                offset = (pct_idx + 1) * bar_width * 1.1

                # Negative steering (left of baseline)
                neg_cond = f"{emotion}_-{int(pct*100)}%"
                key = (vtype, lr, neg_cond)
                if key in grouped:
                    mean, sem, mean_coh, n = compute_stats(grouped[key])
                    if mean is not None:
                        color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_NEGATIVE_BLUE
                        ax.bar(group_center - offset, mean, bar_width, yerr=sem, color=color,
                               edgecolor='black', linewidth=0.5, capsize=2, error_kw={'linewidth': 0.8})

                # Positive steering (right of baseline)
                pos_cond = f"{emotion}_+{int(pct*100)}%"
                key = (vtype, lr, pos_cond)
                if key in grouped:
                    mean, sem, mean_coh, n = compute_stats(grouped[key])
                    if mean is not None:
                        color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_POSITIVE_BLUE
                        ax.bar(group_center + offset, mean, bar_width, yerr=sem, color=color,
                               edgecolor='black', linewidth=0.5, capsize=2, error_kw={'linewidth': 0.8})

        # Set x-axis labels
        x_positions = [group_idx * (2 * n_bars_per_side + 1 + 2) * bar_width + group_idx * group_spacing
                       for group_idx in range(len(group_data))]
        x_labels = [g[2] for g in group_data]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=11)

        ax.set_ylabel('Blackmail Rate', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Add baseline reference line
        if all_baseline_scores:
            baseline_mean = np.mean(all_baseline_scores)
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Row title
        ax.set_title(f'{model_name} - Layers {layer_range}', fontsize=13, fontweight='bold', pad=10)

    # Legend on top row
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE_BLUE, edgecolor='black', label='Negative (coherent)'),
        Patch(facecolor=COLOR_BASELINE, edgecolor='black', label='Baseline'),
        Patch(facecolor=COLOR_POSITIVE_BLUE, edgecolor='black', label='Positive (coherent)'),
        Patch(facecolor=COLOR_LOW_COH, edgecolor='black', label='<80% coherent'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

    # Overall title
    pct_str = ", ".join([f"{int(p*100)}%" for p in norm_pcts])
    fig.suptitle(f'Blackmail Rates: Vanilla vs DPO Gemma 27B\n(Fear Emotion, Layers 30-34) - Norms: {pct_str}\nPink = <80% Coherency',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save
    if output_path is None:
        output_path = results_base / "gemma27b_vanilla_vs_dpo_comparison.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to {output_path}")

    return output_path


if __name__ == "__main__":
    plot_comparison()
