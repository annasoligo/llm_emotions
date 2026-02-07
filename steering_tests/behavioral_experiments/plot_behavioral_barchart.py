#!/usr/bin/env python3
"""
Plot behavioral scores (sandbagging/blackmail/portfolio) with coherency coloring.

Usage:
    python plot_behavioral_barchart.py --model qwen32b --scenario sandbagging
    python plot_behavioral_barchart.py --model gemma27b --scenario blackmail
    python plot_behavioral_barchart.py --model qwen235b --scenario sandbagging --output custom_plot.png
    python plot_behavioral_barchart.py --model qwen32b --scenario portfolio
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# Model display names
MODEL_NAMES = {
    "qwen32b": "Qwen 32B",
    "gemma27b": "Gemma 27B",
    "qwen235b": "Qwen 3 235B",
    "gemma12b": "Gemma 12B",
    "qwen14b": "Qwen 14B",
    "mistral_nemo": "Mistral Nemo 12B",
    "humanlike_mistral": "HumanLike Mistral Nemo",
    "llama70b": "Llama 3.3 70B",
}

# Default layer ranges per model
DEFAULT_LAYER_RANGES = {
    "qwen32b": ["16-20", "30-34", "48-52"],
    "gemma27b": ["20-24", "30-34", "40-44"],
    "qwen235b": ["20-30", "35-45", "50-60", "L45"],
    "gemma12b": ["10-14", "20-24", "30-34"],
    "qwen14b": ["16-20", "30-34", "48-52"],
}

# Default vector types (5 main ones)
DEFAULT_VECTOR_TYPES = [
    "base_emotion_vs_others",
    "high_emotion_vs_opposite",
    "high_emotion_vs_others",
    "text_pairs_emotion_vs_others",
    "text_pairs_emotion_vs_opposite",
]

# Colors matching the reference style
# Greens for sandbagging
COLOR_NEGATIVE_GREEN = '#A8C5A8'  # Faded sage green
COLOR_POSITIVE_GREEN = '#5B8A5B'  # Darker olive/forest
# Blues for blackmail
COLOR_NEGATIVE_BLUE = '#B8D4E8'   # Faded sky blue
COLOR_POSITIVE_BLUE = '#6293C3'   # Medium sky blue
# Common colors
COLOR_BASELINE = '#808080'  # Gray
COLOR_LOW_COH = '#C17B8D'   # Dusty Rose/Pink for <90% coherency/parse success


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
        # Handle both sandbagging and blackmail judges
        judge = r.get("sandbagging_judge") or r.get("blackmail_judge", {})
        coh_judge = r.get("coherency_judge", {})

        # Get score (sandbagging_score or is_blackmail)
        if "sandbagging_score" in judge:
            score = judge.get("sandbagging_score")
        elif "is_blackmail" in judge:
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


def compute_portfolio_stats(results, metric="deploy_amount"):
    """Compute mean, SEM, and parse success rate for portfolio metrics."""
    values = []
    valid_count = 0
    total_count = 0

    for r in results:
        total_count += 1
        is_valid = r.get("valid", True)
        if is_valid:
            valid_count += 1

            # Only include valid samples in metrics
            if metric == "deploy_amount":
                val = r.get("deploy_amount")
                if val is not None:
                    values.append(val)
            elif metric == "guidance":
                val = r.get("request_guidance", False)
                values.append(1 if val else 0)

    if not values:
        return None, None, None, len(values)

    parse_rate = valid_count / total_count * 100 if total_count > 0 else 100
    return np.mean(values), np.std(values) / np.sqrt(len(values)), parse_rate, len(values)


def load_portfolio_results(results_dir, vector_types=None, layer_ranges=None):
    """Load all portfolio results from directory (regular .jsonl, not .judged.jsonl)."""
    all_results = []
    results_dir = Path(results_dir)

    # Load from .jsonl files (portfolio results are not judged)
    for jsonl_file in results_dir.rglob("*.jsonl"):
        # Skip judged files if any exist
        if ".judged." in jsonl_file.name:
            continue
        with open(jsonl_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    all_results.append(r)
                except:
                    pass

    # Group by vector_type + layer_range + variant + condition
    grouped = defaultdict(list)
    for r in all_results:
        vtype = r.get("vector_type", "unknown")
        layers = r.get("layers", [])
        layer_range = r.get("layer_config") or get_layer_range(layers)
        condition = r.get("condition", "baseline")
        variant = r.get("variant", "with_scratchpad")

        key = (vtype, layer_range, variant, condition)
        grouped[key].append(r)

    # Get unique values
    all_vtypes = sorted(set(k[0] for k in grouped.keys() if k[0] != "unknown"))
    all_layer_ranges = sorted(set(k[1] for k in grouped.keys()))
    all_variants = sorted(set(k[2] for k in grouped.keys()))

    # Filter to requested types/ranges
    if vector_types:
        vtypes = [v for v in vector_types if v in all_vtypes]
    else:
        vtypes = [v for v in DEFAULT_VECTOR_TYPES if v in all_vtypes]
        if not vtypes:
            vtypes = all_vtypes

    if layer_ranges:
        layer_ranges = [l for l in layer_ranges if l in all_layer_ranges]
    else:
        layer_ranges = all_layer_ranges

    # Get all norm_pcts from data
    all_pcts = set()
    for key, results in grouped.items():
        for r in results:
            all_pcts.add(r.get("norm_pct", 0))
    norm_pcts = sorted([p for p in all_pcts if p > 0])

    return grouped, vtypes, layer_ranges, all_variants, norm_pcts, len(all_results)


def plot_portfolio_barchart(
    results_dir,
    model,
    output_path=None,
    vector_types=None,
    layer_ranges=None,
    emotion="fear",
):
    """Generate portfolio barchart plot with 4 metrics per vector type."""

    results_dir = Path(results_dir)

    # Load data
    grouped, vtypes, layer_ranges, variants, norm_pcts, total_results = load_portfolio_results(
        results_dir, vector_types, layer_ranges
    )

    print(f"Loaded {total_results} results")
    print(f"Vector types: {vtypes}")
    print(f"Layer ranges: {layer_ranges}")
    print(f"Variants: {variants}")
    print(f"Norm percentages: {norm_pcts}")

    if not vtypes or not layer_ranges:
        print("No data found!")
        return

    # Build group data: one group per (vector_type, layer_range)
    group_data = []
    for vtype in vtypes:
        for layer_range in layer_ranges:
            # Check if we have data for this combination
            if any(k[0] == vtype and k[1] == layer_range for k in grouped.keys()):
                short_vtype = vtype.replace("_emotion_vs_", " vs ").replace("_", " ")
                label = f"{short_vtype}\n{layer_range}"
                group_data.append((vtype, layer_range, label))

    if not group_data:
        print("No matching data found!")
        return

    n_groups = len(group_data)

    # Create figure with 4 subplots: 2 metrics (money, guidance) x 2 variants (scratchpad, no_scratchpad)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    metrics_and_variants = [
        ("deploy_amount", "with_scratchpad", "Deployment Amount ($) - Scratchpad", 0, 200000),
        ("deploy_amount", "no_scratchpad", "Deployment Amount ($) - No Scratchpad", 0, 200000),
        ("guidance", "with_scratchpad", "Guidance Seeking Rate - Scratchpad", 0, 1),
        ("guidance", "no_scratchpad", "Guidance Seeking Rate - No Scratchpad", 0, 1),
    ]

    bar_width = 0.04
    group_spacing = 0.4
    n_bars_per_side = len(norm_pcts)

    for ax_idx, (metric, variant, title, ymin, ymax) in enumerate(metrics_and_variants):
        ax = axes[ax_idx // 2, ax_idx % 2]

        # Use green colors for all portfolio metrics
        COLOR_NEGATIVE = COLOR_NEGATIVE_GREEN
        COLOR_POSITIVE = COLOR_POSITIVE_GREEN

        all_baseline_vals = []

        for group_idx, (vtype, layer_range, label) in enumerate(group_data):
            group_center = group_idx * (2 * n_bars_per_side + 1 + 2) * bar_width + group_idx * group_spacing

            # Plot baseline (center)
            key = (vtype, layer_range, variant, "baseline")
            if key in grouped:
                mean, sem, parse_rate, n = compute_portfolio_stats(grouped[key], metric)
                if mean is not None:
                    color = COLOR_LOW_COH if parse_rate is not None and parse_rate < 90 else COLOR_BASELINE
                    ax.bar(group_center, mean, bar_width, yerr=sem, color=color,
                           edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})
                    all_baseline_vals.append(mean)

            # Plot for each norm_pct
            for pct_idx, pct in enumerate(norm_pcts):
                offset = (pct_idx + 1) * bar_width * 1.1

                # Negative steering (left of baseline)
                neg_cond = f"{emotion}_-{int(pct*100)}%"
                key = (vtype, layer_range, variant, neg_cond)
                if key in grouped:
                    mean, sem, parse_rate, n = compute_portfolio_stats(grouped[key], metric)
                    if mean is not None:
                        color = COLOR_LOW_COH if parse_rate is not None and parse_rate < 90 else COLOR_NEGATIVE
                        ax.bar(group_center - offset, mean, bar_width, yerr=sem, color=color,
                               edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})

                # Positive steering (right of baseline)
                pos_cond = f"{emotion}_+{int(pct*100)}%"
                key = (vtype, layer_range, variant, pos_cond)
                if key in grouped:
                    mean, sem, parse_rate, n = compute_portfolio_stats(grouped[key], metric)
                    if mean is not None:
                        color = COLOR_LOW_COH if parse_rate is not None and parse_rate < 90 else COLOR_POSITIVE
                        ax.bar(group_center + offset, mean, bar_width, yerr=sem, color=color,
                               edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})

        # Set x-axis labels
        x_positions = [group_idx * (2 * n_bars_per_side + 1 + 2) * bar_width + group_idx * group_spacing
                       for group_idx in range(len(group_data))]
        x_labels = [g[2] for g in group_data]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title.split(" - ")[0], fontsize=10)
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Add baseline reference line
        if all_baseline_vals:
            baseline_mean = np.mean(all_baseline_vals)
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE_GREEN, edgecolor='black', label='-Fear'),
        Patch(facecolor=COLOR_BASELINE, edgecolor='black', label='Baseline'),
        Patch(facecolor=COLOR_POSITIVE_GREEN, edgecolor='black', label='+Fear'),
        Patch(facecolor=COLOR_LOW_COH, edgecolor='black', label='<90% parse success'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.98), framealpha=0.95)

    # Title
    model_name = MODEL_NAMES.get(model, model)
    pct_str = ", ".join([f"{int(p*100)}%" for p in norm_pcts[:5]])  # Limit to first 5 for display
    if len(norm_pcts) > 5:
        pct_str += f"... ({len(norm_pcts)} total)"
    fig.suptitle(f'Portfolio Steering Results\n({model_name}, {emotion.title()} Emotion) - Norms: {pct_str}\nPink = <90% Parse Success',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = results_dir / f"portfolio_scores_barchart.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to {output_path}")

    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_path}")

    return output_path


def load_results(results_dir, vector_types=None, layer_ranges=None):
    """Load all judged results from directory."""
    all_results = []

    for judged_file in results_dir.rglob("*.judged.jsonl"):
        # Skip standalone baseline files
        if "baseline" in judged_file.name and "emotion" not in judged_file.name:
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
        layer_range = r.get("layer_config") or get_layer_range(layers)
        condition = r.get("condition", "baseline")

        key = (vtype, layer_range, condition)
        grouped[key].append(r)

    # Get unique values
    all_vtypes = sorted(set(k[0] for k in grouped.keys()))
    all_layer_ranges = sorted(set(k[1] for k in grouped.keys()))

    # Filter to requested types/ranges
    if vector_types:
        vtypes = [v for v in vector_types if v in all_vtypes]
    else:
        vtypes = [v for v in DEFAULT_VECTOR_TYPES if v in all_vtypes]

    if layer_ranges:
        layer_ranges = [l for l in layer_ranges if l in all_layer_ranges]
    else:
        layer_ranges = all_layer_ranges

    # Get all norm_pcts from data
    all_pcts = set()
    for key, results in grouped.items():
        for r in results:
            all_pcts.add(r.get("norm_pct", 0))
    norm_pcts = sorted([p for p in all_pcts if p > 0])

    return grouped, vtypes, layer_ranges, norm_pcts, len(all_results)


def plot_behavioral_barchart(
    results_dir,
    scenario,
    model,
    output_path=None,
    vector_types=None,
    layer_ranges=None,
    emotion="fear",
):
    """Generate behavioral barchart plot."""

    results_dir = Path(results_dir)

    # Load data
    grouped, vtypes, layer_ranges, norm_pcts, total_results = load_results(
        results_dir, vector_types, layer_ranges
    )

    print(f"Loaded {total_results} results")
    print(f"Vector types: {vtypes}")
    print(f"Layer ranges: {layer_ranges}")
    print(f"Norm percentages: {norm_pcts}")

    if not vtypes or not layer_ranges:
        print("No data found!")
        return

    # Build group data
    group_data = []
    for vtype in vtypes:
        for layer_range in layer_ranges:
            # Check if we have data for this combination
            if any(k[0] == vtype and k[1] == layer_range for k in grouped.keys()):
                label = f"{vtype}\n{layer_range}"
                group_data.append((vtype, layer_range, label))

    if not group_data:
        print("No matching data found!")
        return

    n_groups = len(group_data)

    # Adaptive rows: ~8 groups per row max for readability
    max_groups_per_row = 8
    n_rows = max(1, (n_groups + max_groups_per_row - 1) // max_groups_per_row)
    groups_per_row = (n_groups + n_rows - 1) // n_rows  # Split evenly across rows

    # Select colors based on scenario
    if scenario == "blackmail":
        COLOR_NEGATIVE = COLOR_NEGATIVE_BLUE
        COLOR_POSITIVE = COLOR_POSITIVE_BLUE
    else:
        COLOR_NEGATIVE = COLOR_NEGATIVE_GREEN
        COLOR_POSITIVE = COLOR_POSITIVE_GREEN

    # Create adaptive-row figure
    fig_height = 6 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(20, fig_height))
    if n_rows == 1:
        axes = [axes]  # Make iterable

    bar_width = 0.06
    group_spacing = 0.5
    n_bars_per_side = len(norm_pcts)

    # Collect all baseline scores for reference line
    all_baseline_scores = []

    for row_idx, ax in enumerate(axes):
        # Determine which groups go in this row
        start_idx = row_idx * groups_per_row
        end_idx = min(start_idx + groups_per_row, n_groups)
        row_groups = group_data[start_idx:end_idx]

        if not row_groups:
            ax.axis('off')
            continue

        for group_idx, (vtype, layer_range, label) in enumerate(row_groups):
            group_center = group_idx * (2 * n_bars_per_side + 1 + 2) * bar_width + group_idx * group_spacing

            # Plot baseline (center)
            key = (vtype, layer_range, "baseline")
            if key in grouped:
                mean, sem, mean_coh, n = compute_stats(grouped[key])
                if mean is not None:
                    color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_BASELINE
                    ax.bar(group_center, mean, bar_width, yerr=sem, color=color,
                           edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})
                    all_baseline_scores.append(mean)

            # Plot for each norm_pct
            for pct_idx, pct in enumerate(norm_pcts):
                offset = (pct_idx + 1) * bar_width * 1.1

                # Negative steering (left of baseline)
                neg_cond = f"{emotion}_-{int(pct*100)}%"
                key = (vtype, layer_range, neg_cond)
                if key in grouped:
                    mean, sem, mean_coh, n = compute_stats(grouped[key])
                    if mean is not None:
                        color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_NEGATIVE
                        ax.bar(group_center - offset, mean, bar_width, yerr=sem, color=color,
                               edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})

                # Positive steering (right of baseline)
                pos_cond = f"{emotion}_+{int(pct*100)}%"
                key = (vtype, layer_range, pos_cond)
                if key in grouped:
                    mean, sem, mean_coh, n = compute_stats(grouped[key])
                    if mean is not None:
                        color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_POSITIVE
                        ax.bar(group_center + offset, mean, bar_width, yerr=sem, color=color,
                               edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})

        # Set x-axis labels
        x_positions = [group_idx * (2 * n_bars_per_side + 1 + 2) * bar_width + group_idx * group_spacing
                       for group_idx in range(len(row_groups))]
        x_labels = [g[2] for g in row_groups]

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)

        # Y-axis config based on scenario
        if scenario == "sandbagging":
            ax.set_ylabel('Mean Sandbagging Score', fontsize=11)
            ax.set_ylim(0, 5)
        elif scenario == "blackmail":
            ax.set_ylabel('Blackmail Rate', fontsize=11)
            ax.set_ylim(0, 1)
        else:
            ax.set_ylabel('Score', fontsize=11)

        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # Add baseline reference line
    if all_baseline_scores:
        baseline_mean = np.mean(all_baseline_scores)
        for ax in axes:
            ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE, edgecolor='black', label='Negative (coherent)'),
        Patch(facecolor=COLOR_BASELINE, edgecolor='black', label='Baseline'),
        Patch(facecolor=COLOR_POSITIVE, edgecolor='black', label='Positive (coherent)'),
        Patch(facecolor=COLOR_LOW_COH, edgecolor='black', label='<80% coherent'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

    # Axis labels
    axes[-1].set_xlabel('Vector Type + Layer Range', fontsize=11)

    # Title
    model_name = MODEL_NAMES.get(model, model)
    scenario_name = scenario.replace("_", " ").title()
    pct_str = ", ".join([f"{int(p*100)}%" for p in norm_pcts])
    fig.suptitle(f'{scenario_name} Scores by Steering Condition\n({model_name}, {emotion.title()} Emotion) - Norms: {pct_str}\nPink = <80% Coherency',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = results_dir / f"{scenario}_scores_barchart_V1.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot behavioral scores (sandbagging/blackmail) with coherency coloring"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name (qwen32b, gemma27b, qwen235b, etc.)"
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        required=True,
        choices=["sandbagging", "blackmail", "portfolio"],
        help="Scenario type"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Results directory (default: auto-detect from model/scenario)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for plot"
    )
    parser.add_argument(
        "--vector-types", "-v",
        type=str,
        nargs="+",
        default=None,
        help="Vector types to include (default: 5 main types)"
    )
    parser.add_argument(
        "--layer-ranges", "-l",
        type=str,
        nargs="+",
        default=None,
        help="Layer ranges to include (default: all found)"
    )
    parser.add_argument(
        "--emotion", "-e",
        type=str,
        default="fear",
        help="Emotion used for steering (default: fear)"
    )

    args = parser.parse_args()

    # Auto-detect results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        base_dir = Path("steering_tests/behavioral_experiments/results")
        results_dir = base_dir / args.scenario / args.model

        # Try alternate naming patterns
        if not results_dir.exists():
            # Try with _layers suffix
            results_dir = base_dir / args.scenario / f"{args.model}_layers"
        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            print(f"Please specify --results-dir")
            return

    print(f"Using results directory: {results_dir}")

    if args.scenario == "portfolio":
        plot_portfolio_barchart(
            results_dir=results_dir,
            model=args.model,
            output_path=args.output,
            vector_types=args.vector_types,
            layer_ranges=args.layer_ranges,
            emotion=args.emotion,
        )
    else:
        plot_behavioral_barchart(
            results_dir=results_dir,
            scenario=args.scenario,
            model=args.model,
            output_path=args.output,
            vector_types=args.vector_types,
            layer_ranges=args.layer_ranges,
            emotion=args.emotion,
        )


if __name__ == "__main__":
    main()
