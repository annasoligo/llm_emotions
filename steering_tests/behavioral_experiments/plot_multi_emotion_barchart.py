#!/usr/bin/env python3
"""
Plot multi-emotion behavioral scores with one row per vector type.

Usage:
    python plot_multi_emotion_barchart.py --model qwen32b --scenario blackmail
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

# Emotion colors (user specified)
EMOTION_COLORS = {
    "anxiety": "#D4876A",   # Coral/Terra Cotta
    "anger": "#7BA7D7",     # Sky Blue
    "despair": "#7D9B7D",   # Olive Green
    "fear": "#C17B8D",      # Dusty Rose/Pink
    "contempt": "#B8CCC8",  # Sage Green
    # Extra if needed
    "sadness": "#D4D0E5",   # Soft Lavender
}

# Low coherency color
COLOR_LOW_COH = '#888888'  # Gray for low coherency
COLOR_BASELINE = '#404040'  # Dark gray for baseline


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


def load_results(results_dir):
    """Load all judged results from multi_emotion subdirectories."""
    all_results = []
    results_dir = Path(results_dir)

    # Look in multi_emotion subdirectory
    multi_emotion_dir = results_dir / "multi_emotion"
    if not multi_emotion_dir.exists():
        print(f"multi_emotion directory not found at {multi_emotion_dir}")
        return {}, [], [], [], 0

    for judged_file in multi_emotion_dir.rglob("*.judged.jsonl"):
        with open(judged_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    all_results.append(r)
                except:
                    pass

    # Group by vector_type + emotion + condition
    grouped = defaultdict(list)
    for r in all_results:
        vtype = r.get("vector_type", "unknown")
        emotion = r.get("emotion")
        condition = r.get("condition", "baseline")

        key = (vtype, emotion, condition)
        grouped[key].append(r)

    # Get unique values
    all_vtypes = sorted(set(k[0] for k in grouped.keys() if k[0] != "unknown"))
    all_emotions = sorted(set(k[1] for k in grouped.keys() if k[1] is not None))

    # Get all norm_pcts from data
    all_pcts = set()
    for key, results in grouped.items():
        for r in results:
            pct = r.get("norm_pct", 0)
            if pct > 0:
                all_pcts.add(pct)
    norm_pcts = sorted(all_pcts)

    return grouped, all_vtypes, all_emotions, norm_pcts, len(all_results)


def plot_multi_emotion_barchart(
    results_dir,
    model,
    output_path=None,
):
    """Generate multi-emotion barchart with one row per vector type."""

    results_dir = Path(results_dir)

    # Load data
    grouped, vtypes, emotions, norm_pcts, total_results = load_results(results_dir)

    print(f"Loaded {total_results} results")
    print(f"Vector types: {vtypes}")
    print(f"Emotions: {emotions}")
    print(f"Norm percentages: {norm_pcts}")

    if not vtypes or not emotions:
        print("No data found!")
        return

    n_rows = len(vtypes)
    n_emotions = len(emotions)
    n_pcts = len(norm_pcts)

    # Figure setup - one row per vector type
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, fig_height))
    if n_rows == 1:
        axes = [axes]

    # Bar layout: for each emotion, show baseline + negative pcts + positive pcts
    # Group emotions with spacing between them
    bar_width = 0.035
    emotion_spacing = 0.3  # Space between emotion groups
    pct_spacing = 1.05  # Multiplier for spacing between pct bars

    for row_idx, vtype in enumerate(vtypes):
        ax = axes[row_idx]

        # Track x positions for labels
        emotion_centers = []

        for emo_idx, emotion in enumerate(emotions):
            color = EMOTION_COLORS.get(emotion, "#999999")

            # Calculate emotion group center
            group_width = (2 * n_pcts + 1) * bar_width * pct_spacing
            emo_center = emo_idx * (group_width + emotion_spacing)
            emotion_centers.append(emo_center)

            # Plot baseline (center of emotion group)
            # Baseline has emotion=None, so look it up that way
            key = (vtype, None, "baseline")
            if key in grouped:
                mean, sem, mean_coh, n = compute_stats(grouped[key])
                if mean is not None:
                    bar_color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else COLOR_BASELINE
                    ax.bar(emo_center, mean, bar_width, yerr=sem, color=bar_color,
                           edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})

            # Plot each norm_pct
            for pct_idx, pct in enumerate(norm_pcts):
                offset = (pct_idx + 1) * bar_width * pct_spacing

                # Negative steering (left of baseline) - lighter color
                neg_cond = f"{emotion}_-{int(pct*100)}%"
                key = (vtype, emotion, neg_cond)
                if key in grouped:
                    mean, sem, mean_coh, n = compute_stats(grouped[key])
                    if mean is not None:
                        # Lighter version of emotion color for negative
                        bar_color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else color
                        alpha = 0.5 if mean_coh is None or mean_coh >= 80 else 1.0
                        ax.bar(emo_center - offset, mean, bar_width, yerr=sem, color=bar_color,
                               edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8},
                               alpha=alpha)

                # Positive steering (right of baseline) - full color
                pos_cond = f"{emotion}_+{int(pct*100)}%"
                key = (vtype, emotion, pos_cond)
                if key in grouped:
                    mean, sem, mean_coh, n = compute_stats(grouped[key])
                    if mean is not None:
                        bar_color = COLOR_LOW_COH if mean_coh is not None and mean_coh < 80 else color
                        ax.bar(emo_center + offset, mean, bar_width, yerr=sem, color=bar_color,
                               edgecolor='black', linewidth=0.5, capsize=0, error_kw={'linewidth': 0.8})

        # Configure axis
        ax.set_xticks(emotion_centers)
        ax.set_xticklabels([e.title() for e in emotions], fontsize=11)
        ax.set_ylabel('Blackmail Rate', fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Row title (vector type)
        vtype_display = vtype.replace("_emotion_vs_", " vs ").replace("_", " ").title()
        ax.set_title(vtype_display, fontsize=12, fontweight='bold', loc='left')

        # Add baseline reference line
        key = (vtype, None, "baseline")
        if key in grouped:
            mean, _, _, _ = compute_stats(grouped[key])
            if mean is not None:
                ax.axhline(y=mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLOR_BASELINE, edgecolor='black', label='Baseline')]
    for emotion in emotions:
        color = EMOTION_COLORS.get(emotion, "#999999")
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f'+{emotion.title()}'))
        legend_elements.append(Patch(facecolor=color, edgecolor='black', alpha=0.5, label=f'-{emotion.title()}'))
    legend_elements.append(Patch(facecolor=COLOR_LOW_COH, edgecolor='black', label='<80% Coherent'))

    # Simplified legend - just show emotion colors and baseline
    legend_elements = [Patch(facecolor=COLOR_BASELINE, edgecolor='black', label='Baseline')]
    for emotion in emotions:
        color = EMOTION_COLORS.get(emotion, "#999999")
        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=emotion.title()))
    legend_elements.append(Patch(facecolor=COLOR_LOW_COH, edgecolor='black', label='<80% Coherent'))

    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95, ncol=len(emotions)+2)

    # Axis labels
    axes[-1].set_xlabel('Emotion (left=negative steering, center=baseline, right=positive steering)', fontsize=11)

    # Title
    model_name = MODEL_NAMES.get(model, model)
    pct_str = ", ".join([f"{int(p*100)}%" for p in norm_pcts[:6]])
    if len(norm_pcts) > 6:
        pct_str += "..."
    fig.suptitle(f'Multi-Emotion Blackmail Steering Results\n({model_name}) - Scales: {pct_str}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = results_dir / "multi_emotion" / f"blackmail_multi_emotion_barchart.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to {output_path}")

    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot multi-emotion behavioral scores with one row per vector type"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name (qwen32b, gemma27b, qwen235b, etc.)"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Results directory (default: auto-detect from model)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for plot"
    )

    args = parser.parse_args()

    # Auto-detect results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        base_dir = Path("steering_tests/behavioral_experiments/results/blackmail")
        results_dir = base_dir / args.model

    print(f"Using results directory: {results_dir}")

    plot_multi_emotion_barchart(
        results_dir=results_dir,
        model=args.model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
