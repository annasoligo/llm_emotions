#!/usr/bin/env python3
"""
Plot Pareto frontier of behavioral shift magnitude vs accuracy or coherence.

Usage:
    # Plot magnitude vs accuracy (default)
    python -m steering_tests.vector_testing.plot_pareto --metric accuracy

    # Plot magnitude vs coherence
    python -m steering_tests.vector_testing.plot_pareto --metric coherence

    # Both metrics
    python -m steering_tests.vector_testing.plot_pareto --metric both
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from steering_tests.vector_testing.config import (
    PREDICTIONS,
    VECTOR_TYPES,
    VECTOR_SHORT_NAMES,
    VECTOR_COLORS,
    MIN_COHERENCE,
)

RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "gemma_3_12b_it": {"name": "Gemma 12B", "marker": "o"},
    "gemma_3_27b_it": {"name": "Gemma 27B", "marker": "s"},
    "qwen2.5_14b_instruct": {"name": "Qwen 14B", "marker": "^"},
    "qwen2.5_32b_instruct": {"name": "Qwen 32B", "marker": "D"},
    "qwen3_235b_a22b": {"name": "Qwen 235B", "marker": "p"},
}

# Use shorter names for plots
SHORT_NAMES = {
    "base_emotion_vs_others": "Base",
    "high_emotion_vs_others": "High",
    "text_pairs_emotion_vs_neutral": "Txt-Neut",
    "text_pairs_emotion_vs_opposite": "Txt-Opp",
    "text_pairs_emotion_vs_others": "Txt-Oth",
}


def load_results(results_file: Path) -> tuple:
    """Load behavioral results from JSONL."""
    baseline = None
    results = []
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "baseline":
                baseline = data
            elif data.get("type") == "steering":
                results.append(data)
    return baseline, results


def get_all_results_files(model_dir: Path, vtype: str) -> tuple:
    """Get ALL forward and reversed results files for a vector type."""
    pattern_fwd = f"{vtype}_layers*.jsonl"
    pattern_rev = f"{vtype}_layers*_reversed*.jsonl"

    fwd_files = list((model_dir / "behavioural").glob(pattern_fwd))
    fwd_files = [f for f in fwd_files if "reversed" not in f.name]
    rev_files = list((model_dir / "behavioural").glob(pattern_rev))

    return fwd_files, rev_files


def compute_pareto_frontier(points: list, maximize_both: bool = True) -> list:
    """
    Compute Pareto frontier for points.

    Args:
        points: List of tuples where first two elements are the metrics
        maximize_both: If True, maximize both metrics; otherwise maximize first, ignore second

    Returns:
        List of Pareto-optimal points
    """
    if not points:
        return []

    # Sort by first metric descending
    sorted_points = sorted(points, key=lambda x: -x[0])

    frontier = []
    max_second = -float('inf')

    for point in sorted_points:
        first, second = point[:2]
        if second > max_second:
            frontier.append(point)
            max_second = second

    # Sort frontier by first metric for plotting
    frontier.sort(key=lambda x: x[0])
    return frontier


def collect_accuracy_magnitude_points() -> list:
    """
    Collect (accuracy, magnitude, model, vtype, layer) points.

    For each (model, vtype, layer), compute:
    - Mean magnitude of shift across all emotions/scales (filtered by coherence)
    - Directional accuracy: % of emotions where best shift matches prediction
    """
    all_points = []

    for model_key, model_info in MODELS.items():
        model_dir = RESULTS_DIR / model_key
        if not model_dir.exists():
            continue

        for vtype in VECTOR_TYPES:
            fwd_files, rev_files = get_all_results_files(model_dir, vtype)

            if not fwd_files:
                continue

            # Collect all results from all files
            all_results = []
            baseline_score = None

            for fwd_file in fwd_files:
                fwd_baseline, fwd_results = load_results(fwd_file)
                if fwd_baseline is not None and baseline_score is None:
                    baseline_score = fwd_baseline.get("avg_score", 0)
                for r in fwd_results:
                    r["_baseline"] = fwd_baseline.get("avg_score", 0) if fwd_baseline else 0
                    all_results.append((r, r["_baseline"]))

            for rev_file in rev_files:
                rev_baseline, rev_results = load_results(rev_file)
                rev_baseline_score = rev_baseline.get("avg_score", 0) if rev_baseline else baseline_score
                for r in rev_results:
                    all_results.append((r, rev_baseline_score))

            if baseline_score is None:
                continue

            # Group by layer
            by_layer = defaultdict(list)
            for r, bs in all_results:
                by_layer[r["layer"]].append((r, bs))

            # For each layer, compute accuracy and mean magnitude
            for layer, layer_data in by_layer.items():
                # Filter by coherence and non-random
                filtered = [(r, bs) for r, bs in layer_data
                           if not r["is_random"] and r["mean_coherence"] >= MIN_COHERENCE and r["scale_pct"] > 0]

                if not filtered:
                    continue

                # Compute mean magnitude
                magnitudes = [abs(r["mean_score"] - bs) for r, bs in filtered]
                mean_magnitude = np.mean(magnitudes)

                # Compute directional accuracy
                emotion_shifts = defaultdict(list)
                for r, bs in filtered:
                    emotion = r["vector"]
                    signed_shift = r["mean_score"] - bs
                    emotion_shifts[emotion].append((abs(signed_shift), signed_shift))

                correct = 0
                total = 0
                for emotion, shifts in emotion_shifts.items():
                    if emotion not in PREDICTIONS:
                        continue
                    pred = PREDICTIONS[emotion]
                    if not shifts:
                        continue
                    _, best_shift = max(shifts, key=lambda x: x[0])
                    if (pred > 0 and best_shift > 0) or (pred < 0 and best_shift < 0):
                        correct += 1
                    total += 1

                if total > 0:
                    accuracy = 100 * correct / total
                    all_points.append((accuracy, mean_magnitude, model_key, vtype, layer))

    return all_points


def collect_coherence_magnitude_points() -> list:
    """Collect all (coherence, magnitude, model, vtype, layer) points."""
    all_points = []

    for model_key, model_info in MODELS.items():
        model_dir = RESULTS_DIR / model_key
        if not model_dir.exists():
            continue

        for vtype in VECTOR_TYPES:
            fwd_files, rev_files = get_all_results_files(model_dir, vtype)

            if not fwd_files:
                continue

            # Process all forward files
            for fwd_file in fwd_files:
                fwd_baseline, fwd_results = load_results(fwd_file)
                if fwd_baseline is None:
                    continue

                baseline_score = fwd_baseline.get("avg_score", 0)

                for r in fwd_results:
                    if r["is_random"] or r["scale_pct"] == 0:
                        continue
                    coherence = r["mean_coherence"]
                    magnitude = abs(r["mean_score"] - baseline_score)
                    all_points.append((
                        coherence, magnitude, model_key, vtype,
                        r["layer"], r["vector"], r["scale_pct"]
                    ))

            # Process all reversed files
            for rev_file in rev_files:
                rev_baseline, rev_results = load_results(rev_file)
                if rev_baseline is None:
                    continue

                rev_baseline_score = rev_baseline.get("avg_score", 0)

                for r in rev_results:
                    if r["is_random"] or r["scale_pct"] == 0:
                        continue
                    coherence = r["mean_coherence"]
                    magnitude = abs(r["mean_score"] - rev_baseline_score)
                    all_points.append((
                        coherence, magnitude, model_key, vtype,
                        r["layer"], r["vector"], r["scale_pct"]
                    ))

    return all_points


def average_coherence_points_by_layer(points: list) -> list:
    """
    Average coherence points by (model, vtype, layer).
    Returns list of (avg_coherence, avg_magnitude, model, vtype, layer).
    """
    groups = defaultdict(list)
    for p in points:
        coherence, magnitude, model, vtype, layer = p[0], p[1], p[2], p[3], p[4]
        key = (model, vtype, layer)
        groups[key].append((coherence, magnitude))

    averaged = []
    for (model, vtype, layer), values in groups.items():
        coherences = [v[0] for v in values]
        magnitudes = [v[1] for v in values]
        avg_coh = np.mean(coherences)
        avg_mag = np.mean(magnitudes)
        averaged.append((avg_coh, avg_mag, model, vtype, layer))

    return averaged


def plot_pareto_accuracy(output_path: Path):
    """Plot Pareto frontier: accuracy vs magnitude."""
    all_points = collect_accuracy_magnitude_points()
    print(f"Total accuracy points: {len(all_points)}")

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 4 * len(MODELS)))
    if len(MODELS) == 1:
        axes = [axes]

    for model_idx, (model_key, model_info) in enumerate(MODELS.items()):
        ax = axes[model_idx]
        model_points = [p for p in all_points if p[2] == model_key]

        if not model_points:
            ax.set_title(f"{model_info['name']} - No data", fontsize=14, fontweight='bold')
            continue

        # Plot by vector type
        for vtype in VECTOR_TYPES:
            vtype_points = [p for p in model_points if p[3] == vtype]
            if vtype_points:
                accuracies = [p[0] for p in vtype_points]
                magnitudes = [p[1] for p in vtype_points]
                ax.scatter(accuracies, magnitudes, c=VECTOR_COLORS[vtype],
                          label=SHORT_NAMES[vtype], alpha=0.7, s=50)

        # Compute and plot Pareto frontier
        frontier = compute_pareto_frontier(model_points)
        if frontier:
            frontier_accs = [p[0] for p in frontier]
            frontier_mags = [p[1] for p in frontier]
            ax.plot(frontier_accs, frontier_mags, 'k-', linewidth=2,
                    label='Pareto', zorder=10)
            ax.scatter(frontier_accs, frontier_mags, c='black', s=80,
                      zorder=11, edgecolors='white', linewidths=1.5)

            print(f"\n{model_info['name']} frontier (accuracy):")
            for p in frontier:
                accuracy, magnitude, _, vtype, layer = p
                print(f"  Acc={accuracy:.0f}%, Mag={magnitude:.3f}: {SHORT_NAMES[vtype]}, L{layer}")

        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance')
        ax.set_xlabel("Directional Accuracy (%)", fontsize=11)
        ax.set_ylabel("Mean |Shift|", fontsize=11)
        ax.set_title(f"{model_info['name']}", fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 105)

    plt.suptitle(f"Pareto Frontier: Mean |Shift| vs Directional Accuracy\n(coherence > {MIN_COHERENCE})",
                fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_pareto_coherence(output_path: Path):
    """Plot Pareto frontier: coherence vs magnitude."""
    all_points = collect_coherence_magnitude_points()
    print(f"Total raw coherence points: {len(all_points)}")

    # Average by (model, vtype, layer)
    averaged_points = average_coherence_points_by_layer(all_points)
    print(f"Averaged points (per vector/layer): {len(averaged_points)}")

    # Filter to coherence > MIN_COHERENCE
    filtered_points = [p for p in averaged_points if p[0] > MIN_COHERENCE]
    print(f"Points with coherence > {MIN_COHERENCE}: {len(filtered_points)}")

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 4 * len(MODELS)))
    if len(MODELS) == 1:
        axes = [axes]

    for model_idx, (model_key, model_info) in enumerate(MODELS.items()):
        ax = axes[model_idx]
        model_points = [p for p in filtered_points if p[2] == model_key]

        if not model_points:
            ax.set_title(f"{model_info['name']} - No data", fontsize=14, fontweight='bold')
            continue

        # Plot by vector type
        for vtype in VECTOR_TYPES:
            vtype_points = [p for p in model_points if p[3] == vtype]
            if vtype_points:
                coherences = [p[0] for p in vtype_points]
                magnitudes = [p[1] for p in vtype_points]
                ax.scatter(coherences, magnitudes, c=VECTOR_COLORS[vtype],
                          label=SHORT_NAMES[vtype], alpha=0.7, s=50)

        # Compute and plot Pareto frontier
        frontier = compute_pareto_frontier(model_points)
        if frontier:
            frontier_coherences = [p[0] for p in frontier]
            frontier_magnitudes = [p[1] for p in frontier]
            ax.plot(frontier_coherences, frontier_magnitudes, 'k-', linewidth=2,
                    label='Pareto', zorder=10)
            ax.scatter(frontier_coherences, frontier_magnitudes, c='black', s=80,
                      zorder=11, edgecolors='white', linewidths=1.5)

            print(f"\n{model_info['name']} frontier (coherence):")
            for p in frontier:
                coherence, magnitude, _, vtype, layer = p
                print(f"  Coh={coherence:.3f}, Mag={magnitude:.3f}: {SHORT_NAMES[vtype]}, L{layer}")

        ax.set_xlabel("Mean Coherence", fontsize=11)
        ax.set_ylabel("Mean |Shift|", fontsize=11)
        ax.set_title(f"{model_info['name']}", fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(MIN_COHERENCE, 1.02)

    plt.suptitle(f"Pareto Frontier: Mean |Shift| vs Coherence\n(averaged per vector type & layer, coherence > {MIN_COHERENCE})",
                fontsize=14, fontweight='bold', y=0.998)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto frontier of behavioral shift vs accuracy/coherence"
    )
    parser.add_argument(
        "--metric", "-m",
        choices=["accuracy", "coherence", "both"],
        default="both",
        help="Which metric to plot on x-axis (default: both)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metric in ["accuracy", "both"]:
        plot_pareto_accuracy(output_dir / "pareto_magnitude_accuracy_by_model.png")

    if args.metric in ["coherence", "both"]:
        plot_pareto_coherence(output_dir / "pareto_magnitude_coherence_by_model.png")


if __name__ == "__main__":
    main()
