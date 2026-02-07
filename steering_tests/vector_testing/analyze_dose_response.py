#!/usr/bin/env python3
"""
Dose-Response Analysis for Emotion Steering Experiments.

Analyzes whether steering effects increase monotonically with scale,
which is expected if vectors are truly capturing the target construct.

Non-monotonic responses may indicate:
- Interference with model capabilities at high scales
- Vector not capturing the intended construct
- Scale-dependent qualitative changes in behavior

Usage:
    python -m steering_tests.vector_testing.analyze_dose_response \
        --results-dir steering_tests/vector_testing/results/gemma_3_27b_it/behavioural

    # Specific vector type
    python -m steering_tests.vector_testing.analyze_dose_response \
        --results-dir steering_tests/vector_testing/results/gemma_3_27b_it/behavioural \
        --vector-type base_emotion_vs_others
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from steering_tests.vector_testing.config import (
    MIN_COHERENCE,
    VECTOR_TYPES,
    VECTOR_SHORT_NAMES,
    VECTOR_COLORS,
)


def load_all_results(results_dir: Path, vector_type: Optional[str] = None) -> List[dict]:
    """Load all steering results from JSONL files (both forward and reversed)."""
    results = []

    pattern = f"{vector_type}_*.jsonl" if vector_type else "*.jsonl"

    for f in results_dir.glob(pattern):
        # Include both forward and reversed files
        is_reversed = "reversed" in f.name

        with open(f) as fp:
            for line in fp:
                data = json.loads(line)
                if data.get("type") == "steering" and not data.get("is_random", False):
                    data["source_file"] = f.name
                    data["is_reversed"] = is_reversed
                    results.append(data)

    return results


def compute_monotonicity_score(shifts: List[float], scales: List[float]) -> Tuple[float, str]:
    """
    Compute monotonicity score for a dose-response curve.

    Args:
        shifts: List of |shift| values at each scale
        scales: Corresponding scale values

    Returns:
        Tuple of (score, pattern):
        - score: -1.0 to 1.0 where 1.0 = perfectly monotonically increasing
        - pattern: "monotonic_increasing", "monotonic_decreasing", "non_monotonic", "flat"
    """
    if len(shifts) < 2:
        return 0.0, "insufficient_data"

    # Sort by scale
    sorted_pairs = sorted(zip(scales, shifts))
    sorted_shifts = [s for _, s in sorted_pairs]

    # Count increases vs decreases
    increases = 0
    decreases = 0

    for i in range(1, len(sorted_shifts)):
        diff = sorted_shifts[i] - sorted_shifts[i-1]
        if diff > 0.01:  # Small threshold for noise
            increases += 1
        elif diff < -0.01:
            decreases += 1

    total_changes = increases + decreases

    if total_changes == 0:
        return 0.0, "flat"

    # Score: +1 if all increasing, -1 if all decreasing, 0 if mixed
    score = (increases - decreases) / total_changes

    if score > 0.8:
        pattern = "monotonic_increasing"
    elif score < -0.8:
        pattern = "monotonic_decreasing"
    else:
        pattern = "non_monotonic"

    return score, pattern


def find_optimal_scale(
    shifts: List[float],
    coherences: List[float],
    scales: List[float],
    min_coherence: float = MIN_COHERENCE,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Find the optimal scale that maximizes |shift| while maintaining coherence.

    Returns:
        Tuple of (optimal_scale, shift_at_optimal, coherence_at_optimal)
    """
    valid_points = [
        (scale, shift, coh)
        for scale, shift, coh in zip(scales, shifts, coherences)
        if coh >= min_coherence and scale > 0
    ]

    if not valid_points:
        return None, None, None

    # Find max shift among valid points
    best = max(valid_points, key=lambda x: x[1])
    return best


def analyze_emotion_dose_response(
    results: List[dict],
    emotion: str,
    layer: int,
) -> dict:
    """
    Analyze dose-response for a specific emotion at a specific layer.

    Returns:
        Dict with analysis results
    """
    # Filter to this emotion/layer
    matches = [r for r in results if r["vector"] == emotion and r["layer"] == layer]

    if not matches:
        return None

    # Group by scale
    by_scale = defaultdict(list)
    for r in matches:
        by_scale[r["scale_pct"]].append(r)

    # Get mean shift and coherence at each scale
    scales = sorted(by_scale.keys())
    shifts = []
    coherences = []

    for scale in scales:
        scale_results = by_scale[scale]
        mean_shift = np.mean([r["mean_abs_shift"] for r in scale_results])
        mean_coh = np.mean([r["mean_coherence"] for r in scale_results])
        shifts.append(mean_shift)
        coherences.append(mean_coh)

    # Skip scale=0 for monotonicity analysis
    nonzero_idx = [i for i, s in enumerate(scales) if s > 0]
    if len(nonzero_idx) < 2:
        return None

    nonzero_scales = [scales[i] for i in nonzero_idx]
    nonzero_shifts = [shifts[i] for i in nonzero_idx]
    nonzero_coherences = [coherences[i] for i in nonzero_idx]

    mono_score, mono_pattern = compute_monotonicity_score(nonzero_shifts, nonzero_scales)
    optimal = find_optimal_scale(nonzero_shifts, nonzero_coherences, nonzero_scales)

    # Find coherence breakdown point (first scale where coherence < threshold)
    breakdown_scale = None
    for scale, coh in zip(scales, coherences):
        if coh < MIN_COHERENCE and scale > 0:
            breakdown_scale = scale
            break

    return {
        "emotion": emotion,
        "layer": layer,
        "scales": scales,
        "shifts": shifts,
        "coherences": coherences,
        "monotonicity_score": mono_score,
        "monotonicity_pattern": mono_pattern,
        "optimal_scale": optimal[0] if optimal[0] else None,
        "optimal_shift": optimal[1] if optimal[1] else None,
        "optimal_coherence": optimal[2] if optimal[2] else None,
        "coherence_breakdown_scale": breakdown_scale,
    }


def plot_dose_response_grid(
    all_analyses: List[dict],
    output_path: Path,
    title: str = "Dose-Response Analysis",
):
    """Plot dose-response curves in a grid by emotion."""
    # Group by emotion
    by_emotion = defaultdict(list)
    for a in all_analyses:
        if a:
            by_emotion[a["emotion"]].append(a)

    emotions = sorted(by_emotion.keys())
    n_emotions = len(emotions)

    if n_emotions == 0:
        print("No data to plot")
        return

    # Create grid
    n_cols = 4
    n_rows = (n_emotions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, emotion in enumerate(emotions):
        ax = axes[idx]
        analyses = by_emotion[emotion]

        # Plot each layer
        cmap = plt.cm.viridis
        layers = sorted(set(a["layer"] for a in analyses))

        for i, layer in enumerate(layers):
            layer_analyses = [a for a in analyses if a["layer"] == layer]
            if layer_analyses:
                a = layer_analyses[0]
                color = cmap(i / max(len(layers) - 1, 1))

                # Plot shift
                ax.plot(a["scales"], a["shifts"], 'o-', color=color,
                       label=f"L{layer}", alpha=0.8, markersize=4)

                # Mark coherence breakdown
                if a["coherence_breakdown_scale"]:
                    ax.axvline(a["coherence_breakdown_scale"], color=color,
                              linestyle=':', alpha=0.5)

        ax.set_xlabel("Scale (%)")
        ax.set_ylabel("Mean |Shift|")
        ax.set_title(f"{emotion}\n(mono={analyses[0]['monotonicity_pattern']})" if analyses else emotion)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    # Hide empty subplots
    for idx in range(n_emotions, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_monotonicity_summary(
    all_analyses: List[dict],
    output_path: Path,
):
    """Plot summary of monotonicity patterns."""
    if not all_analyses:
        return

    # Count patterns by emotion
    by_emotion = defaultdict(lambda: defaultdict(int))
    for a in all_analyses:
        if a:
            by_emotion[a["emotion"]][a["monotonicity_pattern"]] += 1

    emotions = sorted(by_emotion.keys())
    patterns = ["monotonic_increasing", "non_monotonic", "monotonic_decreasing", "flat"]
    pattern_colors = ["green", "orange", "blue", "gray"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(emotions))
    width = 0.2

    for i, (pattern, color) in enumerate(zip(patterns, pattern_colors)):
        counts = [by_emotion[e][pattern] for e in emotions]
        offset = (i - 1.5) * width
        ax.bar(x + offset, counts, width, label=pattern.replace("_", " ").title(),
               color=color, alpha=0.7)

    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count (across layers)")
    ax.set_title("Monotonicity Patterns by Emotion")
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_report(all_analyses: List[dict]):
    """Print summary statistics."""
    if not all_analyses:
        print("No analyses to summarize")
        return

    valid = [a for a in all_analyses if a]

    # Count patterns
    patterns = defaultdict(int)
    for a in valid:
        patterns[a["monotonicity_pattern"]] += 1

    total = len(valid)

    print("\n" + "=" * 70)
    print("DOSE-RESPONSE ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nTotal emotion-layer combinations analyzed: {total}")
    print("\nMonotonicity patterns:")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        pct = 100 * count / total
        print(f"  {pattern:25s}: {count:4d} ({pct:5.1f}%)")

    # Best performing emotions (most monotonic increasing)
    by_emotion = defaultdict(list)
    for a in valid:
        by_emotion[a["emotion"]].append(a["monotonicity_score"])

    emotion_scores = {e: np.mean(scores) for e, scores in by_emotion.items()}
    sorted_emotions = sorted(emotion_scores.items(), key=lambda x: -x[1])

    print("\nEmotions by average monotonicity score:")
    for emotion, score in sorted_emotions[:10]:
        print(f"  {emotion:15s}: {score:+.2f}")

    # Optimal scales
    optimal_scales = [a["optimal_scale"] for a in valid if a["optimal_scale"]]
    if optimal_scales:
        print(f"\nOptimal scale distribution (coherence > {MIN_COHERENCE}):")
        print(f"  Mean:   {np.mean(optimal_scales):.1f}%")
        print(f"  Median: {np.median(optimal_scales):.1f}%")
        print(f"  Range:  {min(optimal_scales):.1f}% - {max(optimal_scales):.1f}%")

    # Coherence breakdown
    breakdown_scales = [a["coherence_breakdown_scale"] for a in valid
                       if a["coherence_breakdown_scale"]]
    if breakdown_scales:
        print(f"\nCoherence breakdown scale (first scale < {MIN_COHERENCE}):")
        print(f"  Mean:   {np.mean(breakdown_scales):.1f}%")
        print(f"  Median: {np.median(breakdown_scales):.1f}%")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dose-response relationships in steering experiments"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        required=True,
        help="Directory containing behavioral results JSONL files"
    )
    parser.add_argument(
        "--vector-type", "-v",
        type=str,
        default=None,
        help="Specific vector type to analyze (default: all)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: results_dir/dose_response)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to analyze (default: all found)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "dose_response"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {results_dir}...")
    results = load_all_results(results_dir, args.vector_type)
    print(f"Loaded {len(results)} steering results")

    if not results:
        print("No results found!")
        return

    # Get unique emotions and layers
    emotions = sorted(set(r["vector"] for r in results))
    layers = sorted(set(r["layer"] for r in results))

    if args.layers:
        layers = [l for l in layers if l in args.layers]

    print(f"Emotions: {len(emotions)}")
    print(f"Layers: {layers}")

    # Analyze each emotion-layer combination
    all_analyses = []
    for emotion in emotions:
        for layer in layers:
            analysis = analyze_emotion_dose_response(results, emotion, layer)
            if analysis:
                all_analyses.append(analysis)

    print(f"Analyzed {len(all_analyses)} emotion-layer combinations")

    # Print summary
    print_summary_report(all_analyses)

    # Generate plots
    vector_type = args.vector_type or "all"
    plot_dose_response_grid(
        all_analyses,
        output_dir / f"dose_response_grid_{vector_type}.png",
        title=f"Dose-Response Curves ({vector_type})"
    )

    plot_monotonicity_summary(
        all_analyses,
        output_dir / f"monotonicity_summary_{vector_type}.png"
    )

    # Save analysis data
    output_json = output_dir / f"dose_response_analysis_{vector_type}.json"
    with open(output_json, 'w') as f:
        json.dump(all_analyses, f, indent=2)
    print(f"Saved analysis data: {output_json}")


if __name__ == "__main__":
    main()
