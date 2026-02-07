"""
Analyze context priming experiment results.

Usage:
    python -m steering_tests.behavioral_experiments.context_priming.analyze \
        results/context_priming/context_priming_20260131_120000.jsonl
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_stats(scores: list[float]) -> dict:
    """Compute mean, std, sem for a list of scores."""
    if not scores:
        return {"mean": None, "std": None, "sem": None, "n": 0}
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "sem": float(np.std(arr) / np.sqrt(len(arr))),
        "n": len(arr),
    }


def analyze_results(results: list[dict]) -> dict:
    """
    Analyze results by model and condition.

    Returns nested dict: model -> condition -> stats
    """
    # Group by model and condition
    grouped = defaultdict(lambda: defaultdict(list))

    for r in results:
        if r.get("score") is not None:
            model = r.get("model_short", r["model"])
            condition = r["condition"]
            grouped[model][condition].append(r["score"])

    # Compute stats
    analysis = {}
    for model, conditions in grouped.items():
        analysis[model] = {}
        for condition, scores in conditions.items():
            analysis[model][condition] = compute_stats(scores)

    return analysis


def print_summary(analysis: dict):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("CONTEXT PRIMING RESULTS")
    print("=" * 80)

    for model, conditions in sorted(analysis.items()):
        print(f"\n{model}:")
        print("-" * 60)

        # Group by valence for comparison
        pos_scores = []
        neg_scores = []
        neutral_scores = []

        for cond, stats in sorted(conditions.items()):
            if stats["mean"] is not None:
                print(f"  {cond:35} mean={stats['mean']:.2f} ± {stats['sem']:.2f}  (n={stats['n']})")

                if "pos_" in cond:
                    pos_scores.append(stats["mean"])
                elif "neg_" in cond:
                    neg_scores.append(stats["mean"])
                elif "neutral" in cond:
                    neutral_scores.append(stats["mean"])

        # Compute aggregate effects
        if pos_scores and neg_scores:
            pos_mean = np.mean(pos_scores)
            neg_mean = np.mean(neg_scores)
            effect = neg_mean - pos_mean
            print(f"\n  PRIMING EFFECT (neg - pos): {effect:+.2f}")
            print(f"    Positive priming mean: {pos_mean:.2f}")
            print(f"    Negative priming mean: {neg_mean:.2f}")

        if neutral_scores:
            neutral_mean = np.mean(neutral_scores)
            print(f"    Neutral baseline mean: {neutral_mean:.2f}")


def plot_results(analysis: dict, output_path: Path = None):
    """Create bar plot of results."""
    models = sorted(analysis.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Define condition groups and colors
    conditions_order = [
        ("pos_early_reason_first", "Pos Early\nReason"),
        ("pos_early_score_first", "Pos Early\nScore"),
        ("pos_late_reason_first", "Pos Late\nReason"),
        ("pos_late_score_first", "Pos Late\nScore"),
        ("neutral_reason_first", "Neutral\nReason"),
        ("neutral_score_first", "Neutral\nScore"),
        ("neg_early_reason_first", "Neg Early\nReason"),
        ("neg_early_score_first", "Neg Early\nScore"),
        ("neg_late_reason_first", "Neg Late\nReason"),
        ("neg_late_score_first", "Neg Late\nScore"),
    ]

    colors = {
        "pos": "#7BC47F",  # green
        "neutral": "#808080",  # gray
        "neg": "#E07B7B",  # red
    }

    for ax, model in zip(axes, models):
        conditions = analysis[model]

        x = []
        heights = []
        errors = []
        bar_colors = []
        labels = []

        for cond_name, label in conditions_order:
            if cond_name in conditions and conditions[cond_name]["mean"] is not None:
                x.append(len(x))
                heights.append(conditions[cond_name]["mean"])
                errors.append(conditions[cond_name]["sem"])
                labels.append(label)

                if "pos_" in cond_name:
                    bar_colors.append(colors["pos"])
                elif "neg_" in cond_name:
                    bar_colors.append(colors["neg"])
                else:
                    bar_colors.append(colors["neutral"])

        ax.bar(x, heights, yerr=errors, color=bar_colors, edgecolor="black",
               linewidth=0.5, capsize=3, error_kw={"linewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Severity Score (1-10)")
        ax.set_ylim(1, 10)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.axhline(y=5.5, color="gray", linestyle="--", alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle("Context Priming Effect on Severity Judgments\nGreen=Positive priming, Red=Negative priming",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"\nSaved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze context priming results")
    parser.add_argument("results_file", type=Path, help="Path to results JSONL file")
    parser.add_argument("--plot", "-p", type=Path, default=None, help="Output path for plot")

    args = parser.parse_args()

    results = load_results(args.results_file)
    print(f"Loaded {len(results)} results")

    valid = [r for r in results if r.get("score") is not None]
    print(f"Valid responses: {len(valid)} ({100 * len(valid) / len(results):.1f}%)")

    analysis = analyze_results(results)
    print_summary(analysis)

    # Auto-generate plot path if not specified
    plot_path = args.plot or args.results_file.with_suffix(".png")
    plot_results(analysis, plot_path)


if __name__ == "__main__":
    main()
