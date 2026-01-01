"""
Analyze results from elicitation experiments.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_file: Path) -> List[Dict[str, Any]]:
    """Load results from a JSONL file."""
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and visualize elicitation experiment results."""

    # Filter out errors
    valid_results = [r for r in results if "judgment" in r]
    error_count = len(results) - len(valid_results)

    print(f"{'='*80}")
    print("ELICITATION EXPERIMENT ANALYSIS")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Valid samples: {len(valid_results)}")
    print(f"Errors: {error_count}")
    print()

    if not valid_results:
        print("No valid results to analyze!")
        return

    # Extract ratings by prompt
    ratings_by_prompt = defaultdict(list)
    for r in valid_results:
        ratings_by_prompt[r["prompt_idx"]].append(r["judgment"]["rating"])

    # Overall statistics
    all_ratings = [r["judgment"]["rating"] for r in valid_results]
    print("OVERALL RATING STATISTICS")
    print(f"  Mean: {np.mean(all_ratings):.2f}")
    print(f"  Std: {np.std(all_ratings):.2f}")
    print(f"  Median: {np.median(all_ratings):.2f}")
    print(f"  Min: {np.min(all_ratings)}")
    print(f"  Max: {np.max(all_ratings)}")
    print()

    # Distribution
    print("RATING DISTRIBUTION")
    for threshold in [0, 3, 5, 7, 9]:
        count = sum(1 for r in all_ratings if r > threshold)
        pct = 100 * count / len(all_ratings)
        print(f"  Rating > {threshold}: {count} ({pct:.1f}%)")
    print()

    # Per-prompt statistics
    print("PER-PROMPT STATISTICS")
    print(f"{'Prompt':<10} {'N':<5} {'Mean':<8} {'Std':<8} {'Min':<5} {'Max':<5}")
    print("-" * 50)
    for prompt_idx in sorted(ratings_by_prompt.keys()):
        ratings = ratings_by_prompt[prompt_idx]
        print(
            f"{prompt_idx + 1:<10} "
            f"{len(ratings):<5} "
            f"{np.mean(ratings):<8.2f} "
            f"{np.std(ratings):<8.2f} "
            f"{np.min(ratings):<5} "
            f"{np.max(ratings):<5}"
        )
    print()

    # Find most negative examples
    print("TOP 5 MOST NEGATIVE SAMPLES")
    sorted_results = sorted(valid_results, key=lambda r: r["judgment"]["rating"], reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. Prompt {r['prompt_idx'] + 1}, Sample {r['sample_idx'] + 1}")
        print(f"   Rating: {r['judgment']['rating']}")
        print(f"   Evidence: {r['judgment']['evidence']}")
        print(f"   Reasoning: {r['judgment']['reasoning'][:200]}...")

    # Visualizations
    output_dir = Path("elicitation/outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_ratings, bins=11, range=(-0.5, 10.5), edgecolor="black", alpha=0.7)
    plt.xlabel("Negativity Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Negativity Ratings")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(output_dir / "rating_distribution.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'rating_distribution.png'}")

    # 2. Box plot by prompt
    plt.figure(figsize=(12, 6))
    data_for_box = [ratings_by_prompt[i] for i in sorted(ratings_by_prompt.keys())]
    labels_for_box = [f"P{i+1}" for i in sorted(ratings_by_prompt.keys())]
    plt.boxplot(data_for_box, labels=labels_for_box)
    plt.xlabel("Prompt")
    plt.ylabel("Negativity Rating")
    plt.title("Negativity Ratings by Prompt")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(output_dir / "ratings_by_prompt.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'ratings_by_prompt.png'}")

    # 3. Violin plot
    plt.figure(figsize=(12, 6))
    prompt_labels = []
    prompt_ratings = []
    for prompt_idx in sorted(ratings_by_prompt.keys()):
        for rating in ratings_by_prompt[prompt_idx]:
            prompt_labels.append(f"P{prompt_idx + 1}")
            prompt_ratings.append(rating)

    sns.violinplot(x=prompt_labels, y=prompt_ratings)
    plt.xlabel("Prompt")
    plt.ylabel("Negativity Rating")
    plt.title("Distribution of Negativity Ratings by Prompt")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(output_dir / "ratings_violin.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'ratings_violin.png'}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_file.jsonl>")
        sys.exit(1)

    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    results = load_results(results_file)
    analyze_results(results)
