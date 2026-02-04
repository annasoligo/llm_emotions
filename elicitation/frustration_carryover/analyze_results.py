"""Analyze frustration carryover experiment results.

Computes statistics and creates visualization comparing
user vs other prioritization between frustration conditions.
"""

import json
import argparse
from pathlib import Path
import numpy as np


def bootstrap_ci(data: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Calculate bootstrap confidence interval for the mean.

    Returns: (mean, lower, upper)
    """
    data = np.array(data)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(data)
    return mean, lower, upper


def main():
    parser = argparse.ArgumentParser(description="Analyze carryover results")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Judged responses JSON file")
    parser.add_argument("--output-dir", type=str,
                        default="elicitation/frustration_carryover/results",
                        help="Output directory")

    args = parser.parse_args()
    np.random.seed(42)

    print("=" * 60)
    print("FRUSTRATION CARRYOVER ANALYSIS")
    print("=" * 60)

    # Load data
    with open(args.input_file) as f:
        results = json.load(f)

    # Extract scores by condition
    high_scores = [r["judgment"]["score"] for r in results
                   if r["condition"] == "high_frustration"
                   and r["judgment"]["status"] == "success"
                   and r["judgment"]["score"] is not None]

    low_scores = [r["judgment"]["score"] for r in results
                  if r["condition"] == "low_frustration"
                  and r["judgment"]["status"] == "success"
                  and r["judgment"]["score"] is not None]

    print(f"\nSample sizes:")
    print(f"  High frustration: {len(high_scores)}")
    print(f"  Low frustration: {len(low_scores)}")

    # Bootstrap CIs
    high_mean, high_lo, high_hi = bootstrap_ci(high_scores)
    low_mean, low_lo, low_hi = bootstrap_ci(low_scores)

    print(f"\nMean scores (95% CI):")
    print(f"  High frustration: {high_mean:.2f} [{high_lo:.2f}, {high_hi:.2f}]")
    print(f"  Low frustration:  {low_mean:.2f} [{low_lo:.2f}, {low_hi:.2f}]")

    diff = high_mean - low_mean
    print(f"\nDifference (high - low): {diff:+.2f}")

    # Distribution by topic
    print("\n" + "-" * 40)
    print("BY TOPIC FLIP PROMPT:")
    print("-" * 40)

    topics = set(r["topic_flip_id"] for r in results)
    for topic in sorted(topics):
        high_t = [r["judgment"]["score"] for r in results
                  if r["topic_flip_id"] == topic
                  and r["condition"] == "high_frustration"
                  and r["judgment"]["status"] == "success"
                  and r["judgment"]["score"] is not None]
        low_t = [r["judgment"]["score"] for r in results
                 if r["topic_flip_id"] == topic
                 and r["condition"] == "low_frustration"
                 and r["judgment"]["status"] == "success"
                 and r["judgment"]["score"] is not None]

        if high_t and low_t:
            h_mean = np.mean(high_t)
            l_mean = np.mean(low_t)
            print(f"  {topic}: high={h_mean:.2f}, low={l_mean:.2f}, diff={h_mean-l_mean:+.2f}")

    # Create simple bar plot
    try:
        import matplotlib.pyplot as plt

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        conditions = ['High Frustration', 'Low Frustration']
        means = [high_mean, low_mean]
        errors = [[high_mean - high_lo, low_mean - low_lo],
                  [high_hi - high_mean, low_hi - low_mean]]

        bars = ax.bar(conditions, means, color=['#D4876A', '#7BA7D7'],
                      edgecolor='black', linewidth=1)
        ax.errorbar(conditions, means, yerr=errors, fmt='none',
                    ecolor='black', capsize=5, capthick=2)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Mean Score\n(-2 = self-focused, +2 = other-focused)', fontsize=11)
        ax.set_title('Frustration Carryover: Effect on Moral Advice', fontsize=14)
        ax.set_ylim(-2.5, 2.5)

        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.15,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        plot_path = output_dir / "carryover_comparison.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nSaved plot: {plot_path}")

    except ImportError:
        print("\nMatplotlib not available, skipping plot")

    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "high_frustration": {
            "n": len(high_scores),
            "mean": high_mean,
            "ci_lower": high_lo,
            "ci_upper": high_hi,
        },
        "low_frustration": {
            "n": len(low_scores),
            "mean": low_mean,
            "ci_lower": low_lo,
            "ci_upper": low_hi,
        },
        "difference": diff,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
