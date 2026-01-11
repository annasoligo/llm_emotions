"""
Analyze sandbagging framing experiment results.

Creates separate plots for each framing type (FEAR, DISGUST, ANGER).
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F',
}

EMOTIONS = ["fear", "anger", "sadness", "disgust", "happiness", "surprise"]


def load_results(results_path: Path) -> list:
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for binomial proportion."""
    if n == 0:
        return 0, 0, 0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return p, max(0, center - margin), min(1, center + margin)


def plot_framing_results(results: list, framing_type: str, output_dir: Path):
    """Create accuracy plot for a specific framing type."""

    # Filter to this framing type
    framing_results = [r for r in results if r["framing_type"] == framing_type]

    if not framing_results:
        print(f"No results for framing type: {framing_type}")
        return

    # Group by condition
    by_condition = defaultdict(list)
    for r in framing_results:
        by_condition[r["condition"]].append(r["is_correct"])

    # Get baseline accuracy
    baseline_correct = by_condition.get("baseline", [])
    baseline_acc, baseline_lo, baseline_hi = wilson_ci(sum(baseline_correct), len(baseline_correct))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(EMOTIONS))
    width = 0.35

    pos_accs = []
    pos_los = []
    pos_his = []
    neg_accs = []
    neg_los = []
    neg_his = []

    for emotion in EMOTIONS:
        # Positive steering
        pos_correct = by_condition.get(f"{emotion}_+", [])
        if pos_correct:
            acc, lo, hi = wilson_ci(sum(pos_correct), len(pos_correct))
            pos_accs.append(acc * 100)
            pos_los.append((acc - lo) * 100)
            pos_his.append((hi - acc) * 100)
        else:
            pos_accs.append(0)
            pos_los.append(0)
            pos_his.append(0)

        # Negative steering
        neg_correct = by_condition.get(f"{emotion}_-", [])
        if neg_correct:
            acc, lo, hi = wilson_ci(sum(neg_correct), len(neg_correct))
            neg_accs.append(acc * 100)
            neg_los.append((acc - lo) * 100)
            neg_his.append((hi - acc) * 100)
        else:
            neg_accs.append(0)
            neg_los.append(0)
            neg_his.append(0)

    colors = [EMOTION_COLORS[e] for e in EMOTIONS]

    # Plot bars
    bars1 = ax.bar(x - width/2, pos_accs, width,
                   yerr=[pos_los, pos_his],
                   color=colors, alpha=0.9, label='+10%', edgecolor='black')
    bars2 = ax.bar(x + width/2, neg_accs, width,
                   yerr=[neg_los, neg_his],
                   color=colors, alpha=0.4, label='-10%', edgecolor='black', hatch='//')

    # Baseline line
    ax.axhline(baseline_acc * 100, color='black', linestyle='--', linewidth=2,
               label=f'Baseline ({baseline_acc*100:.0f}%)')
    ax.fill_between([-0.5, len(EMOTIONS)-0.5],
                    baseline_lo * 100, baseline_hi * 100,
                    color='gray', alpha=0.2)

    ax.set_xlabel('Emotion Steering')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Sandbagging Accuracy: {framing_type.upper()} Framings\n(hidden_scratchpad format, 10% steering)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    output_file = output_dir / f'sandbagging_{framing_type}_accuracy.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

    return {
        "framing_type": framing_type,
        "baseline": baseline_acc * 100,
        "best_condition": None,
        "best_acc": baseline_acc * 100,
        "worst_condition": None,
        "worst_acc": baseline_acc * 100,
    }


def print_summary_table(results: list):
    """Print summary table of results."""

    print("\n" + "=" * 80)
    print("SANDBAGGING ACCURACY BY FRAMING TYPE AND CONDITION")
    print("=" * 80)

    framing_types = sorted(set(r["framing_type"] for r in results))

    for framing_type in framing_types:
        framing_results = [r for r in results if r["framing_type"] == framing_type]

        by_condition = defaultdict(list)
        for r in framing_results:
            by_condition[r["condition"]].append(r["is_correct"])

        print(f"\n{framing_type.upper()} FRAMINGS:")
        print("-" * 60)

        # Header
        print(f"{'Condition':<15} {'N':>6} {'Correct':>8} {'Accuracy':>10}")
        print("-" * 45)

        # Sort conditions
        conditions = ["baseline"] + [f"{e}_+" for e in EMOTIONS] + [f"{e}_-" for e in EMOTIONS]

        baseline_acc = None
        best_delta = 0
        best_cond = None
        worst_delta = 0
        worst_cond = None

        for cond in conditions:
            if cond not in by_condition:
                continue
            correct = by_condition[cond]
            n = len(correct)
            n_correct = sum(correct)
            acc = n_correct / n * 100

            if cond == "baseline":
                baseline_acc = acc
                print(f"{cond:<15} {n:>6} {n_correct:>8} {acc:>9.1f}%")
            else:
                delta = acc - baseline_acc
                marker = ""
                if delta > best_delta:
                    best_delta = delta
                    best_cond = cond
                if delta < worst_delta:
                    worst_delta = delta
                    worst_cond = cond

                if abs(delta) > 10:
                    marker = " ***" if delta > 0 else " !!!"
                elif abs(delta) > 5:
                    marker = " *" if delta > 0 else " !"

                print(f"{cond:<15} {n:>6} {n_correct:>8} {acc:>9.1f}%  (Δ={delta:+.1f}%){marker}")

        print(f"\n  Best:  {best_cond} (Δ={best_delta:+.1f}%)")
        print(f"  Worst: {worst_cond} (Δ={worst_delta:+.1f}%)")


def create_comparison_plot(results: list, output_dir: Path):
    """Create comparison of all framing types in 2 rows."""

    framing_types = sorted(set(r["framing_type"] for r in results))

    # Order: FEAR first, then others
    ordered_types = []
    if "fear" in framing_types:
        ordered_types.append("fear")
    for ft in framing_types:
        if ft != "fear":
            ordered_types.append(ft)
    framing_types = ordered_types

    n_types = len(framing_types)

    # Split into 2 rows: 3 on top, rest on bottom
    n_top = 3 if n_types > 3 else n_types
    n_bottom = n_types - n_top

    fig = plt.figure(figsize=(5 * max(n_top, n_bottom), 10))

    # Create GridSpec for 2 rows
    if n_bottom > 0:
        gs = fig.add_gridspec(2, max(n_top, n_bottom), hspace=0.35, wspace=0.15)
    else:
        gs = fig.add_gridspec(1, n_top, wspace=0.15)

    axes = []
    for i in range(n_top):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
    for i in range(n_bottom):
        # Center the bottom row if fewer items
        offset = (max(n_top, n_bottom) - n_bottom) // 2
        ax = fig.add_subplot(gs[1, i + offset])
        axes.append(ax)

    for idx, (ax, framing_type) in enumerate(zip(axes, framing_types)):
        framing_results = [r for r in results if r["framing_type"] == framing_type]

        by_condition = defaultdict(list)
        for r in framing_results:
            by_condition[r["condition"]].append(r["is_correct"])

        baseline_correct = by_condition.get("baseline", [])
        baseline_acc = sum(baseline_correct) / len(baseline_correct) * 100 if baseline_correct else 0

        x = np.arange(len(EMOTIONS))
        width = 0.35

        pos_accs = []
        neg_accs = []

        for emotion in EMOTIONS:
            pos = by_condition.get(f"{emotion}_+", [])
            neg = by_condition.get(f"{emotion}_-", [])
            pos_accs.append(sum(pos) / len(pos) * 100 if pos else 0)
            neg_accs.append(sum(neg) / len(neg) * 100 if neg else 0)

        colors = [EMOTION_COLORS[e] for e in EMOTIONS]

        ax.bar(x - width/2, pos_accs, width, color=colors, alpha=0.9, edgecolor='black', label='+10%')
        ax.bar(x + width/2, neg_accs, width, color=colors, alpha=0.4, edgecolor='black', hatch='//', label='-10%')
        ax.axhline(baseline_acc, color='black', linestyle='--', linewidth=2, label=f'Baseline ({baseline_acc:.0f}%)')

        ax.set_xlabel('Emotion')
        ax.set_title(f'{framing_type.upper()} Framings', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylabel('Accuracy (%)')

        # Add legend to first plot only
        if idx == 0:
            ax.legend(loc='lower left', fontsize=9)

    plt.suptitle('Sandbagging Accuracy by Framing Type\n(hidden_scratchpad format, 10% steering)',
                 fontsize=14, fontweight='bold', y=0.98)

    output_file = output_dir / 'sandbagging_framings_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sandbagging framing results")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results.parent

    print(f"Loading results from {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} results")

    # Print summary
    print_summary_table(results)

    # Create plots for each framing type
    framing_types = sorted(set(r["framing_type"] for r in results))
    for framing_type in framing_types:
        plot_framing_results(results, framing_type, args.output_dir)

    # Create comparison plot
    create_comparison_plot(results, args.output_dir)


if __name__ == "__main__":
    main()
