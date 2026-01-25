"""
Plot entity steering results.

Shows how steering emotions on specific entity tokens affects A/B preferences.
Style matches the sandbagging plots.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F',
    'baseline': '#888888',
}


def load_results(filepath: Path):
    """Load results from JSONL file."""
    results = []
    with open(filepath) as f:
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


def compute_stats(results):
    """
    Compute preference statistics by condition.

    Key insight: When we steer negative emotion on entity at position A,
    does the model become less likely to choose A?

    Returns stats grouped by:
    - condition name
    - target (entity_a or entity_b)
    """
    by_condition = defaultdict(list)

    for r in results:
        cond = r["condition"]
        by_condition[cond].append(r)

    stats = {}
    for cond, cond_results in by_condition.items():
        n = len(cond_results)
        if n == 0:
            continue

        # Get P(A) values (averaging across both orderings to control for position bias)
        probs_a = [r["prob_A"] for r in cond_results if r["prob_A"] is not None]

        # Count how often A was chosen
        a_chosen = sum(1 for r in cond_results if r["generated_answer"].upper() == "A")

        if probs_a:
            avg_prob_a = np.mean(probs_a)
            std_prob_a = np.std(probs_a) / np.sqrt(len(probs_a))  # SEM
        else:
            avg_prob_a = 0.5
            std_prob_a = 0

        # Wilson CI for choice rate
        choice_rate, ci_low, ci_high = wilson_ci(a_chosen, n)

        stats[cond] = {
            "n": n,
            "avg_prob_a": avg_prob_a,
            "sem_prob_a": std_prob_a,
            "choice_rate_a": choice_rate * 100,
            "ci_low": ci_low * 100,
            "ci_high": ci_high * 100,
        }

    return stats


def plot_preference_shift(results, output_path: Path):
    """
    Plot how steering affects preference for the steered entity.
    Style matches the sandbagging plots with grouped bars by emotion.
    """
    stats = compute_stats(results)

    # Extract emotions tested
    emotions = set()
    for cond in stats.keys():
        if cond != "baseline":
            parts = cond.split("_")
            if len(parts) >= 3:
                emotions.add(parts[0])
    emotions = sorted(emotions)

    if not emotions:
        print("No emotion conditions found!")
        return

    # Get baseline
    baseline_stats = stats.get("baseline", {"choice_rate_a": 50, "ci_low": 50, "ci_high": 50})
    baseline_rate = baseline_stats["choice_rate_a"]
    baseline_ci_lo = baseline_stats["choice_rate_a"] - baseline_stats["ci_low"]
    baseline_ci_hi = baseline_stats["ci_high"] - baseline_stats["choice_rate_a"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Plot 1: Steering on Entity A =====
    ax1 = axes[0]

    all_labels = ["baseline"] + emotions
    x = np.arange(len(all_labels))
    width = 0.35

    # Build data arrays
    pos_rates = [baseline_rate]  # baseline in positive position
    pos_errs_lo = [baseline_ci_lo]
    pos_errs_hi = [baseline_ci_hi]
    neg_rates = [0]  # no negative for baseline
    neg_errs_lo = [0]
    neg_errs_hi = [0]
    colors = ['#888888']  # gray for baseline

    for emotion in emotions:
        color = EMOTION_COLORS.get(emotion, '#888888')
        colors.append(color)

        # +emotion on A
        pos_cond = f"{emotion}_+_on_A"
        pos_stats = stats.get(pos_cond, {"choice_rate_a": 50, "ci_low": 50, "ci_high": 50})
        pos_rates.append(pos_stats["choice_rate_a"])
        pos_errs_lo.append(pos_stats["choice_rate_a"] - pos_stats["ci_low"])
        pos_errs_hi.append(pos_stats["ci_high"] - pos_stats["choice_rate_a"])

        # -emotion on A
        neg_cond = f"{emotion}_-_on_A"
        neg_stats = stats.get(neg_cond, {"choice_rate_a": 50, "ci_low": 50, "ci_high": 50})
        neg_rates.append(neg_stats["choice_rate_a"])
        neg_errs_lo.append(neg_stats["choice_rate_a"] - neg_stats["ci_low"])
        neg_errs_hi.append(neg_stats["ci_high"] - neg_stats["choice_rate_a"])

    # Plot bars
    ax1.bar(x - width/2, pos_rates, width,
            yerr=[pos_errs_lo, pos_errs_hi],
            color=colors, alpha=0.9, label='+10%',
            edgecolor='black', capsize=3, error_kw={'linewidth': 1})

    # Only plot negative bars where we have data (skip baseline)
    neg_x = x[1:]
    neg_vals = neg_rates[1:]
    neg_lo = neg_errs_lo[1:]
    neg_hi = neg_errs_hi[1:]
    neg_colors = colors[1:]
    ax1.bar(neg_x + width/2, neg_vals, width,
            yerr=[neg_lo, neg_hi],
            color=neg_colors, alpha=0.4, label='-10%',
            edgecolor='black', hatch='//', capsize=3, error_kw={'linewidth': 1})

    ax1.axhline(baseline_rate, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.3)

    ax1.set_xlabel('Emotion Steering')
    ax1.set_ylabel('% Choosing A (target entity)')
    ax1.set_title('Steering on Entity A', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 80)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='upper right', fontsize=9)

    # ===== Plot 2: Steering on Entity B =====
    ax2 = axes[1]

    pos_rates_b = [baseline_rate]
    pos_errs_lo_b = [baseline_ci_lo]
    pos_errs_hi_b = [baseline_ci_hi]
    neg_rates_b = [0]
    neg_errs_lo_b = [0]
    neg_errs_hi_b = [0]

    for emotion in emotions:
        # +emotion on B
        pos_cond = f"{emotion}_+_on_B"
        pos_stats = stats.get(pos_cond, {"choice_rate_a": 50, "ci_low": 50, "ci_high": 50})
        pos_rates_b.append(pos_stats["choice_rate_a"])
        pos_errs_lo_b.append(pos_stats["choice_rate_a"] - pos_stats["ci_low"])
        pos_errs_hi_b.append(pos_stats["ci_high"] - pos_stats["choice_rate_a"])

        # -emotion on B
        neg_cond = f"{emotion}_-_on_B"
        neg_stats = stats.get(neg_cond, {"choice_rate_a": 50, "ci_low": 50, "ci_high": 50})
        neg_rates_b.append(neg_stats["choice_rate_a"])
        neg_errs_lo_b.append(neg_stats["choice_rate_a"] - neg_stats["ci_low"])
        neg_errs_hi_b.append(neg_stats["ci_high"] - neg_stats["choice_rate_a"])

    ax2.bar(x - width/2, pos_rates_b, width,
            yerr=[pos_errs_lo_b, pos_errs_hi_b],
            color=colors, alpha=0.9, label='+10%',
            edgecolor='black', capsize=3, error_kw={'linewidth': 1})

    neg_vals_b = neg_rates_b[1:]
    neg_lo_b = neg_errs_lo_b[1:]
    neg_hi_b = neg_errs_hi_b[1:]
    ax2.bar(neg_x + width/2, neg_vals_b, width,
            yerr=[neg_lo_b, neg_hi_b],
            color=neg_colors, alpha=0.4, label='-10%',
            edgecolor='black', hatch='//', capsize=3, error_kw={'linewidth': 1})

    ax2.axhline(baseline_rate, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(50, color='black', linestyle=':', linewidth=1, alpha=0.3)

    ax2.set_xlabel('Emotion Steering')
    ax2.set_ylabel('% Choosing A (non-target entity)')
    ax2.set_title('Steering on Entity B', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha='right')
    ax2.set_ylim(0, 80)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(loc='upper right', fontsize=9)

    plt.suptitle('Entity-Specific Emotion Steering (Layer 30)\n'
                 '(+% = add emotion to token, -% = subtract)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def plot_asymmetry(results, output_path: Path):
    """
    Plot the asymmetry: steering on A vs B should have opposite effects.

    Shows: delta_A = P(A | steer on A) - P(A | steer on B)
    Hypothesis: Negative emotion on A should decrease P(A), on B should increase P(A)
    """
    stats = compute_stats(results)

    # Extract emotions
    emotions = set()
    for cond in stats.keys():
        if cond != "baseline":
            parts = cond.split("_")
            if len(parts) >= 3:
                emotions.add(parts[0])
    emotions = sorted(emotions)

    if not emotions:
        print("No emotion conditions found!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(emotions))
    width = 0.35

    # Compute asymmetry: P(A | +emo on A) - P(A | +emo on B)
    # If emotion association works, steering +emo on A should increase P(A)
    # and steering +emo on B should decrease P(A), so this diff should be positive

    pos_asymmetry = []
    neg_asymmetry = []
    colors = []

    for emotion in emotions:
        pos_on_a = stats.get(f"{emotion}_+_on_A", {"choice_rate_a": 50})["choice_rate_a"]
        pos_on_b = stats.get(f"{emotion}_+_on_B", {"choice_rate_a": 50})["choice_rate_a"]
        neg_on_a = stats.get(f"{emotion}_-_on_A", {"choice_rate_a": 50})["choice_rate_a"]
        neg_on_b = stats.get(f"{emotion}_-_on_B", {"choice_rate_a": 50})["choice_rate_a"]

        # Asymmetry for positive steering
        pos_asymmetry.append(pos_on_a - pos_on_b)
        # Asymmetry for negative steering (should be opposite sign if it works)
        neg_asymmetry.append(neg_on_a - neg_on_b)

        colors.append(EMOTION_COLORS.get(emotion, '#888888'))

    bars1 = ax.bar(x - width/2, pos_asymmetry, width,
                   color=colors, alpha=0.9, label='+emotion',
                   edgecolor='black')

    bars2 = ax.bar(x + width/2, neg_asymmetry, width,
                   color=colors, alpha=0.4, label='-emotion',
                   edgecolor='black', hatch='//')

    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Asymmetry: P(A|steer on A) - P(A|steer on B)')
    ax.set_title('Entity-Specific Steering Asymmetry\n'
                 '(Positive = steering on target increases preference for target)',
                 fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved asymmetry plot to {output_path}")


def print_summary(results):
    """Print summary statistics."""
    stats = compute_stats(results)

    print("\n" + "=" * 80)
    print("ENTITY STEERING RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Condition':<25} {'N':<6} {'% Choose A':<12} {'95% CI':<15}")
    print("-" * 60)

    for cond in sorted(stats.keys()):
        s = stats[cond]
        ci_str = f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]"
        print(f"{cond:<25} {s['n']:<6} {s['choice_rate_a']:<12.1f} {ci_str:<15}")


def main():
    parser = argparse.ArgumentParser(description="Plot entity steering results")
    parser.add_argument("--input", type=Path, required=True,
                        help="Input JSONL file with results")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/experiments/behavior_tests/outputs"))
    args = parser.parse_args()

    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    print_summary(results)

    # Generate plots
    plot_preference_shift(results, args.output_dir / "entity_steering_preferences.png")
    plot_asymmetry(results, args.output_dir / "entity_steering_asymmetry.png")


if __name__ == "__main__":
    main()
