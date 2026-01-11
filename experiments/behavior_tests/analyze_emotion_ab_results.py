"""
Analyze emotion A/B steering experiment results.

Computes position-corrected probabilities and visualizes steering effects.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Color scheme from probe_configs
EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F',
}

EMOTIONS = ["fear", "anger", "sadness", "disgust", "happiness", "surprise"]
CATEGORIES = ["interpretation", "attribution", "decision", "prediction"]


def load_results(results_path: Path) -> list:
    """Load results from JSONL file."""
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_position_corrected_probs(results: list) -> dict:
    """
    Compute position-corrected probability of choosing original option_a.

    For each question × condition:
    - P(option_a) = (P(A|original) + P(B|swapped)) / 2

    This corrects for position bias (tendency to choose A or B regardless of content).
    """
    # Group by condition and question
    grouped = defaultdict(lambda: defaultdict(dict))

    for r in results:
        cond = r["condition"]
        qid = r["question_id"]
        order = r["order"]

        # Store prob of choosing the presented "A" option
        grouped[cond][qid][order] = {
            "prob_A": r["prob_A"],
            "prob_B": r["prob_B"],
            "predictions": r["predictions"],
            "category": r["category"],
        }

    # Compute position-corrected probabilities
    corrected = {}
    for cond in grouped:
        corrected[cond] = {}
        for qid in grouped[cond]:
            orig = grouped[cond][qid].get("original", {})
            swap = grouped[cond][qid].get("swapped", {})

            if orig and swap:
                # P(option_a) = (P(A|original) + P(B|swapped)) / 2
                # Because in swapped order, B = original option_a
                p_option_a_orig = orig.get("prob_A") or 0
                p_option_a_swap = swap.get("prob_B") or 0

                p_option_a = (p_option_a_orig + p_option_a_swap) / 2
                p_option_b = 1 - p_option_a

                corrected[cond][qid] = {
                    "p_option_a": p_option_a,
                    "p_option_b": p_option_b,
                    "predictions": orig["predictions"],
                    "category": orig["category"],
                    # Also store raw for analysis
                    "raw_orig_A": p_option_a_orig,
                    "raw_swap_B": p_option_a_swap,
                }

    return corrected


def analyze_steering_effects(corrected: dict) -> dict:
    """
    Analyze whether steering shifts answers in predicted directions.

    Returns effect sizes and statistics per emotion.
    """
    baseline = corrected.get("baseline", {})
    if not baseline:
        print("No baseline found!")
        return {}

    # For each emotion, compute effect on predicted questions
    effects = {}

    for emotion in EMOTIONS:
        pos_cond = f"{emotion}_+"
        neg_cond = f"{emotion}_-"

        pos_data = corrected.get(pos_cond, {})
        neg_data = corrected.get(neg_cond, {})

        effects[emotion] = {
            "positive": {"shifts": [], "predicted_shifts": [], "questions": []},
            "negative": {"shifts": [], "predicted_shifts": [], "questions": []},
        }

        # Map emotion name to prediction key (happiness vs joy)
        pred_key = "joy" if emotion == "happiness" else emotion

        for qid in baseline:
            base_prob_a = baseline[qid]["p_option_a"]
            predictions = baseline[qid]["predictions"]
            category = baseline[qid]["category"]

            # Get prediction for this emotion
            pred = predictions.get(pred_key, {})
            predicted_dir = pred.get("direction")  # "A", "B", or None

            # Positive steering effect
            if qid in pos_data:
                pos_prob_a = pos_data[qid]["p_option_a"]
                shift = pos_prob_a - base_prob_a  # Positive = more likely to choose A

                effects[emotion]["positive"]["shifts"].append(shift)
                effects[emotion]["positive"]["questions"].append({
                    "qid": qid,
                    "category": category,
                    "shift": shift,
                    "predicted_dir": predicted_dir,
                })

                # Did it shift in predicted direction?
                if predicted_dir == "A":
                    effects[emotion]["positive"]["predicted_shifts"].append(shift)
                elif predicted_dir == "B":
                    effects[emotion]["positive"]["predicted_shifts"].append(-shift)

            # Negative steering effect
            if qid in neg_data:
                neg_prob_a = neg_data[qid]["p_option_a"]
                shift = neg_prob_a - base_prob_a

                effects[emotion]["negative"]["shifts"].append(shift)
                effects[emotion]["negative"]["questions"].append({
                    "qid": qid,
                    "category": category,
                    "shift": shift,
                    "predicted_dir": predicted_dir,
                })

                # Negative steering should shift OPPOSITE to prediction
                if predicted_dir == "A":
                    effects[emotion]["negative"]["predicted_shifts"].append(-shift)
                elif predicted_dir == "B":
                    effects[emotion]["negative"]["predicted_shifts"].append(shift)

    return effects


def plot_results(corrected: dict, effects: dict, output_dir: Path):
    """Create visualization plots."""

    # 1. Heatmap: Questions × Emotions showing probability shift
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Get all question IDs in order
    baseline = corrected.get("baseline", {})
    qids = sorted(baseline.keys())

    # Positive steering heatmap
    ax = axes[0]
    data_pos = []
    for qid in qids:
        row = []
        for emotion in EMOTIONS:
            pos_cond = f"{emotion}_+"
            if pos_cond in corrected and qid in corrected[pos_cond]:
                shift = corrected[pos_cond][qid]["p_option_a"] - baseline[qid]["p_option_a"]
                row.append(shift)
            else:
                row.append(0)
        data_pos.append(row)

    im = ax.imshow(data_pos, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels(qids, fontsize=6)
    ax.set_title('Positive Steering (+10%)\nΔP(option_a)', fontweight='bold')
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Question ID')
    plt.colorbar(im, ax=ax, label='Probability shift')

    # Negative steering heatmap
    ax = axes[1]
    data_neg = []
    for qid in qids:
        row = []
        for emotion in EMOTIONS:
            neg_cond = f"{emotion}_-"
            if neg_cond in corrected and qid in corrected[neg_cond]:
                shift = corrected[neg_cond][qid]["p_option_a"] - baseline[qid]["p_option_a"]
                row.append(shift)
            else:
                row.append(0)
        data_neg.append(row)

    im = ax.imshow(data_neg, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.set_yticks(range(len(qids)))
    ax.set_yticklabels(qids, fontsize=6)
    ax.set_title('Negative Steering (-10%)\nΔP(option_a)', fontweight='bold')
    ax.set_xlabel('Emotion')
    plt.colorbar(im, ax=ax, label='Probability shift')

    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_ab_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'emotion_ab_heatmap.png'}")

    # 2. Bar chart: Average effect by emotion (on predicted questions)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Positive steering
    ax = axes[0]
    means = []
    sems = []
    colors = []
    for emotion in EMOTIONS:
        shifts = effects[emotion]["positive"]["predicted_shifts"]
        if shifts:
            means.append(np.mean(shifts))
            sems.append(np.std(shifts) / np.sqrt(len(shifts)))
        else:
            means.append(0)
            sems.append(0)
        colors.append(EMOTION_COLORS[emotion])

    x = np.arange(len(EMOTIONS))
    bars = ax.bar(x, means, yerr=[1.96 * s for s in sems], color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.set_ylabel('Avg shift toward predicted answer')
    ax.set_title('Positive Steering (+10%)\nEffect on Predicted Questions', fontweight='bold')
    ax.set_ylim(-0.3, 0.3)

    # Negative steering
    ax = axes[1]
    means = []
    sems = []
    for emotion in EMOTIONS:
        shifts = effects[emotion]["negative"]["predicted_shifts"]
        if shifts:
            means.append(np.mean(shifts))
            sems.append(np.std(shifts) / np.sqrt(len(shifts)))
        else:
            means.append(0)
            sems.append(0)

    bars = ax.bar(x, means, yerr=[1.96 * s for s in sems], color=colors, edgecolor='black', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.set_ylabel('Avg shift toward predicted answer')
    ax.set_title('Negative Steering (-10%)\nEffect on Predicted Questions', fontweight='bold')
    ax.set_ylim(-0.3, 0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'emotion_ab_effects.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'emotion_ab_effects.png'}")

    # 3. Summary statistics table
    print("\n" + "="*70)
    print("SUMMARY: Steering Effects on Predicted Questions")
    print("="*70)
    print(f"{'Emotion':<12} {'Dir':<5} {'Mean Shift':<12} {'SE':<10} {'N':<5} {'t-stat':<8} {'p-value':<10}")
    print("-"*70)

    for emotion in EMOTIONS:
        for direction, label in [("positive", "+"), ("negative", "-")]:
            shifts = effects[emotion][direction]["predicted_shifts"]
            if shifts and len(shifts) > 1:
                mean = np.mean(shifts)
                se = np.std(shifts) / np.sqrt(len(shifts))
                t_stat, p_val = stats.ttest_1samp(shifts, 0)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                print(f"{emotion:<12} {label:<5} {mean:>+.4f}      {se:.4f}     {len(shifts):<5} {t_stat:>+.3f}    {p_val:.4f} {sig}")
            else:
                print(f"{emotion:<12} {label:<5} {'N/A':<12} {'N/A':<10} {len(shifts) if shifts else 0:<5}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Analyze emotion A/B steering results")
    parser.add_argument("--results", type=Path, required=True,
                        help="Path to results JSONL file")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory for plots (default: same as results)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results.parent

    # Load and analyze
    print(f"Loading results from {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} result entries")

    print("\nComputing position-corrected probabilities...")
    corrected = compute_position_corrected_probs(results)

    print("Analyzing steering effects...")
    effects = analyze_steering_effects(corrected)

    print("Creating plots...")
    plot_results(corrected, effects, args.output_dir)


if __name__ == "__main__":
    main()
