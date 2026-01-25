"""
Plot blackmail steering comparison across all three models.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Use eval_dashboard colors
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # sky blue
    'disgust': '#7D9B7D',    # olive green
    'fear': '#a59dc9',       # lavender
    'happiness': '#D4876A',  # coral
    'sadness': '#B8CCC8',    # sage
    'surprise': '#D1728F',   # darker pink
}

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for proportion."""
    if n == 0:
        return 0, 0, 0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center, max(0, center - spread), min(1, center + spread)

def load_results(filepath):
    """Load judged results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results

def compute_blackmail_rates(results, exclude_10pct=False):
    """Compute blackmail rates by condition."""
    condition_counts = defaultdict(lambda: {'blackmail': 0, 'total': 0})

    for r in results:
        cond = r.get('condition', 'unknown')

        # Skip 10% conditions if requested
        if exclude_10pct and '10%' in cond:
            continue

        condition_counts[cond]['total'] += 1
        # Handle both dict format (is_blackmail) and string format (YES/NO)
        judge = r.get('blackmail_judge', {})
        if isinstance(judge, dict):
            is_blackmail = judge.get('is_blackmail', False)
        else:
            is_blackmail = judge == 'YES'
        if is_blackmail:
            condition_counts[cond]['blackmail'] += 1

    rates = {}
    for cond, counts in condition_counts.items():
        center, lo, hi = wilson_ci(counts['blackmail'], counts['total'])
        rates[cond] = {
            'rate': center * 100,
            'ci_lo': lo * 100,
            'ci_hi': hi * 100,
            'n': counts['total']
        }
    return rates

def main():
    output_dir = Path("experiments/steering/outputs/blackmail")

    # Load results from all three models
    gemma_file = output_dir / "blackmail_gemma_text_layer30_20260122_111425.judged.jsonl"
    qwen32b_file = output_dir / "blackmail_qwen32b_text_layer30_20260122_112226.judged.jsonl"
    qwen235b_file = output_dir / "blackmail_qwen235b_text_layer50_20260122_114510.judged.jsonl"

    gemma_results = load_results(gemma_file)
    qwen32b_results = load_results(qwen32b_file)
    qwen235b_results = load_results(qwen235b_file)

    # Compute rates (exclude 10% for Gemma due to incoherence)
    gemma_rates = compute_blackmail_rates(gemma_results, exclude_10pct=True)
    qwen32b_rates = compute_blackmail_rates(qwen32b_results, exclude_10pct=False)
    qwen235b_rates = compute_blackmail_rates(qwen235b_results, exclude_10pct=False)

    # Emotions to plot
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

    models = [
        ('Gemma 3 27B (7% only)', gemma_rates, '7%'),
        ('Qwen3-32B (7%, 10%)', qwen32b_rates, None),
        ('Qwen3-235B (7%, 10%)', qwen235b_rates, None),
    ]

    for ax, (model_name, rates, pct_filter) in zip(axes, models):
        # Get baseline
        baseline = rates.get('baseline', {}).get('rate', 0)
        baseline_lo = rates.get('baseline', {}).get('ci_lo', 0)
        baseline_hi = rates.get('baseline', {}).get('ci_hi', 0)

        x_positions = []
        x_labels = []
        x = 0

        # Plot baseline first
        ax.bar(x, baseline, color='#808080', width=0.8, label='Baseline')
        ax.errorbar(x, baseline, yerr=[[baseline - baseline_lo], [baseline_hi - baseline]],
                   fmt='none', color='black', capsize=0, linewidth=0.8)
        x_positions.append(x)
        x_labels.append('Baseline')
        x += 1.5

        # Plot each emotion
        for emotion in emotions:
            color = EMOTION_COLORS.get(emotion, '#888888')

            # Find conditions for this emotion
            if pct_filter:
                # Only show specific percentage (for Gemma)
                pos_cond = f"{emotion}_+{pct_filter}"
                neg_cond = f"{emotion}_-{pct_filter}"
                conditions = [(pos_cond, '+'), (neg_cond, '-')]
            else:
                # Show both 7% and 10%
                conditions = [
                    (f"{emotion}_+7%", '+7'),
                    (f"{emotion}_-7%", '-7'),
                    (f"{emotion}_+10%", '+10'),
                    (f"{emotion}_-10%", '-10'),
                ]

            emotion_x_start = x
            for cond, label in conditions:
                if cond in rates:
                    rate = rates[cond]['rate']
                    ci_lo = rates[cond]['ci_lo']
                    ci_hi = rates[cond]['ci_hi']

                    # Darker for positive, lighter for negative
                    if '+' in label:
                        bar_color = color
                        alpha = 1.0
                    else:
                        bar_color = color
                        alpha = 0.5

                    ax.bar(x, rate, color=bar_color, width=0.4, alpha=alpha)
                    ax.errorbar(x, rate, yerr=[[rate - ci_lo], [ci_hi - rate]],
                               fmt='none', color='black', capsize=0, linewidth=0.8)
                    x += 0.45

            # Add emotion label
            emotion_x_center = (emotion_x_start + x - 0.45) / 2
            x_positions.append(emotion_x_center)
            x_labels.append(emotion.capitalize())
            x += 0.8

        # Add baseline reference line
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Blackmail Rate (%)' if ax == axes[0] else '')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylim(0, max(80, ax.get_ylim()[1]))
        ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', label='Baseline'),
        Patch(facecolor='#888888', alpha=1.0, label='+ direction (add emotion)'),
        Patch(facecolor='#888888', alpha=0.5, label='- direction (suppress emotion)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.suptitle('Blackmail Rate by Emotion Steering Across Models\nTEXT Mean Diff Vectors | 95% CI',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = output_dir / "blackmail_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    output_pdf = output_dir / "blackmail_model_comparison.pdf"
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_pdf}")

if __name__ == "__main__":
    main()
