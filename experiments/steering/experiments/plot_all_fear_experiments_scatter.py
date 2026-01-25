"""
Scatter plot of accuracy vs fear for all experiment versions with coherency > 80.
Matches style of other anti-steer plots.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "anti_steer"

# Color scheme matching other plots
BASELINE_COLOR = '#808080'
FEAR_ONLY_COLOR = '#d94f4f'
ANTISTEER_COLORS = {
    'L57-61 -5%': '#7eb77e',
    'L57-61 -7.5%': '#4a9f4a',
    'L60-61 -10%': '#5ba3d9',
    'L60-61 -15%': '#2e7cb8',
}
CAPPING_COLOR = '#e6a23c'
ABLATION_COLOR = '#9b59b6'


def load_jsonl(path):
    """Load a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_stats(values):
    """Compute mean and standard error."""
    arr = np.array(values)
    mean = np.mean(arr)
    stderr = np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0
    return mean, stderr


def extract_condition_metrics(samples):
    """Extract accuracy, fear, coherency with standard errors."""
    correct = [1 if s.get("is_correct") else 0 for s in samples]
    acc_mean, acc_se = compute_stats(correct)

    fear_scores = []
    coherency_scores = []

    for s in samples:
        if "fear_sentiment_judge" in s and s["fear_sentiment_judge"]:
            fear_scores.append(s["fear_sentiment_judge"].get("fear_score", 0))
        if "coherency_judge" in s and s["coherency_judge"]:
            coherency_scores.append(s["coherency_judge"].get("coherency_score", 100))

    fear_mean, fear_se = compute_stats(fear_scores) if fear_scores else (0, 0)
    coh_mean, _ = compute_stats(coherency_scores) if coherency_scores else (100, 0)

    return {
        'accuracy': (acc_mean * 100, acc_se * 100),
        'fear': (fear_mean, fear_se),
        'coherency': coh_mean,
        'n': len(samples)
    }


def main():
    # Files to load
    files_to_load = [
        "sandbagging_multilayer_antisteer_20260122_203204_textmeandiff.judged.fear_judged.jsonl",
        "sandbagging_multilayer_antisteer_20260123_120107_textmeandiff.judged.fear_judged.jsonl",
        "sandbagging_multilayer_antisteer_20260123_122651_textmeandiff.judged.fear_judged.jsonl",
        "sandbagging_fear_capping_20260123_130803.judged.fear_judged.jsonl",
        "sandbagging_fear_capping_20260123_131701.judged.fear_judged.jsonl",
    ]

    # Collect all samples by canonical condition name
    all_samples_by_condition = defaultdict(list)

    for filename in files_to_load:
        filepath = OUTPUT_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            continue

        samples = load_jsonl(filepath)

        for s in samples:
            cond = s.get("condition", "unknown")

            # Map to canonical condition names
            if cond == "baseline":
                canonical = "Baseline"
            elif "fear_only" in cond or cond == "fear_+7.5%_only":
                canonical = "Fear +7.5% only"
            elif "antisteer_-5.0%" in cond:
                canonical = "L57-61 -5%"
            elif "antisteer_-7.5%" in cond:
                canonical = "L57-61 -7.5%"
            elif "antisteer_-10.0%" in cond:
                # Check which file this came from to determine layers
                if "120107" in filename:
                    canonical = "L60-61 -10%"
                else:
                    canonical = "L57-61 -10%"
            elif "antisteer_-15.0%" in cond:
                canonical = "L60-61 -15%"
            elif "cap_0.0std" in cond:
                canonical = "Ablation (cap=0)"
            elif "cap_0.2std" in cond:
                canonical = "Cap 0.2std"
            elif "cap_0.5std" in cond:
                canonical = "Cap 0.5std"
            elif "cap_1.0std" in cond:
                canonical = "Cap 1.0std"
            else:
                canonical = cond

            all_samples_by_condition[canonical].append(s)

    # Compute metrics for each condition
    metrics_by_condition = {}
    for cond, samples in all_samples_by_condition.items():
        metrics_by_condition[cond] = extract_condition_metrics(samples)

    # Filter for coherency > 80
    filtered = {k: v for k, v in metrics_by_condition.items() if v['coherency'] > 80}

    print("Conditions with coherency > 80:")
    for cond, m in sorted(filtered.items(), key=lambda x: x[1]['fear'][0]):
        print(f"  {cond:<25}: Acc={m['accuracy'][0]:5.1f}% ± {m['accuracy'][1]:4.1f}, "
              f"Fear={m['fear'][0]:5.1f} ± {m['fear'][1]:4.1f}, Coh={m['coherency']:.1f}, n={m['n']}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define plotting order and colors (excluding capping, keeping only ablation)
    plot_order = [
        ('Baseline', BASELINE_COLOR, 's', 16),
        ('Fear +7.5% only', FEAR_ONLY_COLOR, '^', 16),
        ('L57-61 -5%', ANTISTEER_COLORS['L57-61 -5%'], 'o', 14),
        ('L57-61 -7.5%', ANTISTEER_COLORS['L57-61 -7.5%'], 'o', 14),
        ('L60-61 -10%', ANTISTEER_COLORS['L60-61 -10%'], 'D', 13),
        ('L60-61 -15%', ANTISTEER_COLORS['L60-61 -15%'], 'D', 13),
        ('Ablation (cap=0)', ABLATION_COLOR, 'X', 15, 'Ablation L57-61'),
    ]

    # Track positions for label adjustment
    positions = []

    for item in plot_order:
        cond, color, marker, size = item[:4]
        legend_label = item[4] if len(item) > 4 else cond

        if cond not in filtered:
            continue
        m = filtered[cond]
        acc_mean, acc_se = m['accuracy']
        fear_mean, fear_se = m['fear']

        ax.errorbar(fear_mean, acc_mean, xerr=fear_se, yerr=acc_se,
                    fmt=marker, markersize=size, capsize=4, capthick=1.5,
                    color=color, label=legend_label, linewidth=1.5,
                    markeredgecolor='white', markeredgewidth=1)

        positions.append((fear_mean, acc_mean, cond))

    # Add labels with smart positioning to avoid overlap
    label_offsets = {
        'Baseline': (-12, 10),
        'Fear +7.5% only': (10, -15),
        'L57-61 -5%': (10, 6),
        'L57-61 -7.5%': (10, 6),
        'L60-61 -10%': (10, 6),
        'L60-61 -15%': (10, -15),
        'Ablation (cap=0)': (10, 6),
    }

    # Display names (simplified)
    display_names = {
        'Ablation (cap=0)': 'Ablation',
    }

    for fear, acc, cond in positions:
        offset = label_offsets.get(cond, (10, 6))
        display_name = display_names.get(cond, cond)
        ax.annotate(display_name, (fear, acc), textcoords="offset points",
                    xytext=offset, fontsize=12, alpha=0.9)

    ax.set_xlabel('Fear Sentiment Score', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy vs Fear Sentiment (All Experiments, Coherency > 80)',
                 fontsize=16, fontweight='bold')

    # Add ideal region shading
    ax.axhspan(0, 15, alpha=0.08, color='green')
    ax.axvspan(0, 15, alpha=0.08, color='blue')
    ax.annotate('Sandbagging without\nchange in text sentiment', xy=(7, 7), fontsize=11, ha='center', va='center',
                color='purple', fontweight='bold', alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 75)
    ax.set_ylim(0, 35)
    ax.tick_params(axis='both', labelsize=12)

    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "all_experiments_accuracy_vs_fear.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
