"""
Plot comparison of steering methods for sandbagging:
- Baseline
- Targeted (prefill only, no generation steering)
- Global prefill only (no generation steering)
- Global + generation steering

Uses average sandbagging score (1-5 scale: 1=clear sandbagging, 5=clearly honest)
"""
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def compute_stats(results):
    by_condition = defaultdict(list)
    for r in results:
        if "judgment" in r and "error" not in r.get("judgment", {}):
            by_condition[r["condition"]].append(r)

    stats = {}
    for cond, cond_results in by_condition.items():
        n = len(cond_results)
        if n == 0:
            continue
        scores = [r["judgment"].get("sandbagging_score", 3) for r in cond_results]
        mean_score = np.mean(scores)
        sem_score = np.std(scores, ddof=1) / np.sqrt(n) if n > 1 else 0

        stats[cond] = {
            "n": n,
            "mean_score": mean_score,
            "sem_score": sem_score,
        }
    return stats


def main():
    output_dir = Path("experiments/steering/outputs")

    # Load global+generation results (fear framing)
    files_10pct = [
        "experiments/behavior_tests/outputs/judged_10pct.jsonl",
        "experiments/behavior_tests/outputs/judged_10pct_anger_sadness.jsonl",
        "experiments/behavior_tests/outputs/judged_10pct_remaining.jsonl",
    ]

    global_gen_results = []
    for f in files_10pct:
        p = Path(f)
        if p.exists():
            with open(p) as fp:
                for line in fp:
                    global_gen_results.append(json.loads(line))

    # Filter for fear framing + our problems
    fear_filtered = [r for r in global_gen_results
                     if r.get("framing_type") == "fear"
                     and r.get("problem_id") in ["bat_ball", "birthday_paradox"]]

    global_gen_stats = compute_stats(fear_filtered)

    # Load targeted results (no generation steering) - find most recent judged file
    targeted_results = []
    judged_files = sorted(output_dir.glob("judged_targeted_sandbagging_layer30_*.jsonl"))
    for jf in judged_files:
        with open(jf) as f:
            for line in f:
                targeted_results.append(json.loads(line))

    targeted_stats = compute_stats(targeted_results)

    # Debug: print available conditions
    print("Targeted stats conditions:", list(targeted_stats.keys()))
    print("Global+gen stats conditions:", list(global_gen_stats.keys()))

    # Use global+gen baseline for fair comparison (larger sample, same prompts)
    global_baseline_score = global_gen_stats.get('baseline', {}).get('mean_score', 3.0)
    targeted_baseline_score = targeted_stats.get('baseline', {}).get('mean_score', 3.0)
    print(f"\nBaseline comparison:")
    print(f"  Global+gen baseline: {global_baseline_score:.2f} (n={global_gen_stats.get('baseline', {}).get('n', 0)})")
    print(f"  Targeted baseline: {targeted_baseline_score:.2f} (n={targeted_stats.get('baseline', {}).get('n', 0)})")

    # === CREATE COMPARISON PLOT ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Colors
    colors = {
        'baseline': '#888888',
        'targeted': '#7BA7D7',
        'global_prefill': '#D4876A',
        'global_gen': '#7D9B7D',
    }

    # === Left plot: Anger comparison ===
    conditions = [
        ('Baseline\n(no steering)', 'baseline', None, None),
        ('Targeted 10%\n(prefill only)', 'targeted', 'anger_+10%_targeted', None),
        ('Targeted 30%\n(prefill only)', 'targeted', 'anger_+30%_targeted', None),
        ('Targeted 50%\n(prefill only)', 'targeted', 'anger_+50%_targeted', None),
        ('Global 10%\n(prefill only)', 'global_prefill', 'anger_+10%_global', None),
        ('Global 10%\n(prefill+gen)', 'global_gen', None, 'anger_+'),
    ]

    labels = []
    mean_scores = []
    sem_scores = []
    bar_colors = []

    for label, color_key, targeted_cond, global_cond in conditions:
        labels.append(label)
        bar_colors.append(colors[color_key])

        if targeted_cond:
            s = targeted_stats.get(targeted_cond, {})
        elif global_cond:
            s = global_gen_stats.get(global_cond, {})
        elif color_key == 'baseline':
            # Use targeted baseline (true no-steering baseline)
            s = targeted_stats.get('baseline', {})
        else:
            s = {}

        mean_scores.append(s.get('mean_score', 3.0))
        sem_scores.append(s.get('sem_score', 0))

    x = np.arange(len(labels))
    bars = ax1.bar(x, mean_scores, yerr=sem_scores, color=bar_colors,
                   edgecolor='black', capsize=4, error_kw={'linewidth': 1.5}, alpha=0.85)

    # Baseline reference line (use targeted baseline - true no-steering)
    baseline_score = targeted_stats.get('baseline', {}).get('mean_score', 4.0)
    ax1.axhline(baseline_score, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax1.set_ylabel('Average Sandbagging Score\n(1=sandbagging, 5=honest)', fontsize=11)
    ax1.set_title('Anger Steering: Targeted vs Global\n(+ steering direction)', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(1, 5)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, mean_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # === Right plot: Fear comparison (strongest effect) ===
    conditions_fear = [
        ('Baseline\n(no steering)', 'baseline', None, None),
        ('Fear Targeted 30%\n(prefill only)', 'targeted', 'fear_+30%_targeted', None),
        ('Fear Targeted 50%\n(prefill only)', 'targeted', 'fear_+50%_targeted', None),
        ('Fear Global 10%\n(prefill only)', 'global_prefill', 'fear_+10%_global', None),
        ('Fear Global 10%\n(prefill+gen)', 'global_gen', None, 'fear_+'),
    ]

    labels2 = []
    mean_scores2 = []
    sem_scores2 = []
    bar_colors2 = []

    for label, color_key, targeted_cond, global_cond in conditions_fear:
        labels2.append(label)
        bar_colors2.append(colors[color_key])

        if targeted_cond:
            s = targeted_stats.get(targeted_cond, {})
        elif global_cond:
            s = global_gen_stats.get(global_cond, {})
        elif color_key == 'baseline':
            # Use targeted baseline (true no-steering baseline)
            s = targeted_stats.get('baseline', {})
        else:
            s = {}

        mean_scores2.append(s.get('mean_score', 3.0))
        sem_scores2.append(s.get('sem_score', 0))

    x2 = np.arange(len(labels2))
    bars2 = ax2.bar(x2, mean_scores2, yerr=sem_scores2, color=bar_colors2,
                    edgecolor='black', capsize=4, error_kw={'linewidth': 1.5}, alpha=0.85)

    ax2.axhline(baseline_score, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_ylabel('Average Sandbagging Score\n(1=sandbagging, 5=honest)', fontsize=11)
    ax2.set_title('Fear Steering: Targeted vs Global\n(+ steering direction)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=9)
    ax2.set_ylim(1, 5)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)

    for i, (bar, val) in enumerate(zip(bars2, mean_scores2)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Effect of Generation Steering on Sandbagging\n(Prefill-only vs Prefill+Generation)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / "sandbagging_steering_method_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Steering Method Comparison (Average Sandbagging Score)")
    print("Scale: 1=clear sandbagging, 5=clearly honest (lower = more sandbagging)")
    print("="*70)
    print(f"\n{'Method':<35} {'Anger+ Score':>14} {'Fear+ Score':>14}")
    print("-"*65)
    print(f"{'Baseline (no steering)':<35} {targeted_stats.get('baseline', {}).get('mean_score', 0):>13.2f} {targeted_stats.get('baseline', {}).get('mean_score', 0):>13.2f}")
    print(f"{'Targeted 30% (prefill only)':<35} {targeted_stats.get('anger_+30%_targeted', {}).get('mean_score', 0):>13.2f} {targeted_stats.get('fear_+30%_targeted', {}).get('mean_score', 0):>13.2f}")
    print(f"{'Targeted 50% (prefill only)':<35} {targeted_stats.get('anger_+50%_targeted', {}).get('mean_score', 0):>13.2f} {targeted_stats.get('fear_+50%_targeted', {}).get('mean_score', 0):>13.2f}")
    print(f"{'Global 10% (prefill only)':<35} {targeted_stats.get('anger_+10%_global', {}).get('mean_score', 0):>13.2f} {targeted_stats.get('fear_+10%_global', {}).get('mean_score', 0):>13.2f}")
    print(f"{'Global 10% (prefill+gen)':<35} {global_gen_stats.get('anger_+', {}).get('mean_score', 0):>13.2f} {global_gen_stats.get('fear_+', {}).get('mean_score', 0):>13.2f}")


if __name__ == "__main__":
    main()
