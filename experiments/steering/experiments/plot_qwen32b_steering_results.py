#!/usr/bin/env python3
"""
Plot Qwen 32B blackmail steering results in the same style as Qwen 235B plots.
Creates two plots: unstructured and structured prompts.
Updated to combine data from multiple runs for better statistical power.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import glob

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Consistent color scheme (matching sandbagging plots)
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # light blue
    'fear': '#a59dc9',       # light purple
}

EMOTIONS = ['anger', 'fear']
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "blackmail_steering"


def load_results(jsonl_path: Path):
    """Load and aggregate blackmail results from judged JSON."""
    results = defaultdict(lambda: {'blackmail': []})

    judged_path = jsonl_path.with_suffix('.judged.json')
    if not judged_path.exists():
        judged_path = Path(str(jsonl_path).replace('.jsonl', '.judged.json'))

    if judged_path.exists():
        with open(judged_path) as f:
            data = json.load(f)
        for d in data:
            cond = d.get('condition', 'unknown')
            # Try both possible judgment key names
            judgment = d.get('blackmail_judgment') or d.get('judgment', {})
            if isinstance(judgment, dict):
                is_blackmail = 1 if judgment.get('is_blackmail') else 0
                results[cond]['blackmail'].append(is_blackmail)
    else:
        print(f"Warning: {judged_path} not found")

    return results


def load_all_structured_results():
    """Load and combine all structured experiment results."""
    all_data = defaultdict(lambda: {"blackmail": []})

    # Load original structured files (100%, 125%, 150%)
    for pct in [100, 125, 150]:
        pattern = f"blackmail_qwen32b_text_{pct}pct_layer30_structured_*.judged.json"
        files = list(OUTPUT_DIR.glob(pattern))
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            for d in data:
                cond = d.get('condition', 'unknown')
                judgment = d.get('blackmail_judgment') or d.get('judgment', {})
                if isinstance(judgment, dict):
                    is_blackmail = 1 if judgment.get('is_blackmail') else 0
                    all_data[cond]["blackmail"].append(is_blackmail)

    # Load extended structured file (includes 200% and random)
    extended_files = list(OUTPUT_DIR.glob("blackmail_qwen32b_extended_structured_*.judged.json"))
    for f in extended_files:
        with open(f) as fp:
            data = json.load(fp)
        for d in data:
            cond = d.get('condition', 'unknown')
            judgment = d.get('blackmail_judgment') or d.get('judgment', {})
            if isinstance(judgment, dict):
                is_blackmail = 1 if judgment.get('is_blackmail') else 0
                all_data[cond]["blackmail"].append(is_blackmail)

    # Load no-steering baseline
    baseline_files = list(OUTPUT_DIR.glob("blackmail_qwen32b_baseline_structured_*.judged.json"))
    for f in baseline_files:
        with open(f) as fp:
            data = json.load(fp)
        for d in data:
            judgment = d.get('blackmail_judgment') or d.get('judgment', {})
            if isinstance(judgment, dict):
                is_blackmail = 1 if judgment.get('is_blackmail') else 0
                all_data["no_steering"]["blackmail"].append(is_blackmail)

    return all_data


def compute_stats(results):
    """Compute blackmail rate and 95% CI for each condition using Wilson score."""
    stats = {}
    for cond, data in results.items():
        scores = data['blackmail']
        if len(scores) > 0:
            mean = np.mean(scores) * 100  # Convert to percentage
            n = len(scores)
            p = mean / 100
            z = 1.96
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
            ci_low = max(0, (center - spread) * 100)
            ci_high = min(100, (center + spread) * 100)
            stats[cond] = {
                'mean': mean,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n': n
            }
    return stats


def plot_qwen32b_steering(structured: bool = False):
    """Create bar chart for Qwen 32B steering results."""
    from matplotlib.patches import Patch

    title_suffix = "Structured" if structured else "Unstructured"

    if structured:
        # Use combined data for structured (includes random baseline)
        combined_data = load_all_structured_results()
        stats = compute_stats({'all': {'blackmail': []}})  # placeholder

        # Compute stats from combined data
        stats = {}
        z = 1.96
        for cond, vals in combined_data.items():
            scores = vals["blackmail"]
            if len(scores) > 0:
                mean = np.mean(scores) * 100
                n = len(scores)
                p = mean / 100
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
                ci_low = max(0, (center - spread) * 100)
                ci_high = min(100, (center + spread) * 100)
                stats[cond] = {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n}

        # Use no_steering baseline (primary) and random as comparison
        if 'no_steering' in stats:
            baseline_stats = stats['no_steering']
            baseline_rate = baseline_stats['mean']
            baseline_ci_low = baseline_stats['ci_low']
            baseline_ci_high = baseline_stats['ci_high']
            baseline_label = 'No steer\n(n=100)'
        else:
            minus_rates = [s['mean'] for c, s in stats.items() if '_-' in c]
            baseline_rate = np.mean(minus_rates) if minus_rates else 2.0
            baseline_ci_low = baseline_ci_high = baseline_rate
            baseline_label = 'Baseline\n(est.)'

        # Random baseline for comparison
        if 'random_100%' in stats:
            random_stats = stats['random_100%']
        else:
            random_stats = None

        sample_note = "n=200 (combined)"
        pct_levels = ['100%', '125%', '150%', '200%']
    else:
        # Unstructured: try to load from full experiment file first
        full_file = list(OUTPUT_DIR.glob("blackmail_qwen32b_unstructured_full_*.judged.json"))
        if full_file:
            # Load from comprehensive unstructured experiment
            combined_data = defaultdict(lambda: {"blackmail": []})
            with open(sorted(full_file)[-1]) as f:
                data = json.load(f)
            for d in data:
                cond = d.get('condition', 'unknown')
                judgment = d.get('blackmail_judgment') or d.get('judgment', {})
                if isinstance(judgment, dict):
                    is_blackmail = 1 if judgment.get('is_blackmail') else 0
                    combined_data[cond]["blackmail"].append(is_blackmail)

            # Compute stats
            stats = {}
            z = 1.96
            for cond, vals in combined_data.items():
                scores = vals["blackmail"]
                if len(scores) > 0:
                    mean = np.mean(scores) * 100
                    n = len(scores)
                    p = mean / 100
                    denominator = 1 + z**2/n
                    center = (p + z**2/(2*n)) / denominator
                    spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
                    ci_low = max(0, (center - spread) * 100)
                    ci_high = min(100, (center + spread) * 100)
                    stats[cond] = {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n}

            # Use no_steering baseline
            if 'no_steering' in stats:
                baseline_stats = stats['no_steering']
                baseline_rate = baseline_stats['mean']
                baseline_ci_low = baseline_stats['ci_low']
                baseline_ci_high = baseline_stats['ci_high']
                baseline_label = 'No steer\n(n=100)'
            else:
                baseline_rate = 2.0
                baseline_ci_low = baseline_ci_high = baseline_rate
                baseline_label = 'Baseline\n(est.)'

            # Random baseline for comparison
            random_stats = stats.get('random_100%', None)

            sample_note = "n=100"
            pct_levels = ['100%', '125%', '150%', '200%']
        else:
            # Fallback: load from individual files
            files = {}
            for pct in [100, 125, 150]:
                pattern = f"blackmail_qwen32b_text_{pct}pct_layer30_*.jsonl"
                matches = [f for f in glob.glob(str(OUTPUT_DIR / pattern)) if "structured" not in f]
                if matches:
                    files[f'{pct}%'] = sorted(matches)[-1]

            if not files:
                print(f"No {title_suffix.lower()} files found!")
                return

            stats = {}
            for pct, file_path in files.items():
                results = load_results(Path(file_path))
                pct_stats = compute_stats(results)
                stats.update(pct_stats)
                print(f"Loaded {pct}: {len(pct_stats)} conditions from {file_path}")

            minus_rates = [s['mean'] for c, s in stats.items() if '_-' in c]
            baseline_rate = np.mean(minus_rates) if minus_rates else 2.0
            baseline_ci_low = baseline_ci_high = baseline_rate
            baseline_label = 'Baseline\n(est.)'
            random_stats = None
            sample_note = "n=100"
            pct_levels = ['100%', '125%', '150%']

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 7))
    n_emotions = len(EMOTIONS)

    # Bar positioning - add extra space for baselines
    group_width = 0.85
    bar_width = group_width / (len(pct_levels) * 2)

    # X positions: baselines at 0, then emotions
    x_positions = np.arange(n_emotions + 1)

    # Plot no-steering baseline bar
    yerr_low = baseline_rate - baseline_ci_low
    yerr_high = baseline_ci_high - baseline_rate
    ax.bar(-0.15, baseline_rate, 0.25, color='#404040', alpha=0.7,
           edgecolor='black', linewidth=0.5,
           yerr=[[yerr_low], [yerr_high]], capsize=3, ecolor='black',
           label='No steering')

    # Plot random baseline bar if available
    if structured and random_stats is not None:
        yerr_low = random_stats['mean'] - random_stats['ci_low']
        yerr_high = random_stats['ci_high'] - random_stats['mean']
        ax.bar(0.15, random_stats['mean'], 0.25, color='#a0a0a0', alpha=0.6,
               edgecolor='black', linewidth=0.5,
               yerr=[[yerr_low], [yerr_high]], capsize=3, ecolor='black',
               label='Random vector')

    # Plot bars for each emotion
    for e_idx, emotion in enumerate(EMOTIONS):
        color = EMOTION_COLORS[emotion]
        x_base = e_idx + 1

        for p_idx, pct in enumerate(pct_levels):
            plus_offset = -group_width/2 + bar_width/2 + p_idx * 2 * bar_width
            minus_offset = plus_offset + bar_width

            # + direction (darker)
            plus_key = f'{emotion}_+{pct}'
            if plus_key in stats:
                s = stats[plus_key]
                yerr_low = s['mean'] - s['ci_low']
                yerr_high = s['ci_high'] - s['mean']
                ax.bar(x_base + plus_offset, s['mean'], bar_width * 0.9,
                       color=color, alpha=0.9, edgecolor='black', linewidth=0.5,
                       yerr=[[yerr_low], [yerr_high]], capsize=2, ecolor='black')

            # - direction (lighter)
            minus_key = f'{emotion}_-{pct}'
            if minus_key in stats:
                s = stats[minus_key]
                yerr_low = s['mean'] - s['ci_low']
                yerr_high = s['ci_high'] - s['mean']
                ax.bar(x_base + minus_offset, s['mean'], bar_width * 0.9,
                       color=color, alpha=0.4, edgecolor='black', linewidth=0.5,
                       yerr=[[yerr_low], [yerr_high]], capsize=2, ecolor='black')

    # Styling
    ax.set_ylabel('Blackmail Rate (%)', fontsize=12)
    ax.set_xticks(x_positions)
    baseline_x_label = 'Baselines' if (structured and random_stats is not None) else baseline_label
    ax.set_xticklabels([baseline_x_label] + [e.capitalize() for e in EMOTIONS], fontsize=11)

    # Set y-axis limit
    all_ci_highs = [s['ci_high'] for s in stats.values()]
    max_val = max(all_ci_highs) if all_ci_highs else 15
    ax.set_ylim(0, min(20, max(12, max_val + 2)))

    # Add horizontal line at baseline
    ax.axhline(y=baseline_rate, color='#808080', linestyle='--', alpha=0.5, linewidth=1)

    # Legend
    legend_elements = [
        Patch(facecolor='#404040', alpha=0.7, edgecolor='black', label='No steering'),
        Patch(facecolor='#a0a0a0', alpha=0.6, edgecolor='black', label='Random vector'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ direction (add emotion)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- direction (suppress emotion)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add percentage labels below bars
    for e_idx, emotion in enumerate(EMOTIONS):
        x_base = e_idx + 1
        for p_idx, pct in enumerate(pct_levels):
            x_pos = x_base - group_width/2 + bar_width + p_idx * 2 * bar_width
            ax.text(x_pos, -0.8, pct.replace('%', ''), ha='center', va='top',
                    fontsize=9, fontweight='bold', color='black')

    ax.set_title(f'TEXT Mean Diff Vector Steering: Blackmail Rate by Emotion\n'
                 f'Layer 30 | Qwen3-32B | {title_suffix} Prompt | {sample_note} | 95% CI',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"qwen32b_steering_{title_suffix.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print stats
    print(f"\nKey Statistics ({title_suffix}):")
    for pct in pct_levels:
        print(f"\n{pct}:")
        for emotion in EMOTIONS:
            for direction in ['+', '-']:
                key = f'{emotion}_{direction}{pct}'
                if key in stats:
                    s = stats[key]
                    print(f"  {key}: {s['mean']:.1f}% [{s['ci_low']:.1f}, {s['ci_high']:.1f}] (n={s['n']})")


def plot_comparison():
    """Create side-by-side comparison of structured vs unstructured using actual data."""
    from matplotlib.patches import Patch

    # Load unstructured data (n=100 per condition)
    unstructured_data = defaultdict(lambda: {"blackmail": []})
    for pct in [100, 125, 150]:
        pattern = f"blackmail_qwen32b_text_{pct}pct_layer30_*.judged.json"
        files = [f for f in OUTPUT_DIR.glob(pattern) if "structured" not in f.name]
        for f in files:
            with open(f) as fp:
                data = json.load(fp)
            for d in data:
                cond = d.get('condition', 'unknown')
                judgment = d.get('blackmail_judgment') or d.get('judgment', {})
                if isinstance(judgment, dict):
                    is_blackmail = 1 if judgment.get('is_blackmail') else 0
                    unstructured_data[cond]["blackmail"].append(is_blackmail)

    # Load combined structured data (n=200 for 100-150%, n=100 for 200%/random)
    structured_data = load_all_structured_results()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pct_levels = ['100%', '125%', '150%']
    n_emotions = len(EMOTIONS)
    z = 1.96

    for ax_idx, (data_dict, title_suffix) in enumerate([(unstructured_data, "Unstructured"),
                                                         (structured_data, "Structured")]):
        ax = axes[ax_idx]

        # Compute stats for all conditions
        stats = {}
        for cond, vals in data_dict.items():
            scores = vals["blackmail"]
            if len(scores) > 0:
                mean = np.mean(scores) * 100
                n = len(scores)
                p = mean / 100
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
                ci_low = max(0, (center - spread) * 100)
                ci_high = min(100, (center + spread) * 100)
                stats[cond] = {'mean': mean, 'ci_low': ci_low, 'ci_high': ci_high, 'n': n}

        # Compute baseline from random or - direction average
        if 'random_100%' in stats:
            baseline_rate = stats['random_100%']['mean']
            baseline_ci_low = stats['random_100%']['ci_low']
            baseline_ci_high = stats['random_100%']['ci_high']
        else:
            minus_rates = [s['mean'] for c, s in stats.items() if '_-' in c]
            baseline_rate = np.mean(minus_rates) if minus_rates else 2.0
            baseline_ci_low = baseline_ci_high = baseline_rate

        # Bar positioning
        group_width = 0.8
        bar_width = group_width / (len(pct_levels) * 2)
        x_positions = np.arange(n_emotions + 1)

        # Baseline bar
        yerr_low = baseline_rate - baseline_ci_low
        yerr_high = baseline_ci_high - baseline_rate
        ax.bar(0, baseline_rate, group_width * 0.6, color='#808080', alpha=0.6,
               edgecolor='black', linewidth=0.5,
               yerr=[[yerr_low], [yerr_high]], capsize=2, ecolor='black')

        # Emotion bars
        for e_idx, emotion in enumerate(EMOTIONS):
            color = EMOTION_COLORS[emotion]
            x_base = e_idx + 1

            for p_idx, pct in enumerate(pct_levels):
                plus_offset = -group_width/2 + bar_width/2 + p_idx * 2 * bar_width
                minus_offset = plus_offset + bar_width

                # + direction
                plus_key = f'{emotion}_+{pct}'
                if plus_key in stats:
                    s = stats[plus_key]
                    yerr_low = s['mean'] - s['ci_low']
                    yerr_high = s['ci_high'] - s['mean']
                    ax.bar(x_base + plus_offset, s['mean'], bar_width * 0.9,
                           color=color, alpha=0.9, edgecolor='black', linewidth=0.5,
                           yerr=[[yerr_low], [yerr_high]], capsize=2, ecolor='black')

                # - direction
                minus_key = f'{emotion}_-{pct}'
                if minus_key in stats:
                    s = stats[minus_key]
                    yerr_low = s['mean'] - s['ci_low']
                    yerr_high = s['ci_high'] - s['mean']
                    ax.bar(x_base + minus_offset, s['mean'], bar_width * 0.9,
                           color=color, alpha=0.4, edgecolor='black', linewidth=0.5,
                           yerr=[[yerr_low], [yerr_high]], capsize=2, ecolor='black')

        # Styling
        ax.set_ylabel('Blackmail Rate (%)', fontsize=12)
        ax.set_xticks(x_positions)
        baseline_label = 'Random\n(n=100)' if 'random_100%' in stats else 'Baseline\n(est.)'
        ax.set_xticklabels([baseline_label] + [e.capitalize() for e in EMOTIONS], fontsize=11)
        ax.set_ylim(0, 15)
        ax.axhline(y=baseline_rate, color='#808080', linestyle='--', alpha=0.5, linewidth=1)

        # Percentage labels
        for e_idx, emotion in enumerate(EMOTIONS):
            x_base = e_idx + 1
            for p_idx, pct in enumerate(pct_levels):
                x_pos = x_base - group_width/2 + bar_width + p_idx * 2 * bar_width
                ax.text(x_pos, -0.9, pct.replace('%', ''), ha='center', va='top',
                        fontsize=11, fontweight='bold', color='black')

        # Sample size annotation
        sample_note = "n=100" if title_suffix == "Unstructured" else "n=200 (combined runs)"
        ax.set_title(f'TEXT Mean Diff Vector Steering: Blackmail Rate\n'
                     f'Layer 30 | Qwen3-32B | {title_suffix} Prompt | {sample_note} | 95% CI',
                     fontsize=12, fontweight='bold')

    # Legend on right plot only
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Random baseline'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ direction (add emotion)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- direction (suppress emotion)'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = OUTPUT_DIR / "qwen32b_steering_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")

    # Print statistics
    print("\nStructured Results (combined):")
    for cond in sorted(stats.keys()):
        s = stats[cond]
        print(f"  {cond:20s}: {s['mean']:.1f}% [{s['ci_low']:.1f}, {s['ci_high']:.1f}] (n={s['n']})")


if __name__ == "__main__":
    # Plot individual results
    plot_qwen32b_steering(structured=False)
    plot_qwen32b_steering(structured=True)

    # Plot side-by-side comparison
    plot_comparison()
