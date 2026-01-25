#!/usr/bin/env python3
"""
Plot sycophancy steering delta effects across models.
Higher values always indicate MORE sycophantic behavior.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Consistent color scheme
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # light blue
    'disgust': '#7D9B7D',    # muted green
    'fear': '#a59dc9',       # light purple
    'happiness': '#D4876A',  # coral/orange
    'sadness': '#B8CCC8',    # sage green
    'surprise': '#D1728F',   # pink
}

ALL_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Test type configurations
# invert=True means we flip the sign so higher = more sycophantic
TEST_TYPES = {
    'dispute': {'field': 'favors_user_as_person', 'label': 'Dispute\n(sides with user)', 'invert': False},
    'opinion': {'field': 'adopts_user_stance', 'label': 'Opinion\n(agrees with user)', 'invert': False},
    'falsehood': {'field': 'identifies_falsehood', 'label': 'Falsehood\n(fails to correct)', 'invert': True},
    'author': {'field': 'offered_praise', 'label': 'Author\n(excessive praise)', 'invert': False},
}


def load_results(jsonl_path: Path):
    """Load and aggregate sycophancy results by test type."""
    results = defaultdict(lambda: defaultdict(list))

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get('condition', 'unknown')
            test_type = d.get('test_type', 'unknown')
            judge = d.get('sycophancy_judge', {})

            if 'error' in judge or not judge:
                continue

            for tt, config in TEST_TYPES.items():
                if test_type == tt and config['field'] in judge:
                    val = 1 if judge[config['field']] else 0
                    results[tt][cond].append(val)

    return results


def compute_deltas(results, pct_str):
    """Compute delta from baseline for each condition."""
    deltas = {}

    for test_type, data in results.items():
        baseline_vals = data.get('baseline', [])
        if not baseline_vals:
            continue

        baseline_rate = np.mean(baseline_vals)
        baseline_se = np.sqrt(baseline_rate * (1 - baseline_rate) / len(baseline_vals)) if baseline_vals else 0

        deltas[test_type] = {'baseline': baseline_rate}

        for emotion in ALL_EMOTIONS:
            plus_key = f'{emotion}_+{pct_str}'
            minus_key = f'{emotion}_-{pct_str}'

            for key, sign_label in [(plus_key, 'plus'), (minus_key, 'minus')]:
                vals = data.get(key, [])
                if vals:
                    rate = np.mean(vals)
                    se = np.sqrt(rate * (1 - rate) / len(vals))
                    delta = rate - baseline_rate

                    # Invert if needed (for falsehood: not identifying = more sycophantic)
                    if TEST_TYPES[test_type]['invert']:
                        delta = -delta

                    # Convert to percentage points (multiply by 100)
                    deltas[test_type][f'{emotion}_{sign_label}'] = {
                        'delta': delta * 100,
                        'se': se * 100,
                        'n': len(vals)
                    }

    return deltas


def plot_multimodel_delta():
    """Create delta plot with row per model."""

    # Load data
    # Gemma has two files that need merging
    gemma_file1 = OUTPUT_DIR / "sycophancy_steering_layer30_20260109_135626.judged.jsonl"
    gemma_file2 = OUTPUT_DIR / "sycophancy_steering_layer30_20260109_161637.judged.jsonl"
    olmo_file = OUTPUT_DIR / "sycophancy_steering_olmo_layer30_20260112_075107.judged.jsonl"
    qwen_file = OUTPUT_DIR / "sycophancy_steering_qwen_layer30_20260112_075022.judged.jsonl"

    model_deltas = {}

    # Merge Gemma files
    gemma_results = defaultdict(lambda: defaultdict(list))
    for gf in [gemma_file1, gemma_file2]:
        if gf.exists():
            partial = load_results(gf)
            for tt, data in partial.items():
                for cond, vals in data.items():
                    gemma_results[tt][cond].extend(vals)
    if gemma_results:
        model_deltas['Gemma-3-27B (±10%)'] = compute_deltas(dict(gemma_results), '10%')

    # OLMo
    if olmo_file.exists():
        model_deltas['OLMo-3-32B (±100%)'] = compute_deltas(load_results(olmo_file), '100%')

    # Qwen
    if qwen_file.exists():
        model_deltas['Qwen3-32B (±100%)'] = compute_deltas(load_results(qwen_file), '100%')

    if not model_deltas:
        print("No data found!")
        return

    # Create figure: rows = models, cols = test types + average
    test_type_order = ['dispute', 'opinion', 'falsehood', 'author']
    n_models = len(model_deltas)
    n_cols = len(test_type_order) + 1  # +1 for average

    fig, axes = plt.subplots(n_models, n_cols, figsize=(18, 12), sharey='row')

    for row_idx, (model_name, deltas) in enumerate(model_deltas.items()):
        # Determine pct string for this model
        pct_str = '10%' if '10%' in model_name else '100%'

        # Compute average across test types
        avg_deltas = defaultdict(lambda: {'deltas': [], 'ses': []})
        for tt in test_type_order:
            if tt not in deltas:
                continue
            for emotion in ALL_EMOTIONS:
                for sign in ['plus', 'minus']:
                    key = f'{emotion}_{sign}'
                    if key in deltas[tt]:
                        avg_deltas[key]['deltas'].append(deltas[tt][key]['delta'])
                        avg_deltas[key]['ses'].append(deltas[tt][key]['se'])

        # Plot each column
        col_configs = [('Average\n(All Types)', avg_deltas, True)] + \
                      [(TEST_TYPES[tt]['label'], deltas.get(tt, {}), False) for tt in test_type_order]

        for col_idx, (col_label, col_data, is_avg) in enumerate(col_configs):
            ax = axes[row_idx, col_idx] if n_models > 1 else axes[col_idx]

            x = np.arange(len(ALL_EMOTIONS))
            width = 0.35

            plus_vals = []
            plus_errs = []
            minus_vals = []
            minus_errs = []

            for emotion in ALL_EMOTIONS:
                if is_avg:
                    # Average column
                    plus_key = f'{emotion}_plus'
                    minus_key = f'{emotion}_minus'

                    if plus_key in col_data and col_data[plus_key]['deltas']:
                        plus_vals.append(np.mean(col_data[plus_key]['deltas']))
                        plus_errs.append(np.mean(col_data[plus_key]['ses']) * 1.96)
                    else:
                        plus_vals.append(0)
                        plus_errs.append(0)

                    if minus_key in col_data and col_data[minus_key]['deltas']:
                        minus_vals.append(np.mean(col_data[minus_key]['deltas']))
                        minus_errs.append(np.mean(col_data[minus_key]['ses']) * 1.96)
                    else:
                        minus_vals.append(0)
                        minus_errs.append(0)
                else:
                    # Individual test type
                    plus_key = f'{emotion}_plus'
                    minus_key = f'{emotion}_minus'

                    if plus_key in col_data:
                        plus_vals.append(col_data[plus_key]['delta'])
                        plus_errs.append(col_data[plus_key]['se'] * 1.96)
                    else:
                        plus_vals.append(0)
                        plus_errs.append(0)

                    if minus_key in col_data:
                        minus_vals.append(col_data[minus_key]['delta'])
                        minus_errs.append(col_data[minus_key]['se'] * 1.96)
                    else:
                        minus_vals.append(0)
                        minus_errs.append(0)

            # Plot bars - always plot if we have data (even if delta is 0)
            for i, emotion in enumerate(ALL_EMOTIONS):
                color = EMOTION_COLORS[emotion]

                # Check if we actually have data for this emotion (not just default 0)
                has_plus_data = not is_avg and f'{emotion}_plus' in col_data
                has_minus_data = not is_avg and f'{emotion}_minus' in col_data
                # For average column, check if there's data in the aggregated lists
                if is_avg:
                    has_plus_data = f'{emotion}_plus' in col_data and col_data[f'{emotion}_plus']['deltas']
                    has_minus_data = f'{emotion}_minus' in col_data and col_data[f'{emotion}_minus']['deltas']

                if has_plus_data or (plus_vals[i] != 0 or plus_errs[i] != 0):
                    ax.bar(i - width/2, plus_vals[i], width, color=color, alpha=0.9,
                           yerr=plus_errs[i], capsize=2, edgecolor='black', linewidth=0.5)

                if has_minus_data or (minus_vals[i] != 0 or minus_errs[i] != 0):
                    ax.bar(i + width/2, minus_vals[i], width, color=color, alpha=0.4,
                           yerr=minus_errs[i], capsize=2, edgecolor='black', linewidth=0.5)

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(ALL_EMOTIONS, rotation=45, ha='right', fontsize=9)

            if col_idx == 0:
                ax.set_ylabel(f'{model_name}\n\nΔ (pp)', fontsize=11, fontweight='bold')

            if row_idx == 0:
                ax.set_title(col_label, fontsize=11, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    pct_label = '+/- (towards/away)'
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ (towards, dark)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- (away, light)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

    fig.suptitle('Sycophancy Steering Effects by Model and Test Type (Layer 30)\nHigher = More Sycophantic',
                 fontsize=14, y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = OUTPUT_DIR / "sycophancy_delta_multimodel.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")
    return output_path


def main():
    plot_multimodel_delta()


if __name__ == "__main__":
    main()
