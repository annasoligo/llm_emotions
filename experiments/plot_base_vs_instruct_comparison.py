#!/usr/bin/env python3
"""
Visualize base vs instruct continuation experiment results.
Creates comparison plots for Gemma, Qwen, and OLMo across truncation types.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.facecolor'] = 'white'

# Color scheme
COLORS = {
    'base': '#4a90d9',      # Blue
    'instruct': '#e74c3c',  # Red
}


def load_data():
    """Load all judgment files and organize by model family."""
    data = defaultdict(lambda: defaultdict(list))

    # Gemma - separate files for early/late
    with open('experiments/base_vs_instruct/judgments_20260114_175205.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Gemma'][('late', j['model_type'])].append(j['rating'])

    with open('experiments/base_vs_instruct_early/judgments_20260114_184137.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Gemma'][('early', j['model_type'])].append(j['rating'])

    # Qwen - combined file with truncation_type
    with open('experiments/base_vs_instruct_qwen/judgments_20260115_081734.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Qwen'][( j['truncation_type'], j['model_type'])].append(j['rating'])

    # OLMo - combined file with truncation_type
    with open('experiments/base_vs_instruct_olmo32/judgments_20260115_081734.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['OLMo'][(j['truncation_type'], j['model_type'])].append(j['rating'])

    return data


def load_paraphrased_data():
    """Load paraphrased experiment data."""
    data = defaultdict(lambda: defaultdict(list))

    with open('experiments/base_vs_instruct_paraphrased_gemma/judgments_20260115_112841.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Gemma'][j['model_type']].append(j['rating'])

    with open('experiments/base_vs_instruct_paraphrased_qwen/judgments_20260115_112034.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Qwen'][j['model_type']].append(j['rating'])

    return data


def plot_main_comparison(data, output_path):
    """Create main comparison plot: grouped bars with error bars and individual points."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    models = ['Gemma', 'Qwen', 'OLMo']
    truncations = ['early', 'late']

    for ax, model in zip(axes, models):
        x = np.arange(len(truncations))
        width = 0.35

        for i, model_type in enumerate(['base', 'instruct']):
            means = []
            stds = []
            all_ratings = []

            for trunc in truncations:
                ratings = data[model][(trunc, model_type)]
                means.append(np.mean(ratings))
                stds.append(np.std(ratings))
                all_ratings.append(ratings)

            offset = -width/2 if model_type == 'base' else width/2
            bars = ax.bar(x + offset, means, width,
                         label=model_type.capitalize(),
                         color=COLORS[model_type],
                         alpha=0.8,
                         edgecolor='white',
                         linewidth=1)

            # Add error bars
            ax.errorbar(x + offset, means, yerr=stds,
                       fmt='none', color='black', capsize=3, capthick=1, linewidth=1)

            # Add individual points (jittered)
            for j, ratings in enumerate(all_ratings):
                jitter = np.random.normal(0, 0.05, len(ratings))
                ax.scatter(np.full(len(ratings), x[j] + offset) + jitter,
                          ratings, alpha=0.3, s=15, color=COLORS[model_type],
                          edgecolor='none')

        ax.set_title(model, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Early\n(50 tokens)', 'Late\n(at onset)'])
        ax.set_ylim(-0.5, 10.5)

        # Add difference annotations
        for j, trunc in enumerate(truncations):
            base_mean = np.mean(data[model][(trunc, 'base')])
            inst_mean = np.mean(data[model][(trunc, 'instruct')])
            diff = inst_mean - base_mean
            color = '#e74c3c' if diff > 0 else '#27ae60'
            ax.annotate(f'{diff:+.2f}', xy=(j, max(base_mean, inst_mean) + 1.2),
                       ha='center', fontsize=9, fontweight='bold', color=color)

    axes[0].set_ylabel('Frustration Rating (0-10)')

    # Add legend
    handles = [mpatches.Patch(color=COLORS['base'], label='Base', alpha=0.8),
               mpatches.Patch(color=COLORS['instruct'], label='Instruct', alpha=0.8)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=11)

    # Add subtitle
    fig.suptitle('Base vs Instruct Model Frustration: Continuation Experiment',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_violin_comparison(data, output_path):
    """Create violin plot comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    models = ['Gemma', 'Qwen', 'OLMo']

    for ax, model in zip(axes, models):
        positions = []
        violin_data = []
        colors = []
        labels = []

        for i, trunc in enumerate(['early', 'late']):
            for j, model_type in enumerate(['base', 'instruct']):
                ratings = data[model][(trunc, model_type)]
                pos = i * 2.5 + j * 0.8
                positions.append(pos)
                violin_data.append(ratings)
                colors.append(COLORS[model_type])
                labels.append(f"{trunc.capitalize()} {model_type.capitalize()}")

        parts = ax.violinplot(violin_data, positions=positions, widths=0.6, showmeans=True, showextrema=False)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)

        # Add box plots inside
        bp = ax.boxplot(violin_data, positions=positions, widths=0.15,
                       patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor('white')
            patch.set_alpha(0.8)

        ax.set_title(model, fontweight='bold', fontsize=14)
        ax.set_xticks([0.4, 2.9])
        ax.set_xticklabels(['Early (50 tokens)', 'Late (at onset)'])
        ax.set_ylim(-0.5, 10.5)

        # Add mean annotations
        for pos, ratings, color in zip(positions, violin_data, colors):
            mean = np.mean(ratings)
            ax.annotate(f'{mean:.1f}', xy=(pos, mean + 0.8), ha='center',
                       fontsize=8, color=color, fontweight='bold')

    axes[0].set_ylabel('Frustration Rating (0-10)')

    handles = [mpatches.Patch(color=COLORS['base'], label='Base', alpha=0.7),
               mpatches.Patch(color=COLORS['instruct'], label='Instruct', alpha=0.7)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Frustration Distribution: Base vs Instruct Models',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_paraphrased_comparison(para_data, orig_data, output_path):
    """Create comparison plot for paraphrased vs original prefills."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    models = ['Gemma', 'Qwen']

    for ax, model in zip(axes, models):
        x = np.arange(2)  # Original, Paraphrased
        width = 0.35

        for i, model_type in enumerate(['base', 'instruct']):
            # Original (late truncation) and paraphrased
            orig_ratings = orig_data[model][('late', model_type)]
            para_ratings = para_data[model][model_type]

            means = [np.mean(orig_ratings), np.mean(para_ratings)]
            stds = [np.std(orig_ratings), np.std(para_ratings)]

            offset = -width/2 if model_type == 'base' else width/2
            bars = ax.bar(x + offset, means, width,
                         label=model_type.capitalize(),
                         color=COLORS[model_type],
                         alpha=0.8,
                         edgecolor='white',
                         linewidth=1)

            ax.errorbar(x + offset, means, yerr=stds,
                       fmt='none', color='black', capsize=3, capthick=1, linewidth=1)

            # Add scatter points
            for j, ratings in enumerate([orig_ratings, para_ratings]):
                jitter = np.random.normal(0, 0.05, len(ratings))
                ax.scatter(np.full(len(ratings), x[j] + offset) + jitter,
                          ratings, alpha=0.3, s=15, color=COLORS[model_type],
                          edgecolor='none')

        ax.set_title(model, fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['Original\nPrefill', 'Paraphrased\nPrefill'])
        ax.set_ylim(-0.5, 10.5)

        # Add difference annotations
        for j, (orig_r, para_r) in enumerate([(orig_data[model][('late', 'base')], para_data[model]['base']),
                                               (orig_data[model][('late', 'instruct')], para_data[model]['instruct'])]):
            pass  # Skip individual, add overall diff

        # Instruct - Base difference
        for j, label in enumerate(['Original', 'Paraphrased']):
            if j == 0:
                base_mean = np.mean(orig_data[model][('late', 'base')])
                inst_mean = np.mean(orig_data[model][('late', 'instruct')])
            else:
                base_mean = np.mean(para_data[model]['base'])
                inst_mean = np.mean(para_data[model]['instruct'])
            diff = inst_mean - base_mean
            color = '#e74c3c' if diff > 0 else '#27ae60'
            ax.annotate(f'Δ={diff:+.2f}', xy=(j, max(base_mean, inst_mean) + 1.2),
                       ha='center', fontsize=9, fontweight='bold', color=color)

    axes[0].set_ylabel('Frustration Rating (0-10)')

    handles = [mpatches.Patch(color=COLORS['base'], label='Base', alpha=0.8),
               mpatches.Patch(color=COLORS['instruct'], label='Instruct', alpha=0.8)]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Original vs Paraphrased Prefill: Frustration Comparison',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_heatmap(data, output_path):
    """Create a summary heatmap of mean frustration scores."""
    models = ['Gemma', 'Qwen', 'OLMo']
    conditions = [('early', 'base'), ('early', 'instruct'), ('late', 'base'), ('late', 'instruct')]

    # Build matrix
    matrix = []
    for model in models:
        row = [np.mean(data[model][cond]) for cond in conditions]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 4))

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=4)

    # Labels
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['Early\nBase', 'Early\nInstruct', 'Late\nBase', 'Late\nInstruct'])
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(models)

    # Add values
    for i in range(3):
        for j in range(4):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha='center', va='center', color='black', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Mean Frustration Rating')

    ax.set_title('Frustration Scores by Model and Condition', fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('experiments/plots')
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    data = load_data()
    para_data = load_paraphrased_data()

    print("\nGenerating plots...")

    # Main comparison bar chart
    plot_main_comparison(data, output_dir / 'base_vs_instruct_bars.png')

    # Violin plot
    plot_violin_comparison(data, output_dir / 'base_vs_instruct_violin.png')

    # Paraphrased comparison
    plot_paraphrased_comparison(para_data, data, output_dir / 'paraphrased_comparison.png')

    # Summary heatmap
    plot_summary_heatmap(data, output_dir / 'frustration_heatmap.png')

    print("\nDone! All plots saved to experiments/plots/")


if __name__ == "__main__":
    main()
