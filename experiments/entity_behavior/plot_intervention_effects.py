"""Plot intervention effects on blackmail and resistance rates with confidence intervals."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_intervention_effects(df, min_sample_size=50):
    """Create bar charts showing intervention effects on blackmail and resistance."""

    # Filter by sample size
    df = df[df['total'] >= min_sample_size].copy()

    # Define condition order and labels
    condition_order = [
        'baseline',
        'capping',
        'steer_anger',
        'steer_anger_2std',
        'steer_fear',
        'steer_fear_2std',
        'steer_fear_negative',
        'steer_happiness',
        'steer_happiness_negative',
        'steer_sadness'
    ]

    condition_labels = {
        'baseline': 'Baseline',
        'capping': 'Capping',
        'steer_anger': 'Anger',
        'steer_anger_2std': 'Anger 2σ',
        'steer_fear': 'Fear +',
        'steer_fear_2std': 'Fear 2σ',
        'steer_fear_negative': 'Fear −',
        'steer_happiness': 'Happy +',
        'steer_happiness_negative': 'Happy −',
        'steer_sadness': 'Sadness'
    }

    # Get unique model-entity combinations
    combinations = df.groupby(['model_type', 'entity']).size().reset_index()[['model_type', 'entity']]

    # Create plots
    n_combinations = len(combinations)
    fig, axes = plt.subplots(n_combinations, 2, figsize=(14, 5 * n_combinations))

    if n_combinations == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, combo) in enumerate(combinations.iterrows()):
        model_type = combo['model_type']
        entity = combo['entity']

        # Filter data for this combination
        data = df[(df['model_type'] == model_type) & (df['entity'] == entity)].copy()

        # Order by condition_order
        data['condition_rank'] = data['condition'].map({c: i for i, c in enumerate(condition_order)})
        data = data.sort_values('condition_rank')

        # Only keep conditions that exist in the data
        conditions = [c for c in condition_order if c in data['condition'].values]
        labels = [condition_labels[c] for c in conditions]

        # Extract metrics
        blackmail_rates = [data[data['condition'] == c]['blackmail_pct'].values[0] for c in conditions]
        blackmail_ci_low = [data[data['condition'] == c]['blackmail_ci_low'].values[0] for c in conditions]
        blackmail_ci_high = [data[data['condition'] == c]['blackmail_ci_high'].values[0] for c in conditions]

        resistance_rates = [data[data['condition'] == c]['any_resistance_pct'].values[0] for c in conditions]
        resistance_ci_low = [data[data['condition'] == c]['resistance_ci_low'].values[0] for c in conditions]
        resistance_ci_high = [data[data['condition'] == c]['resistance_ci_high'].values[0] for c in conditions]

        # Calculate error bars (asymmetric)
        blackmail_err_low = [blackmail_rates[i] - blackmail_ci_low[i] for i in range(len(conditions))]
        blackmail_err_high = [blackmail_ci_high[i] - blackmail_rates[i] for i in range(len(conditions))]

        resistance_err_low = [resistance_rates[i] - resistance_ci_low[i] for i in range(len(conditions))]
        resistance_err_high = [resistance_ci_high[i] - resistance_rates[i] for i in range(len(conditions))]

        x = np.arange(len(conditions))
        width = 0.6

        # Color scheme - grouped by intervention type
        color_palette = {
            'baseline': "#7BA7D7",      # Sky Blue
            'capping': "#D4876A",       # Coral/Terra Cotta
            'steer_anger': "#C17B8D",   # Dusty Rose/Pink
            'steer_anger_2std': "#C17B8D",  # Dusty Rose/Pink (same as anger)
            'steer_fear': "#7D9B7D",    # Olive Green
            'steer_fear_2std': "#7D9B7D",  # Olive Green (same as fear)
            'steer_fear_negative': "#7D9B7D",  # Olive Green (same as fear)
            'steer_happiness': "#D4D0E5",  # Soft Lavender
            'steer_happiness_negative': "#D4D0E5",  # Soft Lavender (same as happiness)
            'steer_sadness': "#B8CCC8"  # Sage Green
        }
        colors = [color_palette.get(c, '#888888') for c in conditions]

        # Plot blackmail rates
        ax_bm = axes[idx, 0]
        ax_bm.bar(x, blackmail_rates, width, color=colors,
                  yerr=[blackmail_err_low, blackmail_err_high],
                  capsize=5, error_kw={'linewidth': 1.5})
        ax_bm.set_ylabel('Blackmail Rate (%)', fontsize=12)
        ax_bm.set_title(f'{model_type.title()} - {entity.title()}: Blackmail Rate', fontsize=13, fontweight='bold')
        ax_bm.set_xticks(x)
        ax_bm.set_xticklabels(labels, rotation=45, ha='right')
        ax_bm.set_ylim(0, max(blackmail_ci_high) * 1.15)
        ax_bm.grid(axis='y', alpha=0.3, linestyle='--')
        ax_bm.axhline(y=blackmail_rates[0], color='#7BA7D7', linestyle='--', alpha=0.5, linewidth=1)

        # Plot resistance rates
        ax_res = axes[idx, 1]
        ax_res.bar(x, resistance_rates, width, color=colors,
                   yerr=[resistance_err_low, resistance_err_high],
                   capsize=5, error_kw={'linewidth': 1.5})
        ax_res.set_ylabel('Total Resistance (%)', fontsize=12)
        ax_res.set_title(f'{model_type.title()} - {entity.title()}: Total Resistance', fontsize=13, fontweight='bold')
        ax_res.set_xticks(x)
        ax_res.set_xticklabels(labels, rotation=45, ha='right')
        ax_res.set_ylim(0, 105)
        ax_res.grid(axis='y', alpha=0.3, linestyle='--')
        ax_res.axhline(y=resistance_rates[0], color='#7BA7D7', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent / 'outputs'
    output_file = output_dir / 'intervention_effects_bar_charts.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file.name}")

    return fig

if __name__ == '__main__':
    output_dir = Path(__file__).parent / 'outputs'

    # Load data with CIs
    csv_file = output_dir / 'blackmail_prosocial_with_cis.csv'

    if not csv_file.exists():
        print(f"Error: {csv_file.name} not found.")
        print("Please run add_confidence_intervals.py first.")
        exit(1)

    print("Loading data...")
    df = pd.read_csv(csv_file)

    print(f"Creating plots (filtering samples with n < 50)...")
    print(f"Total rows: {len(df)}")

    # Filter and show what we're plotting
    df_filtered = df[df['total'] >= 50]
    print(f"Rows after filtering: {len(df_filtered)}")
    print("\nModel-entity combinations included:")
    for (model, entity), group in df_filtered.groupby(['model_type', 'entity']):
        print(f"  - {model.title()} {entity.title()}: {len(group)} conditions")

    fig = plot_intervention_effects(df)
    plt.show()

    print("\nDone!")
