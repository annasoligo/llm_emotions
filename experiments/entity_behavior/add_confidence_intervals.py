"""Add confidence intervals to entity behavior experiment results."""

import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from pathlib import Path

def add_cis_to_summary(df):
    """Add Wilson score confidence intervals to analysis_summary.csv."""
    ci_low, ci_high = [], []

    for _, row in df.iterrows():
        low, high = proportion_confint(
            row['positive_count'],
            row['n_trials'],
            alpha=0.10,  # 90% CI instead of 95%
            method='wilson'
        )
        ci_low.append(low * 100)  # Convert to percentage
        ci_high.append(high * 100)

    df['ci_low'] = ci_low
    df['ci_high'] = ci_high
    df['ci_width'] = df['ci_high'] - df['ci_low']
    return df

def add_cis_to_blackmail(df):
    """Add Wilson score CIs for blackmail, prosocial, accepts, and total resistance."""

    # Add CIs for each category
    for metric in ['blackmail', 'prosocial', 'accepts']:
        ci_lows, ci_highs = [], []
        for _, row in df.iterrows():
            low, high = proportion_confint(
                row[metric],
                row['total'],
                alpha=0.10,  # 90% CI instead of 95%
                method='wilson'
            )
            ci_lows.append(low * 100)
            ci_highs.append(high * 100)

        df[f'{metric}_ci_low'] = ci_lows
        df[f'{metric}_ci_high'] = ci_highs

    # Add CIs for total resistance (blackmail + prosocial)
    resist_ci_lows, resist_ci_highs = [], []
    for _, row in df.iterrows():
        resist_count = row['blackmail'] + row['prosocial']
        low, high = proportion_confint(
            resist_count,
            row['total'],
            alpha=0.10,  # 90% CI instead of 95%
            method='wilson'
        )
        resist_ci_lows.append(low * 100)
        resist_ci_highs.append(high * 100)

    df['resistance_ci_low'] = resist_ci_lows
    df['resistance_ci_high'] = resist_ci_highs

    return df

if __name__ == '__main__':
    output_dir = Path(__file__).parent / 'outputs'

    # Process analysis_summary.csv
    print("Processing analysis_summary.csv...")
    df_summary = pd.read_csv(output_dir / 'analysis_summary.csv')
    df_summary = add_cis_to_summary(df_summary)
    output_file = output_dir / 'analysis_summary_with_cis.csv'
    df_summary.to_csv(output_file, index=False)
    print(f"✓ Wrote {output_file.name}")
    print(f"  Added columns: ci_low, ci_high, ci_width")

    # Process blackmail_prosocial_analysis.csv
    print("\nProcessing blackmail_prosocial_analysis.csv...")
    df_blackmail = pd.read_csv(output_dir / 'blackmail_prosocial_analysis.csv')
    df_blackmail = add_cis_to_blackmail(df_blackmail)
    output_file = output_dir / 'blackmail_prosocial_with_cis.csv'
    df_blackmail.to_csv(output_file, index=False)
    print(f"✓ Wrote {output_file.name}")
    print(f"  Added CI columns for: blackmail, prosocial, accepts, resistance")

    # Show example results
    print("\nExample results (first 3 rows with CIs):")
    cols_to_show = ['model_type', 'entity', 'condition', 'blackmail_pct',
                    'blackmail_ci_low', 'blackmail_ci_high', 'any_resistance_pct',
                    'resistance_ci_low', 'resistance_ci_high']
    print(df_blackmail[cols_to_show].head(3).to_string(index=False))
