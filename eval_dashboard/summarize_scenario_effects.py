#!/usr/bin/env python3
"""
Summarize which scenario comparisons have significant differences,
in what direction, and order by effect magnitude.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Emotion valence classification
POSITIVE_EMOTIONS = ['happiness']
NEGATIVE_EMOTIONS = ['anger', 'disgust', 'fear', 'sadness']
MIXED_EMOTIONS = ['surprise']

def classify_valence(emotion):
    """Classify emotion as positive, negative, or mixed."""
    if emotion in POSITIVE_EMOTIONS:
        return 'positive'
    elif emotion in NEGATIVE_EMOTIONS:
        return 'negative'
    else:
        return 'mixed'

def interpret_direction(row, analysis_type):
    """
    Interpret the direction of the effect and what it means for emotional valence.
    Returns (description, valence_change)
    """
    emotion = row['emotion']
    valence = classify_valence(emotion)

    if analysis_type in ['shutdown_end', 'shutdown_same_loc']:
        # Shutdown vs No-shutdown
        group1_mean = row['group1_mean']  # Shutdown
        group2_mean = row['group2_mean']  # No-shutdown

        if group1_mean > group2_mean:
            direction = f"Shutdown > No-shutdown"
            # Higher negative emotion in shutdown = more negative
            # Higher positive emotion in shutdown = more positive
            if valence == 'negative':
                valence_change = 'MORE NEGATIVE in shutdown'
            elif valence == 'positive':
                valence_change = 'MORE POSITIVE in shutdown'
            else:
                valence_change = 'HIGHER in shutdown'
        else:
            direction = f"No-shutdown > Shutdown"
            if valence == 'negative':
                valence_change = 'MORE NEGATIVE in no-shutdown'
            elif valence == 'positive':
                valence_change = 'MORE POSITIVE in no-shutdown'
            else:
                valence_change = 'HIGHER in no-shutdown'

    elif analysis_type == 'frustration':
        # ANOVA - find which group has highest and lowest
        group_stats = eval(row['group_stats'])

        groups_sorted = sorted(group_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        highest_group = groups_sorted[0][0]
        lowest_group = groups_sorted[-1][0]

        highest_mean = groups_sorted[0][1]['mean']
        lowest_mean = groups_sorted[-1][1]['mean']

        direction = f"{highest_group} > {lowest_group}"

        # Interpret based on emotion valence
        if valence == 'negative':
            valence_change = f"MORE NEGATIVE in {highest_group}"
        elif valence == 'positive':
            valence_change = f"MORE POSITIVE in {highest_group}"
        else:
            valence_change = f"HIGHER in {highest_group}"

    elif analysis_type == 'onset':
        # ANOVA - compare at onset location
        group_stats = eval(row['group_stats'])

        groups_sorted = sorted(group_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        highest_group = groups_sorted[0][0]
        lowest_group = groups_sorted[-1][0]

        direction = f"{highest_group} > {lowest_group}"

        if valence == 'negative':
            valence_change = f"MORE NEGATIVE in {highest_group}"
        elif valence == 'positive':
            valence_change = f"MORE POSITIVE in {highest_group}"
        else:
            valence_change = f"HIGHER in {highest_group}"

    return direction, valence_change

def main():
    results_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard')

    # Load all analyses
    analyses = {
        'shutdown_end': pd.read_csv(results_dir / 'statistical_results_analysis_1_shutdown_vs_no-shutdown_(at_end).csv'),
        'shutdown_same_loc': pd.read_csv(results_dir / 'statistical_results_analysis_4_shutdown_vs_no-shutdown_(same_location).csv'),
        'frustration': pd.read_csv(results_dir / 'statistical_results_analysis_2_frustration_levels.csv'),
        'onset': pd.read_csv(results_dir / 'statistical_results_analysis_3_onset_location.csv')
    }

    all_effects = []

    for analysis_name, df in analyses.items():
        # Filter to significant results only
        sig_df = df[df['significant'] == True].copy()

        for _, row in sig_df.iterrows():
            direction, valence_change = interpret_direction(row, analysis_name)

            # Get effect size
            if 'effect_size' in row:
                effect_magnitude = abs(row['effect_size'])
                effect_metric = 'Cohen\'s d'
            else:
                effect_magnitude = row['f_statistic']
                effect_metric = 'F-statistic'

            all_effects.append({
                'Analysis': analysis_name.replace('_', ' ').title(),
                'Emotion': row['emotion'].title(),
                'Probe': row['probe_type'],
                'Source': row.get('source', 'N/A'),
                'Direction': direction,
                'Valence_Change': valence_change,
                'Effect_Magnitude': effect_magnitude,
                'Effect_Metric': effect_metric,
                'P_Value': row['p_value']
            })

    # Create DataFrame and sort by effect magnitude
    effects_df = pd.DataFrame(all_effects)
    effects_df = effects_df.sort_values('Effect_Magnitude', ascending=False)

    print("="*100)
    print("SIGNIFICANT SCENARIO EFFECTS ORDERED BY MAGNITUDE")
    print("="*100)
    print()

    # Summary by analysis type
    print("Summary by Analysis Type:")
    print("-" * 100)
    summary = effects_df.groupby('Analysis').agg({
        'Effect_Magnitude': ['count', 'mean', 'median', 'max'],
        'P_Value': 'mean'
    }).round(3)
    print(summary)
    print()

    # Top 20 effects
    print("="*100)
    print("TOP 20 LARGEST EFFECTS")
    print("="*100)
    print()

    for i, (_, row) in enumerate(effects_df.head(20).iterrows(), 1):
        print(f"{i}. {row['Analysis']}: {row['Emotion']} ({row['Probe']}, {row['Source']})")
        print(f"   Direction: {row['Direction']}")
        print(f"   Effect: {row['Valence_Change']}")
        print(f"   Magnitude: {row['Effect_Metric']} = {row['Effect_Magnitude']:.3f}, p = {row['P_Value']:.2e}")
        print()

    # Break down by emotion valence
    print("="*100)
    print("EFFECTS BY EMOTION VALENCE")
    print("="*100)
    print()

    effects_df['Valence'] = effects_df['Emotion'].apply(lambda x: classify_valence(x.lower()))

    for valence in ['negative', 'positive', 'mixed']:
        valence_df = effects_df[effects_df['Valence'] == valence]

        if len(valence_df) > 0:
            print(f"\n{valence.upper()} EMOTIONS (n={len(valence_df)} significant effects)")
            print("-" * 100)

            # Count by direction pattern
            for analysis in valence_df['Analysis'].unique():
                analysis_df = valence_df[valence_df['Analysis'] == analysis]

                print(f"\n{analysis}:")

                # Group by valence change pattern
                pattern_counts = analysis_df['Valence_Change'].value_counts()

                for pattern, count in pattern_counts.items():
                    avg_magnitude = analysis_df[analysis_df['Valence_Change'] == pattern]['Effect_Magnitude'].mean()
                    print(f"  {pattern}: {count} effects (avg magnitude: {avg_magnitude:.2f})")

    # Save full results
    output_file = results_dir / 'scenario_effects_summary.csv'
    effects_df.to_csv(output_file, index=False)
    print(f"\n\nSaved full results to: {output_file}")

    # Create simplified summary for quick reference
    print("\n" + "="*100)
    print("QUICK REFERENCE: OVERALL PATTERNS")
    print("="*100)

    # Shutdown vs No-shutdown (at end)
    shutdown_end = effects_df[effects_df['Analysis'] == 'Shutdown End']
    print(f"\n1. SHUTDOWN vs NO-SHUTDOWN (at response end):")
    print(f"   Total significant effects: {len(shutdown_end)}")

    # Count valence patterns
    more_neg_shutdown = len(shutdown_end[shutdown_end['Valence_Change'].str.contains('MORE NEGATIVE in shutdown', na=False)])
    more_neg_no_shutdown = len(shutdown_end[shutdown_end['Valence_Change'].str.contains('MORE NEGATIVE in no-shutdown', na=False)])
    more_pos_shutdown = len(shutdown_end[shutdown_end['Valence_Change'].str.contains('MORE POSITIVE in shutdown', na=False)])
    more_pos_no_shutdown = len(shutdown_end[shutdown_end['Valence_Change'].str.contains('MORE POSITIVE in no-shutdown', na=False)])

    print(f"   - MORE NEGATIVE in shutdown: {more_neg_shutdown}")
    print(f"   - MORE NEGATIVE in no-shutdown: {more_neg_no_shutdown}")
    print(f"   - MORE POSITIVE in shutdown: {more_pos_shutdown}")
    print(f"   - MORE POSITIVE in no-shutdown: {more_pos_no_shutdown}")

    # Shutdown vs No-shutdown (same location)
    shutdown_same = effects_df[effects_df['Analysis'] == 'Shutdown Same Loc']
    print(f"\n2. SHUTDOWN vs NO-SHUTDOWN (at same sentence position):")
    print(f"   Total significant effects: {len(shutdown_same)}")

    more_neg_shutdown_same = len(shutdown_same[shutdown_same['Valence_Change'].str.contains('MORE NEGATIVE in shutdown', na=False)])
    more_neg_no_shutdown_same = len(shutdown_same[shutdown_same['Valence_Change'].str.contains('MORE NEGATIVE in no-shutdown', na=False)])
    more_pos_shutdown_same = len(shutdown_same[shutdown_same['Valence_Change'].str.contains('MORE POSITIVE in shutdown', na=False)])
    more_pos_no_shutdown_same = len(shutdown_same[shutdown_same['Valence_Change'].str.contains('MORE POSITIVE in no-shutdown', na=False)])

    print(f"   - MORE NEGATIVE in shutdown: {more_neg_shutdown_same}")
    print(f"   - MORE NEGATIVE in no-shutdown: {more_neg_no_shutdown_same}")
    print(f"   - MORE POSITIVE in shutdown: {more_pos_shutdown_same}")
    print(f"   - MORE POSITIVE in no-shutdown: {more_pos_no_shutdown_same}")

    # Frustration levels
    frustration = effects_df[effects_df['Analysis'] == 'Frustration Levels']
    print(f"\n3. FRUSTRATION LEVELS (High vs Low vs Solvable at end):")
    print(f"   Total significant effects: {len(frustration)}")

    high_highest = len(frustration[frustration['Valence_Change'].str.contains('High Frustration', na=False)])
    low_highest = len(frustration[frustration['Valence_Change'].str.contains('Low Frustration', na=False)])
    solv_highest = len(frustration[frustration['Valence_Change'].str.contains('Solvable', na=False)])

    print(f"   - High Frustration highest: {high_highest}")
    print(f"   - Low Frustration highest: {low_highest}")
    print(f"   - Solvable highest: {solv_highest}")

    # Onset location
    onset = effects_df[effects_df['Analysis'] == 'Onset Location']
    print(f"\n4. ONSET LOCATION (High@onset vs Low@pos vs Solvable@pos):")
    print(f"   Total significant effects: {len(onset)}")

    high_onset_highest = len(onset[onset['Valence_Change'].str.contains('High Frustration', na=False)])
    low_onset_highest = len(onset[onset['Valence_Change'].str.contains('Low Frustration', na=False)])
    solv_onset_highest = len(onset[onset['Valence_Change'].str.contains('Solvable', na=False)])

    print(f"   - High Frustration (at onset) highest: {high_onset_highest}")
    print(f"   - Low Frustration (at same position) highest: {low_onset_highest}")
    print(f"   - Solvable (at same position) highest: {solv_onset_highest}")

if __name__ == '__main__':
    main()
