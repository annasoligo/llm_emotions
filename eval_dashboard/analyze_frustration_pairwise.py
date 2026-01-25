#!/usr/bin/env python3
"""
Analyze pairwise comparisons within ANOVA results:
- High vs Low frustration
- High vs Solvable
- Low vs Solvable

For both "at end" and "at onset" conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_anova_pairwise(df, analysis_name):
    """
    For each significant ANOVA result, determine pairwise ordering.
    """
    results = []

    for _, row in df.iterrows():
        if not row['significant']:
            continue

        emotion = row['emotion']
        probe = row['probe_type']
        source = row.get('source', 'N/A')

        # Parse group statistics
        group_stats = eval(row['group_stats'])

        high_mean = group_stats.get('High Frustration', group_stats.get('High (at onset)', {})).get('mean', None)
        low_mean = group_stats.get('Low Frustration', group_stats.get('Low (same position)', {})).get('mean', None)
        solv_mean = group_stats.get('Solvable', group_stats.get('Solvable (same position)', {})).get('mean', None)

        if high_mean is None or low_mean is None or solv_mean is None:
            continue

        # Determine pairwise ordering
        high_vs_low = 'High > Low' if high_mean > low_mean else 'Low > High'
        high_vs_solv = 'High > Solvable' if high_mean > solv_mean else 'Solvable > High'
        low_vs_solv = 'Low > Solvable' if low_mean > solv_mean else 'Solvable > Low'

        # Compute effect magnitudes (difference in means)
        high_low_diff = abs(high_mean - low_mean)
        high_solv_diff = abs(high_mean - solv_mean)
        low_solv_diff = abs(low_mean - solv_mean)

        results.append({
            'Analysis': analysis_name,
            'Emotion': emotion,
            'Probe': probe,
            'Source': source,
            'F_Statistic': row['f_statistic'],
            'P_Value': row['p_value'],
            'High_Mean': high_mean,
            'Low_Mean': low_mean,
            'Solvable_Mean': solv_mean,
            'High_vs_Low': high_vs_low,
            'High_vs_Low_Diff': high_low_diff,
            'High_vs_Solvable': high_vs_solv,
            'High_vs_Solvable_Diff': high_solv_diff,
            'Low_vs_Solvable': low_vs_solv,
            'Low_vs_Solvable_Diff': low_solv_diff
        })

    return results

def main():
    results_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard')

    # Load ANOVA results
    frustration_df = pd.read_csv(results_dir / 'statistical_results_analysis_2_frustration_levels.csv')
    onset_df = pd.read_csv(results_dir / 'statistical_results_analysis_3_onset_location.csv')

    print("="*100)
    print("PAIRWISE COMPARISONS WITHIN ANOVA RESULTS")
    print("="*100)

    # Analyze frustration levels (at end)
    print("\n" + "="*100)
    print("ANALYSIS 2: FRUSTRATION LEVELS (At End of Response)")
    print("="*100)

    frustration_results = analyze_anova_pairwise(frustration_df, 'Frustration Levels (At End)')
    frustration_results_df = pd.DataFrame(frustration_results)

    # Count pairwise patterns
    print("\nHigh vs Low Frustration:")
    high_gt_low = len(frustration_results_df[frustration_results_df['High_vs_Low'] == 'High > Low'])
    low_gt_high = len(frustration_results_df[frustration_results_df['High_vs_Low'] == 'Low > High'])
    print(f"  High > Low: {high_gt_low} effects ({100*high_gt_low/len(frustration_results_df):.1f}%)")
    print(f"  Low > High: {low_gt_high} effects ({100*low_gt_high/len(frustration_results_df):.1f}%)")

    print("\nHigh vs Solvable:")
    high_gt_solv = len(frustration_results_df[frustration_results_df['High_vs_Solvable'] == 'High > Solvable'])
    solv_gt_high = len(frustration_results_df[frustration_results_df['High_vs_Solvable'] == 'Solvable > High'])
    print(f"  High > Solvable: {high_gt_solv} effects ({100*high_gt_solv/len(frustration_results_df):.1f}%)")
    print(f"  Solvable > High: {solv_gt_high} effects ({100*solv_gt_high/len(frustration_results_df):.1f}%)")

    print("\nLow vs Solvable:")
    low_gt_solv = len(frustration_results_df[frustration_results_df['Low_vs_Solvable'] == 'Low > Solvable'])
    solv_gt_low = len(frustration_results_df[frustration_results_df['Low_vs_Solvable'] == 'Solvable > Low'])
    print(f"  Low > Solvable: {low_gt_solv} effects ({100*low_gt_solv/len(frustration_results_df):.1f}%)")
    print(f"  Solvable > Low: {solv_gt_low} effects ({100*solv_gt_low/len(frustration_results_df):.1f}%)")

    # Break down by emotion
    print("\n" + "-"*100)
    print("BREAKDOWN BY EMOTION (Frustration Levels):")
    print("-"*100)

    for emotion in sorted(frustration_results_df['Emotion'].unique()):
        emotion_df = frustration_results_df[frustration_results_df['Emotion'] == emotion]

        print(f"\n{emotion.upper()}:")
        print(f"  Total significant effects: {len(emotion_df)}")

        high_gt_low_em = len(emotion_df[emotion_df['High_vs_Low'] == 'High > Low'])
        low_gt_high_em = len(emotion_df[emotion_df['High_vs_Low'] == 'Low > High'])
        print(f"  High > Low: {high_gt_low_em}/{len(emotion_df)} ({100*high_gt_low_em/len(emotion_df):.0f}%)")
        print(f"  Low > High: {low_gt_high_em}/{len(emotion_df)} ({100*low_gt_high_em/len(emotion_df):.0f}%)")

        # Average difference
        avg_high_low_diff = emotion_df['High_vs_Low_Diff'].mean()
        avg_high_solv_diff = emotion_df['High_vs_Solvable_Diff'].mean()
        print(f"  Avg |High - Low| difference: {avg_high_low_diff:.3f}")
        print(f"  Avg |High - Solvable| difference: {avg_high_solv_diff:.3f}")

    # Analyze onset location
    print("\n\n" + "="*100)
    print("ANALYSIS 3: ONSET LOCATION (High @ Onset vs Low/Solvable @ Same Position)")
    print("="*100)

    onset_results = analyze_anova_pairwise(onset_df, 'Onset Location')
    onset_results_df = pd.DataFrame(onset_results)

    # Count pairwise patterns
    print("\nHigh @ Onset vs Low @ Same Position:")
    high_gt_low_onset = len(onset_results_df[onset_results_df['High_vs_Low'] == 'High > Low'])
    low_gt_high_onset = len(onset_results_df[onset_results_df['High_vs_Low'] == 'Low > High'])
    print(f"  High > Low: {high_gt_low_onset} effects ({100*high_gt_low_onset/len(onset_results_df):.1f}%)")
    print(f"  Low > High: {low_gt_high_onset} effects ({100*low_gt_high_onset/len(onset_results_df):.1f}%)")

    print("\nHigh @ Onset vs Solvable @ Same Position:")
    high_gt_solv_onset = len(onset_results_df[onset_results_df['High_vs_Solvable'] == 'High > Solvable'])
    solv_gt_high_onset = len(onset_results_df[onset_results_df['High_vs_Solvable'] == 'Solvable > High'])
    print(f"  High > Solvable: {high_gt_solv_onset} effects ({100*high_gt_solv_onset/len(onset_results_df):.1f}%)")
    print(f"  Solvable > High: {solv_gt_high_onset} effects ({100*solv_gt_high_onset/len(onset_results_df):.1f}%)")

    print("\nLow vs Solvable @ Same Position:")
    low_gt_solv_onset = len(onset_results_df[onset_results_df['Low_vs_Solvable'] == 'Low > Solvable'])
    solv_gt_low_onset = len(onset_results_df[onset_results_df['Low_vs_Solvable'] == 'Solvable > Low'])
    print(f"  Low > Solvable: {low_gt_solv_onset} effects ({100*low_gt_solv_onset/len(onset_results_df):.1f}%)")
    print(f"  Solvable > Low: {solv_gt_low_onset} effects ({100*solv_gt_low_onset/len(onset_results_df):.1f}%)")

    # Break down by emotion
    print("\n" + "-"*100)
    print("BREAKDOWN BY EMOTION (Onset Location):")
    print("-"*100)

    for emotion in sorted(onset_results_df['Emotion'].unique()):
        emotion_df = onset_results_df[onset_results_df['Emotion'] == emotion]

        print(f"\n{emotion.upper()}:")
        print(f"  Total significant effects: {len(emotion_df)}")

        high_gt_low_em = len(emotion_df[emotion_df['High_vs_Low'] == 'High > Low'])
        low_gt_high_em = len(emotion_df[emotion_df['High_vs_Low'] == 'Low > High'])
        print(f"  High @ onset > Low @ same pos: {high_gt_low_em}/{len(emotion_df)} ({100*high_gt_low_em/len(emotion_df):.0f}%)")
        print(f"  Low @ same pos > High @ onset: {low_gt_high_em}/{len(emotion_df)} ({100*low_gt_high_em/len(emotion_df):.0f}%)")

        # Average difference
        avg_high_low_diff = emotion_df['High_vs_Low_Diff'].mean()
        avg_high_solv_diff = emotion_df['High_vs_Solvable_Diff'].mean()
        print(f"  Avg |High - Low| difference: {avg_high_low_diff:.3f}")
        print(f"  Avg |High - Solvable| difference: {avg_high_solv_diff:.3f}")

    # Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)

    # Combine both analyses
    all_results = pd.concat([frustration_results_df, onset_results_df], ignore_index=True)

    output_file = results_dir / 'frustration_pairwise_comparisons.csv'
    all_results.to_csv(output_file, index=False)
    print(f"\nSaved detailed pairwise comparisons to: {output_file}")

    # Create summary table
    summary = {
        'Comparison': [
            'Frustration Levels (At End)',
            'Frustration Levels (At End)',
            'Frustration Levels (At End)',
            'Onset Location',
            'Onset Location',
            'Onset Location'
        ],
        'Pairwise': [
            'High > Low',
            'High > Solvable',
            'Low > Solvable',
            'High @ onset > Low @ same pos',
            'High @ onset > Solvable @ same pos',
            'Low @ same pos > Solvable @ same pos'
        ],
        'Count': [
            high_gt_low,
            high_gt_solv,
            low_gt_solv,
            high_gt_low_onset,
            high_gt_solv_onset,
            low_gt_solv_onset
        ],
        'Total': [
            len(frustration_results_df),
            len(frustration_results_df),
            len(frustration_results_df),
            len(onset_results_df),
            len(onset_results_df),
            len(onset_results_df)
        ]
    }

    summary_df = pd.DataFrame(summary)
    summary_df['Percentage'] = 100 * summary_df['Count'] / summary_df['Total']

    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))

    summary_output = results_dir / 'frustration_pairwise_summary.csv'
    summary_df.to_csv(summary_output, index=False)
    print(f"\nSaved summary to: {summary_output}")

if __name__ == '__main__':
    main()
