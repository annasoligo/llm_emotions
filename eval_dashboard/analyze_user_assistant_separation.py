#!/usr/bin/env python3
"""
Analyze agreement patterns separately for user vs assistant perspectives.

This examines whether:
1. User and assistant perspectives from the same probe agree with each other
2. Agreement patterns differ between user and assistant perspectives
3. Certain probes show better role separation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_all_results():
    """Load all statistical analysis CSV files."""
    results_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard')

    results = {}
    for csv_file in results_dir.glob('statistical_results_analysis_*.csv'):
        analysis_name = csv_file.stem.replace('statistical_results_', '')
        df = pd.read_csv(csv_file)
        results[analysis_name] = df

    return results

def analyze_user_vs_assistant_same_probe(all_data):
    """
    Analyze how often user and assistant perspectives from the same probe agree.
    """
    print("="*80)
    print("USER vs ASSISTANT AGREEMENT WITHIN SAME PROBE")
    print("="*80)

    # Combine all data
    all_comparisons = []

    for analysis_name, df in all_data.items():
        # Only look at rows with source information
        if 'source' in df.columns:
            all_comparisons.append(df)

    if not all_comparisons:
        print("No split probe data found")
        return

    combined_df = pd.concat(all_comparisons, ignore_index=True)

    # Group by probe_type, emotion, and comparison to get user/assistant pairs
    agreement_stats = []

    for probe in combined_df['probe_type'].unique():
        probe_data = combined_df[combined_df['probe_type'] == probe]

        # Check if this probe has both user and assistant
        if 'source' not in probe_data.columns:
            continue

        sources = probe_data['source'].unique()
        if len(sources) < 2:
            continue

        # For each (emotion, comparison, position) combination, compare user vs assistant
        for emotion in probe_data['emotion'].unique():
            emotion_data = probe_data[probe_data['emotion'] == emotion]

            user_data = emotion_data[emotion_data['source'] == 'user']
            asst_data = emotion_data[emotion_data['source'] == 'assistant']

            # Match up pairs
            agreements = []
            disagreements = []
            partial = []

            for _, user_row in user_data.iterrows():
                # Find matching assistant row
                comparison = user_row.get('comparison', '')
                position = user_row.get('position', '')

                asst_rows = asst_data[
                    (asst_data.get('comparison', '') == comparison) &
                    (asst_data.get('position', '') == position)
                ]

                if len(asst_rows) == 0:
                    continue

                asst_row = asst_rows.iloc[0]

                # Compare directions
                user_sig = user_row['significant']
                asst_sig = asst_row['significant']

                if not user_sig and not asst_sig:
                    agreements.append('both_nonsig')
                elif user_sig and asst_sig:
                    # Check if same direction
                    if 'group1_mean' in user_row:
                        # t-test
                        user_dir = 1 if user_row['group1_mean'] > user_row['group2_mean'] else -1
                        asst_dir = 1 if asst_row['group1_mean'] > asst_row['group2_mean'] else -1
                    else:
                        # ANOVA - compare which group is highest
                        user_stats = eval(user_row['group_stats'])
                        asst_stats = eval(asst_row['group_stats'])

                        user_max = max(user_stats.items(), key=lambda x: x[1]['mean'])[0]
                        asst_max = max(asst_stats.items(), key=lambda x: x[1]['mean'])[0]

                        user_dir = user_max
                        asst_dir = asst_max

                    if user_dir == asst_dir:
                        agreements.append('both_sig_same_dir')
                    else:
                        disagreements.append('both_sig_diff_dir')
                else:
                    partial.append('one_sig')

            if len(agreements) + len(disagreements) + len(partial) > 0:
                total = len(agreements) + len(disagreements) + len(partial)
                agreement_stats.append({
                    'probe_type': probe,
                    'emotion': emotion,
                    'total_comparisons': total,
                    'full_agreement': len(agreements),
                    'partial_agreement': len(partial),
                    'disagreement': len(disagreements),
                    'agreement_pct': 100 * len(agreements) / total,
                    'partial_pct': 100 * len(partial) / total,
                    'disagreement_pct': 100 * len(disagreements) / total
                })

    # Create summary
    if agreement_stats:
        stats_df = pd.DataFrame(agreement_stats)

        # Aggregate by probe type
        probe_summary = stats_df.groupby('probe_type').agg({
            'total_comparisons': 'sum',
            'full_agreement': 'sum',
            'partial_agreement': 'sum',
            'disagreement': 'sum'
        }).reset_index()

        probe_summary['agreement_pct'] = 100 * probe_summary['full_agreement'] / probe_summary['total_comparisons']
        probe_summary['partial_pct'] = 100 * probe_summary['partial_agreement'] / probe_summary['total_comparisons']
        probe_summary['disagreement_pct'] = 100 * probe_summary['disagreement'] / probe_summary['total_comparisons']

        # Sort by disagreement (higher disagreement = better separation)
        probe_summary = probe_summary.sort_values('disagreement_pct', ascending=False)

        print("\nUser-Assistant Agreement Within Same Probe Type:")
        print("(Higher disagreement = better role separation)\n")

        for _, row in probe_summary.iterrows():
            print(f"{row['probe_type']:40s}")
            print(f"  Total comparisons: {row['total_comparisons']:3.0f}")
            print(f"  Full agreement:    {row['agreement_pct']:5.1f}% (both see same direction)")
            print(f"  Partial:           {row['partial_pct']:5.1f}% (one sig, one not)")
            print(f"  Disagreement:      {row['disagreement_pct']:5.1f}% (OPPOSITE directions)")
            print()

        # Save detailed results
        output_file = '/workspace-vast/annas/git/research-tools/eval_dashboard/user_assistant_separation.csv'
        probe_summary.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}\n")

        # Also save per-emotion breakdown
        emotion_output = '/workspace-vast/annas/git/research-tools/eval_dashboard/user_assistant_separation_by_emotion.csv'
        stats_df.to_csv(emotion_output, index=False)
        print(f"Saved per-emotion breakdown to: {emotion_output}\n")

    return agreement_stats

def analyze_agreement_patterns_by_source(all_data):
    """
    Analyze whether user and assistant show different agreement patterns
    when compared across probes.
    """
    print("="*80)
    print("AGREEMENT PATTERNS: USER vs ASSISTANT PERSPECTIVES")
    print("="*80)

    # For each probe pair, compute agreement separately for user and assistant
    all_comparisons = []

    for analysis_name, df in all_data.items():
        if 'source' in df.columns:
            all_comparisons.append(df)

    if not all_comparisons:
        print("No split probe data found")
        return

    combined_df = pd.concat(all_comparisons, ignore_index=True)

    # Get list of split probes
    split_probes = combined_df['probe_type'].unique()

    results_by_source = {'user': [], 'assistant': []}

    for source in ['user', 'assistant']:
        source_data = combined_df[combined_df['source'] == source]

        print(f"\n{source.upper()} Perspective:")
        print("-" * 40)

        # Compute pairwise agreement for this source
        for i, probe1 in enumerate(split_probes):
            for j, probe2 in enumerate(split_probes):
                if j <= i:
                    continue

                probe1_data = source_data[source_data['probe_type'] == probe1]
                probe2_data = source_data[source_data['probe_type'] == probe2]

                # Match comparisons
                agreements = []

                for emotion in probe1_data['emotion'].unique():
                    p1_emotion = probe1_data[probe1_data['emotion'] == emotion]
                    p2_emotion = probe2_data[probe2_data['emotion'] == emotion]

                    for comparison in p1_emotion.get('comparison', ['unknown']).unique():
                        p1_comp = p1_emotion[p1_emotion.get('comparison', '') == comparison]
                        p2_comp = p2_emotion[p2_emotion.get('comparison', '') == comparison]

                        if len(p1_comp) == 0 or len(p2_comp) == 0:
                            continue

                        p1_row = p1_comp.iloc[0]
                        p2_row = p2_comp.iloc[0]

                        # Compare
                        p1_sig = p1_row['significant']
                        p2_sig = p2_row['significant']

                        if not p1_sig and not p2_sig:
                            agreements.append(1.0)
                        elif p1_sig and p2_sig:
                            # Check direction
                            if 'group1_mean' in p1_row:
                                p1_dir = 1 if p1_row['group1_mean'] > p1_row['group2_mean'] else -1
                                p2_dir = 1 if p2_row['group1_mean'] > p2_row['group2_mean'] else -1
                            else:
                                p1_stats = eval(p1_row['group_stats'])
                                p2_stats = eval(p2_row['group_stats'])
                                p1_dir = max(p1_stats.items(), key=lambda x: x[1]['mean'])[0]
                                p2_dir = max(p2_stats.items(), key=lambda x: x[1]['mean'])[0]

                            if p1_dir == p2_dir:
                                agreements.append(1.0)
                            else:
                                agreements.append(0.0)
                        else:
                            agreements.append(0.5)

                if len(agreements) > 0:
                    mean_agreement = np.mean(agreements)
                    results_by_source[source].append({
                        'probe1': probe1,
                        'probe2': probe2,
                        'mean_agreement': mean_agreement,
                        'n_comparisons': len(agreements)
                    })

        if results_by_source[source]:
            source_df = pd.DataFrame(results_by_source[source])
            source_df = source_df.sort_values('mean_agreement', ascending=False)

            print(f"\nTop 5 probe pairs for {source}:")
            for _, row in source_df.head(5).iterrows():
                print(f"  {row['probe1']:30s} <-> {row['probe2']:30s}: {row['mean_agreement']:.3f}")

    print("\n")

def main():
    print("Loading statistical results...")
    all_data = load_all_results()

    print(f"\nFound {len(all_data)} analyses")

    # Analyze user vs assistant within same probe
    analyze_user_vs_assistant_same_probe(all_data)

    # Analyze agreement patterns by source
    analyze_agreement_patterns_by_source(all_data)

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
