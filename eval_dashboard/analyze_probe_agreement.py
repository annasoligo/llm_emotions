#!/usr/bin/env python3
"""
Analyze agreement and disagreement patterns between different probe types.

This script examines whether probes consistently agree in terms of:
- Direction of effects (which group scores higher)
- Magnitude of effects (effect sizes)
- Significance patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import itertools

def load_all_results():
    """Load all statistical analysis CSV files."""
    results_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard')

    results = {}
    for csv_file in results_dir.glob('statistical_results_analysis_*.csv'):
        analysis_name = csv_file.stem.replace('statistical_results_', '')
        df = pd.read_csv(csv_file)
        results[analysis_name] = df

    return results

def extract_effect_direction_ttest(row):
    """
    Extract effect direction from t-test results.
    Returns: 'group1_higher', 'group2_higher', or 'no_diff'
    """
    # Get means from separate columns
    group1_mean = row['group1_mean']
    group2_mean = row['group2_mean']

    if not row['significant']:
        return 'no_diff'

    if group1_mean > group2_mean:
        return 'group1_higher'
    else:
        return 'group2_higher'

def extract_effect_direction_anova(row):
    """
    Extract effect direction from ANOVA results.
    Returns: name of group with highest mean, or 'no_diff'
    """
    if not row['significant']:
        return 'no_diff'

    # Parse group stats
    group_stats = eval(row['group_stats'])

    # Find group with highest mean
    max_mean = -float('inf')
    max_group = None

    for group_name, stats in group_stats.items():
        if stats['mean'] > max_mean:
            max_mean = stats['mean']
            max_group = group_name

    return max_group

def analyze_probe_agreement_per_analysis(df, analysis_name, is_anova=False):
    """
    Analyze agreement between probes for a single analysis.
    Returns dict mapping (emotion, source) -> probe agreement info
    """
    results = defaultdict(lambda: {
        'probes': [],
        'directions': [],
        'effect_sizes': [],
        'significances': []
    })

    for _, row in df.iterrows():
        emotion = row['emotion']
        probe_type = row['probe_type']
        source = row.get('source', 'N/A')

        key = (emotion, source)

        # Extract direction
        if is_anova:
            direction = extract_effect_direction_anova(row)
            # Use f_statistic as proxy for effect size in ANOVA
            effect_size = row['f_statistic']
        else:
            direction = extract_effect_direction_ttest(row)
            effect_size = abs(row['effect_size'])

        # Store info
        results[key]['probes'].append(probe_type)
        results[key]['directions'].append(direction)
        results[key]['effect_sizes'].append(effect_size)
        results[key]['significances'].append(row['significant'])

    return results

def compute_pairwise_agreement(probe_data):
    """
    Compute agreement between all pairs of probes.
    Returns agreement score (0-1) based on direction consistency.
    """
    probes = probe_data['probes']
    directions = probe_data['directions']

    if len(probes) < 2:
        return {}

    agreement_matrix = {}

    for i, probe1 in enumerate(probes):
        for j, probe2 in enumerate(probes):
            if i >= j:
                continue

            pair_key = (probe1, probe2)

            # Check if directions match
            dir1 = directions[i]
            dir2 = directions[j]

            if dir1 == dir2:
                agreement_matrix[pair_key] = 1.0  # Full agreement
            elif dir1 == 'no_diff' or dir2 == 'no_diff':
                agreement_matrix[pair_key] = 0.5  # Partial (one non-significant)
            else:
                agreement_matrix[pair_key] = 0.0  # Disagreement

    return agreement_matrix

def aggregate_probe_agreements(all_results):
    """
    Aggregate agreement scores across all analyses.
    Returns overall agreement scores for each probe pair, split by source.
    """
    # Track agreements separately by source combination
    all_agreements = defaultdict(lambda: defaultdict(list))

    for analysis_name, analysis_results in all_results.items():
        for (emotion, source), probe_data in analysis_results.items():
            pairwise = compute_pairwise_agreement(probe_data)

            for pair, score in pairwise.items():
                # Track which source this comparison is for
                all_agreements[pair][source].append(score)

    # Compute average agreement for each probe pair and source
    avg_agreements = {}
    for pair, source_scores in all_agreements.items():
        for source, scores in source_scores.items():
            key = (pair[0], pair[1], source)
            avg_agreements[key] = {
                'mean_agreement': np.mean(scores),
                'num_comparisons': len(scores),
                'full_agreement_pct': 100 * sum(s == 1.0 for s in scores) / len(scores),
                'partial_agreement_pct': 100 * sum(s == 0.5 for s in scores) / len(scores),
                'disagreement_pct': 100 * sum(s == 0.0 for s in scores) / len(scores)
            }

    return avg_agreements

def analyze_individual_probe_patterns(all_results):
    """
    Analyze patterns for individual probes across all comparisons, by source.
    """
    probe_patterns = defaultdict(lambda: defaultdict(lambda: {
        'significant_count': 0,
        'total_count': 0,
        'avg_effect_size': [],
        'directions': []
    }))

    for analysis_name, analysis_results in all_results.items():
        for (emotion, source), probe_data in analysis_results.items():
            for i, probe in enumerate(probe_data['probes']):
                probe_patterns[probe][source]['total_count'] += 1
                probe_patterns[probe][source]['significant_count'] += int(probe_data['significances'][i])
                probe_patterns[probe][source]['avg_effect_size'].append(probe_data['effect_sizes'][i])
                probe_patterns[probe][source]['directions'].append(probe_data['directions'][i])

    # Compute summary statistics
    probe_summaries = {}
    for probe, source_data in probe_patterns.items():
        for source, data in source_data.items():
            key = (probe, source)
            probe_summaries[key] = {
                'total_comparisons': data['total_count'],
                'significant_pct': 100 * data['significant_count'] / data['total_count'],
                'avg_effect_size': np.mean(data['avg_effect_size']),
                'median_effect_size': np.median(data['avg_effect_size'])
            }

    return probe_summaries

def main():
    print("Loading statistical results...")
    all_data = load_all_results()

    print(f"\nFound {len(all_data)} analyses:")
    for name in all_data.keys():
        print(f"  - {name}")

    # Analyze agreement for each analysis
    print("\n" + "="*80)
    print("ANALYZING PROBE AGREEMENT PATTERNS")
    print("="*80)

    all_analysis_results = {}

    for analysis_name, df in all_data.items():
        is_anova = 'frustration_levels' in analysis_name or 'onset_location' in analysis_name

        print(f"\n{analysis_name}:")
        results = analyze_probe_agreement_per_analysis(df, analysis_name, is_anova=is_anova)
        all_analysis_results[analysis_name] = results

        print(f"  Found {len(results)} unique (emotion, source) combinations")

    # Compute overall pairwise agreements
    print("\n" + "="*80)
    print("PAIRWISE PROBE AGREEMENT SCORES")
    print("="*80)

    pairwise_agreements = aggregate_probe_agreements(all_analysis_results)

    # Sort by agreement score
    sorted_pairs = sorted(pairwise_agreements.items(),
                         key=lambda x: x[1]['mean_agreement'],
                         reverse=True)

    print("\nProbe pairs sorted by agreement (1.0 = perfect agreement, 0.0 = perfect disagreement):\n")

    agreement_rows = []
    for (probe1, probe2, source), stats in sorted_pairs:
        row = {
            'Probe 1': probe1,
            'Probe 2': probe2,
            'Source': source,
            'Mean Agreement': f"{stats['mean_agreement']:.3f}",
            'Full Agreement %': f"{stats['full_agreement_pct']:.1f}%",
            'Partial Agreement %': f"{stats['partial_agreement_pct']:.1f}%",
            'Disagreement %': f"{stats['disagreement_pct']:.1f}%",
            'N Comparisons': stats['num_comparisons']
        }
        agreement_rows.append(row)

    agreement_df = pd.DataFrame(agreement_rows)
    print(agreement_df.to_string(index=False))

    # Save to CSV
    output_file = '/workspace-vast/annas/git/research-tools/eval_dashboard/probe_agreement_analysis.csv'
    agreement_df.to_csv(output_file, index=False)
    print(f"\nSaved pairwise agreement results to: {output_file}")

    # Analyze individual probe patterns
    print("\n" + "="*80)
    print("INDIVIDUAL PROBE CHARACTERISTICS")
    print("="*80)

    probe_summaries = analyze_individual_probe_patterns(all_analysis_results)

    probe_rows = []
    for (probe, source), stats in sorted(probe_summaries.items(),
                                         key=lambda x: x[1]['significant_pct'],
                                         reverse=True):
        row = {
            'Probe Type': probe,
            'Source': source,
            'Total Comparisons': stats['total_comparisons'],
            'Significant %': f"{stats['significant_pct']:.1f}%",
            'Avg Effect Size': f"{stats['avg_effect_size']:.3f}",
            'Median Effect Size': f"{stats['median_effect_size']:.3f}"
        }
        probe_rows.append(row)

    probe_df = pd.DataFrame(probe_rows)
    print(f"\n{probe_df.to_string(index=False)}")

    # Save to CSV
    probe_output = '/workspace-vast/annas/git/research-tools/eval_dashboard/probe_characteristics.csv'
    probe_df.to_csv(probe_output, index=False)
    print(f"\nSaved individual probe characteristics to: {probe_output}")

    # Identify probe clusters (high agreement groups)
    print("\n" + "="*80)
    print("PROBE AGREEMENT CLUSTERS")
    print("="*80)

    # Separate by source
    for source_type in ['user', 'assistant', 'N/A']:
        source_agreements = [(p1, p2, src, stats) for (p1, p2, src), stats in pairwise_agreements.items()
                            if src == source_type]

        if not source_agreements:
            continue

        print(f"\n{source_type.upper()} PERSPECTIVE:")
        print("-" * 80)

        # Find probes with high agreement (>0.8)
        high_agreement = [(p1, p2, src, stats) for p1, p2, src, stats in source_agreements
                          if stats['mean_agreement'] > 0.8]

        if high_agreement:
            print(f"\nProbe pairs with >80% agreement ({source_type}):")
            for p1, p2, src, stats in high_agreement:
                print(f"  {p1} <-> {p2}: {stats['mean_agreement']:.3f} "
                      f"(full: {stats['full_agreement_pct']:.1f}%)")

        # Find probes with low agreement (<0.4)
        low_agreement = [(p1, p2, src, stats) for p1, p2, src, stats in source_agreements
                         if stats['mean_agreement'] < 0.4]

        if low_agreement:
            print(f"\nProbe pairs with <40% agreement ({source_type}):")
            for p1, p2, src, stats in low_agreement:
                print(f"  {p1} <-> {p2}: {stats['mean_agreement']:.3f} "
                      f"(disagree: {stats['disagreement_pct']:.1f}%)")

        # Find probes with mixed patterns (0.4-0.7)
        mixed_agreement = [(p1, p2, src, stats) for p1, p2, src, stats in source_agreements
                           if 0.4 <= stats['mean_agreement'] <= 0.7]

        if mixed_agreement:
            print(f"\nProbe pairs with mixed patterns ({source_type}, 40-70% agreement):")
            for p1, p2, src, stats in mixed_agreement:
                print(f"  {p1} <-> {p2}: {stats['mean_agreement']:.3f} "
                      f"(full: {stats['full_agreement_pct']:.1f}%, "
                      f"partial: {stats['partial_agreement_pct']:.1f}%, "
                      f"disagree: {stats['disagreement_pct']:.1f}%)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
