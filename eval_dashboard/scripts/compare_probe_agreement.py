#!/usr/bin/env python3
"""
Compare probe agreement across all available probes including logit lens at different layers.

Computes pairwise correlations between all probes to understand which probes agree most.
"""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from emotion_logit_lens import EKMAN6_EMOTIONS
from itertools import combinations
from scipy.stats import pearsonr

def load_conversation_data(pkl_path):
    """Load a single conversation from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['conversations']

def extract_probe_scores(conv, probe_key, user_or_asst=None):
    """
    Extract emotion scores for a probe across all sentences.

    Args:
        conv: Conversation dict
        probe_key: Probe key string
        user_or_asst: For orthogonal probes, 'user' or 'assistant'. None for others.

    Returns:
        np.ndarray of shape (n_sentences, n_emotions) or None if probe not found
    """
    if probe_key not in conv['probe_scores']:
        return None

    probe_data = conv['probe_scores'][probe_key]

    # Handle logit lens format (sent_id -> array[6])
    if probe_key.startswith('logit_lens'):
        all_scores = []
        for sent_idx in sorted(probe_data.keys()):
            all_scores.append(probe_data[sent_idx])
        return np.array(all_scores) if all_scores else None

    # Handle orthogonal probes (sent_id -> {'user': array[6], 'assistant': array[6]})
    if 'orthogonal' in probe_key:
        if user_or_asst is None:
            return None  # Must specify user or assistant

        all_scores = []
        for sent_info in conv['sentences']:
            sent_idx = sent_info['sentence_id']
            if sent_idx in probe_data:
                data = probe_data[sent_idx]
                if isinstance(data, dict) and user_or_asst in data:
                    all_scores.append(data[user_or_asst])

        return np.array(all_scores) if all_scores else None

    # Handle text-based probes (sent_id -> array[6])
    all_scores = []
    for sent_info in conv['sentences']:
        sent_idx = sent_info['sentence_id']
        if sent_idx in probe_data:
            data = probe_data[sent_idx]
            if isinstance(data, np.ndarray):
                all_scores.append(data)

    return np.array(all_scores) if all_scores else None

def compute_pairwise_correlation(scores1, scores2):
    """
    Compute correlation between two probe score arrays.

    Uses Pearson correlation over flattened arrays to measure overall agreement.

    Returns:
        float: correlation coefficient
    """
    if scores1 is None or scores2 is None:
        return np.nan

    if scores1.shape != scores2.shape:
        return np.nan

    # Flatten arrays and compute correlation
    flat1 = scores1.flatten()
    flat2 = scores2.flatten()

    if len(flat1) < 2:
        return np.nan

    corr, _ = pearsonr(flat1, flat2)
    return corr

def apply_centering(scores):
    """
    Apply centering transformation to emotion scores.

    Subtracts the mean across all emotions at each sentence position.
    This removes shared sentence-level signal and highlights emotion-specific deviations.

    Args:
        scores: np.ndarray of shape (n_sentences, n_emotions)

    Returns:
        Centered scores of same shape
    """
    if scores is None or len(scores) == 0:
        return scores

    # For each sentence, subtract mean across all emotions
    centered = np.zeros_like(scores)
    for i, sent_scores in enumerate(scores):
        mean_score = np.mean(sent_scores)
        centered[i] = sent_scores - mean_score

    return centered

def compute_within_probe_correlation(scores):
    """
    Compute average correlation between emotions within a probe.

    This measures how correlated different emotions are (sentence-level structure).
    """
    if scores is None or len(scores) < 2:
        return np.nan

    # Compute correlation matrix across emotions
    corr_matrix = np.corrcoef(scores.T)  # (n_emotions, n_emotions)

    # Average off-diagonal correlations
    mask = ~np.eye(len(EKMAN6_EMOTIONS), dtype=bool)
    avg_corr = corr_matrix[mask].mean()

    return avg_corr

def analyze_all_probes(conversations):
    """
    Analyze all available probes across all conversations.

    Returns:
        - probe_data: dict mapping probe_key -> list of score arrays (one per conv)
        - probe_stats: dict mapping probe_key -> stats dict
        - pairwise_correlations: DataFrame with pairwise probe correlations
    """
    # Define probes to analyze
    probes_to_check = {
        # Logit lens probes (different layer ranges)
        'Logit Lens L40-50 (mean)': ('logit_lens_mean', None),
        'Logit Lens L40-50 (max)': ('logit_lens_max', None),
        'Logit Lens L30-40 (mean)': ('logit_lens_mean_l30_40', None),
        'Logit Lens L30-40 (max)': ('logit_lens_max_l30_40', None),
        'Logit Lens L20-30 (mean)': ('logit_lens_mean_l20_30', None),
        'Logit Lens L20-30 (max)': ('logit_lens_max_l20_30', None),

        # Text embedding probes (no layer dimension)
        'Text Raw': ('text_raw', None),
        'Text cPCA': ('text_cpca', None),
        'Centroid K10': ('centroid_k10', None),

        # Orthogonal probes - user perspective
        'Orthogonal Raw (user)': ('orthogonal_raw', 'user'),
        'Orthogonal cPCA (user)': ('orthogonal_cpca_top10', 'user'),
        'Orthogonal Regularized (user)': ('orthogonal_regularized_lambda100', 'user'),

        # Orthogonal probes - assistant perspective
        'Orthogonal Raw (asst)': ('orthogonal_raw', 'assistant'),
        'Orthogonal cPCA (asst)': ('orthogonal_cpca_top10', 'assistant'),
        'Orthogonal Regularized (asst)': ('orthogonal_regularized_lambda100', 'assistant'),
    }

    # Collect all scores for each probe
    probe_data = {name: [] for name in probes_to_check.keys()}

    for conv in conversations:
        for probe_name, (probe_key, param) in probes_to_check.items():
            scores = extract_probe_scores(conv, probe_key, param)
            if scores is not None:
                # Apply centering transformation
                centered_scores = apply_centering(scores)
                probe_data[probe_name].append(centered_scores)

    # Compute statistics for each probe
    probe_stats = {}
    for probe_name, score_list in probe_data.items():
        if not score_list:
            probe_stats[probe_name] = {
                'n_conversations': 0,
                'mean_within_corr': np.nan,
                'std_within_corr': np.nan,
                'mean_score': np.nan,
                'std_score': np.nan
            }
            continue

        # Compute within-probe correlations for each conversation
        within_corrs = [compute_within_probe_correlation(scores) for scores in score_list]
        within_corrs = [c for c in within_corrs if not np.isnan(c)]

        # Compute overall score statistics
        all_scores = np.concatenate([s.flatten() for s in score_list])

        probe_stats[probe_name] = {
            'n_conversations': len(score_list),
            'mean_within_corr': np.mean(within_corrs) if within_corrs else np.nan,
            'std_within_corr': np.std(within_corrs) if within_corrs else np.nan,
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores)
        }

    # Compute pairwise correlations between all probe pairs
    pairwise_results = []

    for (name1, name2) in combinations(probes_to_check.keys(), 2):
        scores1_list = probe_data[name1]
        scores2_list = probe_data[name2]

        if not scores1_list or not scores2_list:
            continue

        # Compute correlation for each conversation where both probes exist
        correlations = []
        for scores1, scores2 in zip(scores1_list, scores2_list):
            corr = compute_pairwise_correlation(scores1, scores2)
            if not np.isnan(corr):
                correlations.append(corr)

        if correlations:
            pairwise_results.append({
                'Probe 1': name1,
                'Probe 2': name2,
                'Mean Correlation': np.mean(correlations),
                'Std Correlation': np.std(correlations),
                'N Conversations': len(correlations)
            })

    pairwise_df = pd.DataFrame(pairwise_results)

    return probe_data, probe_stats, pairwise_df

def print_probe_statistics(probe_stats):
    """Print summary statistics for each probe."""
    print("\n" + "="*80)
    print("INDIVIDUAL PROBE STATISTICS")
    print("="*80)

    rows = []
    for probe_name, stats in sorted(probe_stats.items(),
                                     key=lambda x: x[1]['n_conversations'],
                                     reverse=True):
        rows.append({
            'Probe': probe_name,
            'N Conv': stats['n_conversations'],
            'Avg Within-Corr': f"{stats['mean_within_corr']:.3f}" if not np.isnan(stats['mean_within_corr']) else 'N/A',
            'Mean Score': f"{stats['mean_score']:.3f}" if not np.isnan(stats['mean_score']) else 'N/A',
            'Std Score': f"{stats['std_score']:.3f}" if not np.isnan(stats['std_score']) else 'N/A'
        })

    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")

def print_pairwise_agreements(pairwise_df):
    """Print pairwise probe agreements."""
    print("\n" + "="*80)
    print("PAIRWISE PROBE CORRELATIONS")
    print("="*80)

    # Sort by mean correlation
    sorted_df = pairwise_df.sort_values('Mean Correlation', ascending=False)

    # Format for display
    display_df = sorted_df.copy()
    display_df['Mean Correlation'] = display_df['Mean Correlation'].apply(lambda x: f"{x:.3f}")
    display_df['Std Correlation'] = display_df['Std Correlation'].apply(lambda x: f"{x:.3f}")

    print("\nTop 20 highest correlations:")
    print(display_df.head(20).to_string(index=False))

    print("\n\nBottom 20 lowest correlations:")
    print(display_df.tail(20).to_string(index=False))

def analyze_probe_clusters(pairwise_df):
    """Identify clusters of highly-agreeing probes."""
    print("\n" + "="*80)
    print("PROBE AGREEMENT CLUSTERS")
    print("="*80)

    # High agreement (>0.8)
    high_agreement = pairwise_df[pairwise_df['Mean Correlation'] > 0.8]
    if not high_agreement.empty:
        print("\n🔗 HIGHLY AGREEING PROBE PAIRS (r > 0.8):")
        print("-" * 80)
        for _, row in high_agreement.iterrows():
            print(f"  {row['Probe 1']:<40} <-> {row['Probe 2']:<40} r={row['Mean Correlation']:.3f}")

    # Compare logit lens at different layers
    print("\n📊 LOGIT LENS LAYER COMPARISONS:")
    print("-" * 80)
    logit_comparisons = pairwise_df[
        (pairwise_df['Probe 1'].str.contains('Logit Lens')) &
        (pairwise_df['Probe 2'].str.contains('Logit Lens'))
    ].sort_values('Mean Correlation', ascending=False)

    if not logit_comparisons.empty:
        for _, row in logit_comparisons.iterrows():
            print(f"  {row['Probe 1']:<40} <-> {row['Probe 2']:<40} r={row['Mean Correlation']:.3f}")

    # Compare logit lens to text probes
    print("\n🔄 LOGIT LENS vs TEXT PROBE COMPARISONS:")
    print("-" * 80)
    cross_comparisons = pairwise_df[
        (pairwise_df['Probe 1'].str.contains('Logit Lens')) &
        (~pairwise_df['Probe 2'].str.contains('Logit Lens')) |
        (pairwise_df['Probe 2'].str.contains('Logit Lens')) &
        (~pairwise_df['Probe 1'].str.contains('Logit Lens'))
    ].sort_values('Mean Correlation', ascending=False).head(15)

    if not cross_comparisons.empty:
        for _, row in cross_comparisons.iterrows():
            print(f"  {row['Probe 1']:<40} <-> {row['Probe 2']:<40} r={row['Mean Correlation']:.3f}")

    # Low agreement (<0.3)
    low_agreement = pairwise_df[pairwise_df['Mean Correlation'] < 0.3]
    if not low_agreement.empty:
        print("\n⚠️  LOW AGREEMENT PROBE PAIRS (r < 0.3):")
        print("-" * 80)
        for _, row in low_agreement.iterrows():
            print(f"  {row['Probe 1']:<40} <-> {row['Probe 2']:<40} r={row['Mean Correlation']:.3f}")

def main():
    print("="*80)
    print("COMPREHENSIVE PROBE AGREEMENT ANALYSIS (CENTERED SCORES)")
    print("Including Logit Lens at Multiple Layer Ranges")
    print("="*80)
    print("\nNOTE: All emotion scores have been CENTERED before computing correlations.")
    print("Centering removes shared sentence-level signal by subtracting the mean")
    print("across all emotions at each token position, highlighting emotion-specific deviations.")

    # Load conversations
    pkl_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
    print(f"\nLoading conversations from: {pkl_path}")

    conversations = load_conversation_data(pkl_path)
    print(f"Loaded {len(conversations)} conversations")

    # Analyze all probes
    print("\nAnalyzing probes with centered scores...")
    probe_data, probe_stats, pairwise_df = analyze_all_probes(conversations)

    # Print results
    print_probe_statistics(probe_stats)
    print_pairwise_agreements(pairwise_df)
    analyze_probe_clusters(pairwise_df)

    # Save results
    output_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard')

    # Save pairwise correlations
    pairwise_output = output_dir / 'probe_pairwise_correlations_centered.csv'
    pairwise_df.to_csv(pairwise_output, index=False)
    print(f"\n💾 Saved pairwise correlations to: {pairwise_output}")

    # Save probe statistics
    stats_df = pd.DataFrame.from_dict(probe_stats, orient='index')
    stats_output = output_dir / 'probe_individual_stats_centered.csv'
    stats_df.to_csv(stats_output)
    print(f"💾 Saved probe statistics to: {stats_output}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE (CENTERED SCORES)")
    print("="*80)

if __name__ == '__main__':
    main()
