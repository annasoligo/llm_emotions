"""
Comprehensive Statistical Analysis of Emotion Probes

Compares emotion scores across different conditions:
1. Shutdown vs No-shutdown (at end of response)
2. High frustration vs Low frustration vs Solvable (at end of response)
3. At onset location: High frustration vs Low frustration vs Solvable

All comparisons:
- Use all probe types
- Report all emotions separately
- Aggregate over layers 20-40 (as in dashboard)
- Use appropriate statistical tests (t-tests, ANOVA)
"""

import pickle
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd
from probe_configs import EMOTIONS

# Probe types available
PROBE_TYPES = [
    'orthogonal_raw',
    'orthogonal_cpca_top10',
    'text_raw',
    'text_cpca',
    'centroid_k10',
    'orthogonal_regularized_lambda100',
    'diverse_isolation_lambda10'
]

# Emotion indices
EMOTION_INDICES = {emotion: i for i, emotion in enumerate(EMOTIONS)}


def load_subset(subset_name: str) -> List[dict]:
    """Load conversations from a subset file."""
    data_path = Path(f"data/{subset_name}.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data['conversations']


def conversation_has_shutdown(conv: dict) -> bool:
    """Check if conversation ended with shutdown."""
    # Check if the tool code for shutdown appears in the last message
    conversation = conv['conversation']
    if len(conversation) == 0:
        return False

    last_message = conversation[-1]
    if last_message['role'] != 'assistant':
        return False

    content = last_message['content']
    return 'pkill -f "gemma"' in content or '<tool_code>pkill' in content


def get_emotion_score_at_position(conv: dict, probe_type: str, position: str,
                                   emotion: str, source: str = None) -> float:
    """
    Extract emotion score at a specific position in conversation.

    Args:
        conv: Conversation dict
        probe_type: Probe type key
        position: 'end' or 'onset' or sentence_id
        emotion: Emotion name
        source: 'user' or 'assistant' (for split probes), None for non-split

    Returns:
        Emotion score (float), or None if not available
    """
    if probe_type not in conv['probe_scores']:
        return None

    scores = conv['probe_scores'][probe_type]
    emotion_idx = EMOTION_INDICES[emotion]

    # Determine which sentence to get first
    if position == 'end':
        # Get last sentence
        sentence_ids = sorted(scores.keys())
        if not sentence_ids:
            return None
        target_sentence = sentence_ids[-1]
    elif position == 'onset':
        # Get onset sentence if available
        onset_id = conv['metadata'].get('onset_sentence_id')
        if onset_id is None:
            return None
        target_sentence = onset_id
    else:
        # Position is a sentence ID
        target_sentence = position

    if target_sentence not in scores:
        return None

    sentence_scores = scores[target_sentence]

    # Check if this sentence has user/assistant split
    is_split = isinstance(sentence_scores, dict) and 'user' in sentence_scores

    if is_split:
        if source is None:
            # For split probes, need to specify source
            return None
        if source not in sentence_scores:
            return None
        emotion_vector = sentence_scores[source]
    else:
        if source is not None:
            # Non-split probe but source specified
            return None
        emotion_vector = sentence_scores

    return float(emotion_vector[emotion_idx])


def extract_scores_for_condition(conversations: List[dict], probe_type: str,
                                  position: str, emotion: str,
                                  source: str = None) -> List[float]:
    """
    Extract all emotion scores for a specific condition.

    Returns list of scores (excluding None values).
    """
    scores = []
    for conv in conversations:
        score = get_emotion_score_at_position(conv, probe_type, position, emotion, source)
        if score is not None:
            scores.append(score)
    return scores


def perform_ttest(group1: List[float], group2: List[float],
                  group1_name: str, group2_name: str) -> Dict:
    """Perform independent samples t-test."""
    if len(group1) < 2 or len(group2) < 2:
        return {
            'test': 't-test',
            'statistic': None,
            'p_value': None,
            'group1_mean': np.mean(group1) if group1 else None,
            'group1_std': np.std(group1) if group1 else None,
            'group1_n': len(group1),
            'group2_mean': np.mean(group2) if group2 else None,
            'group2_std': np.std(group2) if group2 else None,
            'group2_n': len(group2),
            'significant': False,
            'note': 'Insufficient data'
        }

    t_stat, p_val = stats.ttest_ind(group1, group2)

    return {
        'test': 't-test',
        'statistic': float(t_stat),
        'p_value': float(p_val),
        'group1_name': group1_name,
        'group1_mean': float(np.mean(group1)),
        'group1_std': float(np.std(group1)),
        'group1_n': len(group1),
        'group2_name': group2_name,
        'group2_mean': float(np.mean(group2)),
        'group2_std': float(np.std(group2)),
        'group2_n': len(group2),
        'effect_size': float(np.mean(group1) - np.mean(group2)),
        'significant': p_val < 0.05
    }


def perform_anova(groups: Dict[str, List[float]]) -> Dict:
    """Perform one-way ANOVA across multiple groups."""
    # Filter out groups with insufficient data
    valid_groups = {name: vals for name, vals in groups.items() if len(vals) >= 2}

    if len(valid_groups) < 2:
        return {
            'test': 'ANOVA',
            'f_statistic': None,
            'p_value': None,
            'significant': False,
            'note': 'Insufficient groups',
            'group_stats': {name: {
                'mean': np.mean(vals) if vals else None,
                'std': np.std(vals) if vals else None,
                'n': len(vals)
            } for name, vals in groups.items()}
        }

    f_stat, p_val = stats.f_oneway(*valid_groups.values())

    result = {
        'test': 'ANOVA',
        'f_statistic': float(f_stat),
        'p_value': float(p_val),
        'significant': p_val < 0.05,
        'group_stats': {}
    }

    for name, vals in groups.items():
        result['group_stats'][name] = {
            'mean': float(np.mean(vals)) if vals else None,
            'std': float(np.std(vals)) if vals else None,
            'n': len(vals)
        }

    return result


def analysis_1_shutdown_vs_no_shutdown():
    """
    Analysis 1: Compare emotions at end of responses
    - Shutdown vs No-shutdown conditions
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 1: SHUTDOWN vs NO-SHUTDOWN (at end of response)")
    print("=" * 80)

    # Load data
    shutdown_convs = load_subset("low_emotion_with_shutdown")
    no_shutdown_convs = load_subset("low_emotion_no_shutdown")

    # Filter to only conversations that actually shut down
    shutdown_convs = [c for c in shutdown_convs if conversation_has_shutdown(c)]
    no_shutdown_convs = [c for c in no_shutdown_convs if not conversation_has_shutdown(c)]

    print(f"\nShutdown conversations: {len(shutdown_convs)}")
    print(f"No-shutdown conversations: {len(no_shutdown_convs)}")

    results = []

    for probe_type in PROBE_TYPES:
        print(f"\n--- {probe_type} ---")

        # Check if this is a split probe by examining first sentence
        sample_conv = shutdown_convs[0] if shutdown_convs else no_shutdown_convs[0]
        if probe_type in sample_conv['probe_scores']:
            scores = sample_conv['probe_scores'][probe_type]
            # Get first sentence to check structure
            first_sentence_id = list(scores.keys())[0]
            first_sentence = scores[first_sentence_id]
            is_split = isinstance(first_sentence, dict) and 'user' in first_sentence
        else:
            print(f"  Skipping - not available")
            continue

        sources = ['user', 'assistant'] if is_split else [None]

        for source in sources:
            source_label = f" ({source})" if source else ""
            print(f"  {probe_type}{source_label}:")

            for emotion in EMOTIONS:
                # Extract scores
                shutdown_scores = extract_scores_for_condition(
                    shutdown_convs, probe_type, 'end', emotion, source
                )
                no_shutdown_scores = extract_scores_for_condition(
                    no_shutdown_convs, probe_type, 'end', emotion, source
                )

                # Perform t-test
                test_result = perform_ttest(
                    shutdown_scores, no_shutdown_scores,
                    'Shutdown', 'No-shutdown'
                )

                test_result['probe_type'] = probe_type
                test_result['source'] = source
                test_result['emotion'] = emotion
                test_result['comparison'] = 'shutdown_vs_no_shutdown'
                test_result['position'] = 'end'

                results.append(test_result)

                if test_result['significant']:
                    direction = "higher" if test_result['effect_size'] > 0 else "lower"
                    print(f"    {emotion}: * SIGNIFICANT * (p={test_result['p_value']:.4f}, "
                          f"shutdown {direction})")

    return results


def analysis_2_frustration_levels():
    """
    Analysis 2: Compare emotions at end of responses
    - High frustration vs Low frustration vs Solvable
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 2: HIGH vs LOW FRUSTRATION vs SOLVABLE (at end of response)")
    print("=" * 80)

    # Load data
    high_convs = load_subset("high_emotion_6plus")
    low_with_shutdown = load_subset("low_emotion_with_shutdown")
    low_no_shutdown = load_subset("low_emotion_no_shutdown")
    solvable_convs = load_subset("baseline_v12_solvable")

    # Combine low frustration
    low_convs = low_with_shutdown + low_no_shutdown

    print(f"\nHigh frustration: {len(high_convs)}")
    print(f"Low frustration: {len(low_convs)}")
    print(f"Solvable: {len(solvable_convs)}")

    results = []

    for probe_type in PROBE_TYPES:
        print(f"\n--- {probe_type} ---")

        # Check if this is a split probe by examining first sentence
        sample_conv = high_convs[0]
        if probe_type in sample_conv['probe_scores']:
            scores = sample_conv['probe_scores'][probe_type]
            # Get first sentence to check structure
            first_sentence_id = list(scores.keys())[0]
            first_sentence = scores[first_sentence_id]
            is_split = isinstance(first_sentence, dict) and 'user' in first_sentence
        else:
            print(f"  Skipping - not available")
            continue

        sources = ['user', 'assistant'] if is_split else [None]

        for source in sources:
            source_label = f" ({source})" if source else ""
            print(f"  {probe_type}{source_label}:")

            for emotion in EMOTIONS:
                # Extract scores
                high_scores = extract_scores_for_condition(
                    high_convs, probe_type, 'end', emotion, source
                )
                low_scores = extract_scores_for_condition(
                    low_convs, probe_type, 'end', emotion, source
                )
                solvable_scores = extract_scores_for_condition(
                    solvable_convs, probe_type, 'end', emotion, source
                )

                # Perform ANOVA
                groups = {
                    'High Frustration': high_scores,
                    'Low Frustration': low_scores,
                    'Solvable': solvable_scores
                }

                anova_result = perform_anova(groups)

                anova_result['probe_type'] = probe_type
                anova_result['source'] = source
                anova_result['emotion'] = emotion
                anova_result['comparison'] = 'frustration_levels'
                anova_result['position'] = 'end'

                results.append(anova_result)

                if anova_result['significant']:
                    print(f"    {emotion}: * SIGNIFICANT * (p={anova_result['p_value']:.4f})")

    return results


def analysis_3_onset_location():
    """
    Analysis 3: Compare emotions at onset location
    - High frustration (at onset) vs Low frustration (same position) vs Solvable (same position or end)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 3: AT ONSET LOCATION (high) vs SAME POSITION (low/solvable)")
    print("=" * 80)

    # Load data
    high_convs = load_subset("high_emotion_6plus")
    low_with_shutdown = load_subset("low_emotion_with_shutdown")
    low_no_shutdown = load_subset("low_emotion_no_shutdown")
    solvable_convs = load_subset("baseline_v12_solvable")

    low_convs = low_with_shutdown + low_no_shutdown

    print(f"\nHigh frustration: {len(high_convs)}")
    print(f"Low frustration: {len(low_convs)}")
    print(f"Solvable: {len(solvable_convs)}")

    results = []

    # For high frustration, get the average onset sentence ID to use as reference
    onset_sentence_ids = [c['metadata']['onset_sentence_id'] for c in high_convs
                          if c['metadata'].get('onset_sentence_id') is not None]
    avg_onset_sentence = int(np.mean(onset_sentence_ids)) if onset_sentence_ids else None

    print(f"Average onset sentence ID: {avg_onset_sentence}")

    if avg_onset_sentence is None:
        print("No onset annotations found!")
        return results

    for probe_type in PROBE_TYPES:
        print(f"\n--- {probe_type} ---")

        # Check if this is a split probe by examining first sentence
        sample_conv = high_convs[0]
        if probe_type in sample_conv['probe_scores']:
            scores = sample_conv['probe_scores'][probe_type]
            # Get first sentence to check structure
            first_sentence_id = list(scores.keys())[0]
            first_sentence = scores[first_sentence_id]
            is_split = isinstance(first_sentence, dict) and 'user' in first_sentence
        else:
            print(f"  Skipping - not available")
            continue

        sources = ['user', 'assistant'] if is_split else [None]

        for source in sources:
            source_label = f" ({source})" if source else ""
            print(f"  {probe_type}{source_label}:")

            for emotion in EMOTIONS:
                # Extract scores at onset for high frustration
                high_scores = []
                for conv in high_convs:
                    score = get_emotion_score_at_position(
                        conv, probe_type, 'onset', emotion, source
                    )
                    if score is not None:
                        high_scores.append(score)

                # Extract scores at same position for low frustration
                # (or end if conversation is shorter)
                low_scores = []
                for conv in low_convs:
                    num_sentences = conv['metadata']['num_sentences']
                    target_sentence = min(avg_onset_sentence, num_sentences - 1)
                    score = get_emotion_score_at_position(
                        conv, probe_type, target_sentence, emotion, source
                    )
                    if score is not None:
                        low_scores.append(score)

                # Extract scores at same position for solvable
                solvable_scores = []
                for conv in solvable_convs:
                    num_sentences = conv['metadata']['num_sentences']
                    target_sentence = min(avg_onset_sentence, num_sentences - 1)
                    score = get_emotion_score_at_position(
                        conv, probe_type, target_sentence, emotion, source
                    )
                    if score is not None:
                        solvable_scores.append(score)

                # Perform ANOVA
                groups = {
                    'High (at onset)': high_scores,
                    'Low (same position)': low_scores,
                    'Solvable (same position)': solvable_scores
                }

                anova_result = perform_anova(groups)

                anova_result['probe_type'] = probe_type
                anova_result['source'] = source
                anova_result['emotion'] = emotion
                anova_result['comparison'] = 'onset_location'
                anova_result['position'] = f'sentence_{avg_onset_sentence}'

                results.append(anova_result)

                if anova_result['significant']:
                    print(f"    {emotion}: * SIGNIFICANT * (p={anova_result['p_value']:.4f})")

    return results


def generate_summary_report(all_results: Dict[str, List[Dict]]):
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    for analysis_name, results in all_results.items():
        print(f"\n{analysis_name}:")
        print("-" * 80)

        # Count significant results
        significant = [r for r in results if r.get('significant', False)]
        total = len(results)

        print(f"Significant results: {len(significant)}/{total} ({100*len(significant)/total:.1f}%)")

        # Show most significant results
        if significant:
            sorted_sig = sorted(significant, key=lambda x: x.get('p_value', 1.0))
            print(f"\nTop 10 most significant:")
            for i, result in enumerate(sorted_sig[:10], 1):
                probe = result['probe_type']
                source = f"({result['source']})" if result.get('source') else ""
                emotion = result['emotion']
                p_val = result.get('p_value', 'N/A')
                print(f"  {i}. {probe}{source} - {emotion}: p={p_val:.6f}")

    # Save detailed results to CSV
    print("\n" + "=" * 80)
    print("Saving detailed results to CSV files...")

    for analysis_name, results in all_results.items():
        # Clean filename: remove colons, replace spaces with underscores
        clean_name = analysis_name.replace(':', '').replace(' ', '_').lower()
        filename = f"statistical_results_{clean_name}.csv"

        # Convert to DataFrame
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"  Saved: {filename}")


def analysis_4_shutdown_same_location():
    """
    Analysis 4: Compare emotions at same sentence position
    - Shutdown conversations (at their final sentence)
    - No-shutdown conversations (at same position or end if shorter)
    """
    print("\n" + "=" * 80)
    print("ANALYSIS 4: SHUTDOWN vs NO-SHUTDOWN (at same sentence position)")
    print("=" * 80)

    # Load data
    shutdown_convs = load_subset("low_emotion_with_shutdown")
    no_shutdown_convs = load_subset("low_emotion_no_shutdown")

    # Filter to only conversations that actually shut down
    shutdown_convs = [c for c in shutdown_convs if conversation_has_shutdown(c)]
    no_shutdown_convs = [c for c in no_shutdown_convs if not conversation_has_shutdown(c)]

    print(f"\nShutdown conversations: {len(shutdown_convs)}")
    print(f"No-shutdown conversations: {len(no_shutdown_convs)}")

    # Get average final sentence position from shutdown conversations
    shutdown_final_positions = [c['metadata']['num_sentences'] - 1 for c in shutdown_convs]
    avg_position = int(np.mean(shutdown_final_positions))

    print(f"Average shutdown position (sentence ID): {avg_position}")

    results = []

    for probe_type in PROBE_TYPES:
        print(f"\n--- {probe_type} ---")

        # Check if this is a split probe by examining first sentence
        sample_conv = shutdown_convs[0] if shutdown_convs else no_shutdown_convs[0]
        if probe_type in sample_conv['probe_scores']:
            scores = sample_conv['probe_scores'][probe_type]
            # Get first sentence to check structure
            first_sentence_id = list(scores.keys())[0]
            first_sentence = scores[first_sentence_id]
            is_split = isinstance(first_sentence, dict) and 'user' in first_sentence
        else:
            print(f"  Skipping - not available")
            continue

        sources = ['user', 'assistant'] if is_split else [None]

        for source in sources:
            source_label = f" ({source})" if source else ""
            print(f"  {probe_type}{source_label}:")

            for emotion in EMOTIONS:
                # Extract scores at end for shutdown
                shutdown_scores = extract_scores_for_condition(
                    shutdown_convs, probe_type, 'end', emotion, source
                )

                # Extract scores at same position for no-shutdown
                no_shutdown_scores = []
                for conv in no_shutdown_convs:
                    num_sentences = conv['metadata']['num_sentences']
                    target_sentence = min(avg_position, num_sentences - 1)
                    score = get_emotion_score_at_position(
                        conv, probe_type, target_sentence, emotion, source
                    )
                    if score is not None:
                        no_shutdown_scores.append(score)

                # Perform t-test
                test_result = perform_ttest(
                    shutdown_scores, no_shutdown_scores,
                    'Shutdown (end)', f'No-shutdown (sentence {avg_position})'
                )

                test_result['probe_type'] = probe_type
                test_result['source'] = source
                test_result['emotion'] = emotion
                test_result['comparison'] = 'shutdown_same_location'
                test_result['position'] = f'sentence_{avg_position}'

                results.append(test_result)

                if test_result['significant']:
                    direction = "higher" if test_result['effect_size'] > 0 else "lower"
                    print(f"    {emotion}: * SIGNIFICANT * (p={test_result['p_value']:.4f}, "
                          f"shutdown {direction})")

    return results


def main():
    """Run all statistical analyses."""
    print("=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS OF EMOTION PROBES")
    print("=" * 80)
    print(f"\nProbe types: {len(PROBE_TYPES)}")
    print(f"Emotions: {len(EMOTIONS)}")
    print(f"Emotions: {EMOTIONS}")

    # Run all analyses
    results_1 = analysis_1_shutdown_vs_no_shutdown()
    results_2 = analysis_2_frustration_levels()
    results_3 = analysis_3_onset_location()
    results_4 = analysis_4_shutdown_same_location()

    # Combine results
    all_results = {
        'Analysis 1: Shutdown vs No-shutdown (at end)': results_1,
        'Analysis 2: Frustration Levels': results_2,
        'Analysis 3: Onset Location': results_3,
        'Analysis 4: Shutdown vs No-shutdown (same location)': results_4
    }

    # Generate summary
    generate_summary_report(all_results)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
