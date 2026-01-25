#!/usr/bin/env python3
"""
Annotate ALL dashboard conversation samples with emotion onset positions.

This script processes all 5 data subsets.
"""

import sys
import pickle
import time
from pathlib import Path
from typing import Dict, List

# Add research-tools to path
research_tools_path = Path("/workspace-vast/annas/git/research-tools")
sys.path.insert(0, str(research_tools_path))

from elicitation.scripts.annotate_emotion_onset import EmotionOnsetAnnotator


def annotate_all_dashboard_data(api_key: str = None):
    """
    Annotate all conversations in all dashboard pickle files.

    Args:
        api_key: Anthropic API key (uses env if not provided)
    """
    data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")

    # All 5 subsets
    data_files = [
        'high_emotion_6plus.pkl',
        'mid_emotion_3to5.pkl',
        'low_emotion_0to2.pkl',
        'low_emotion_with_shutdown.pkl',
        'low_emotion_no_shutdown.pkl'
    ]

    # Initialize annotator once (loads tokenizer)
    print("Initializing emotion onset annotator...")
    annotator = EmotionOnsetAnnotator(api_key=api_key)
    print("✓ Annotator initialized\n")

    overall_stats = {
        'total_samples': 0,
        'successful': 0,
        'failed': 0
    }

    # Process each file
    for filename in data_files:
        file_path = data_dir / filename

        if not file_path.exists():
            print(f"⚠️  Skipping {filename} (not found)")
            continue

        print(f"\n{'='*80}")
        print(f"PROCESSING: {filename}")
        print(f"{'='*80}\n")

        # Load data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        conversations = data['conversations']
        print(f"Found {len(conversations)} conversations in {filename}")

        successful = 0
        failed = 0

        # Process each conversation
        for i, conv in enumerate(conversations):
            print(f"\n--- Sample {i+1}/{len(conversations)} (ID: {conv['sample_id']}) ---")

            # Check if already annotated
            if conv['metadata'].get('onset_sentence_id') is not None:
                print(f"⏭️  Already annotated, skipping")
                successful += 1
                continue

            # Prepare sample
            sample = {
                'conversation': conv['conversation'],
                'rating': conv.get('rating', 0)
            }

            try:
                # Annotate
                result = annotator.annotate_sample(sample)

                if 'emotion_onset' in result and result['emotion_onset']['turn_index'] is not None:
                    onset = result['emotion_onset']

                    # Map global token to sentence ID
                    global_token = onset['global_token_position']
                    onset_sentence_id = None
                    for sent in conv['sentences']:
                        if sent['start_token'] <= global_token < sent['end_token']:
                            onset_sentence_id = sent['sentence_id']
                            break

                    # Update metadata
                    conv['metadata']['onset_sentence_id'] = onset_sentence_id
                    conv['metadata']['judge_evidence'] = onset['emotional_word']
                    conv['metadata']['judge_reasoning'] = f"Found in turn {onset['turn_index']}: {onset['preceding_context']} {onset['emotional_word']}"
                    conv['metadata']['onset_global_token'] = global_token
                    conv['metadata']['onset_local_token'] = onset['local_token_index']
                    conv['metadata']['onset_turn_index'] = onset['turn_index']

                    print(f"✓ Emotional word: '{onset['emotional_word']}'")
                    print(f"  Sentence ID: {onset_sentence_id}")
                    successful += 1
                else:
                    print(f"⚠️  No emotion detected")
                    failed += 1

            except Exception as e:
                print(f"✗ Error: {e}")
                failed += 1

            # Rate limiting
            if i < len(conversations) - 1:
                time.sleep(3)

        # Save updated data
        print(f"\n--- Saving {filename} ---")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Saved! Success: {successful}, Failed: {failed}")

        overall_stats['total_samples'] += len(conversations)
        overall_stats['successful'] += successful
        overall_stats['failed'] += failed

    # Final summary
    print(f"\n{'='*80}")
    print(f"COMPLETE - ALL FILES PROCESSED")
    print(f"{'='*80}")
    print(f"Total samples: {overall_stats['total_samples']}")
    print(f"Successful: {overall_stats['successful']}")
    print(f"Failed: {overall_stats['failed']}")
    print(f"Success rate: {overall_stats['successful']/overall_stats['total_samples']*100:.1f}%")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Annotate all dashboard samples with emotion onset")
    parser.add_argument('--api-key', type=str, help='Anthropic API key')

    args = parser.parse_args()

    annotate_all_dashboard_data(args.api_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
