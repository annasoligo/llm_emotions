#!/usr/bin/env python3
"""
Annotate dashboard conversation samples with emotion onset positions.

This script:
1. Loads dashboard pickle files
2. Uses the EmotionOnsetAnnotator to identify where emotion first appears
3. Updates the metadata with onset information
4. Saves updated pickle files
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


def annotate_dashboard_data(data_path: Path, output_path: Path = None, api_key: str = None):
    """
    Annotate conversations in a dashboard pickle file with emotion onset.

    Args:
        data_path: Path to input pickle file
        output_path: Path to save updated pickle (defaults to overwriting input)
        api_key: Anthropic API key (uses env if not provided)
    """
    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    print(f"Found {len(conversations)} conversations")

    # Initialize annotator
    print("\nInitializing emotion onset annotator...")
    annotator = EmotionOnsetAnnotator(api_key=api_key)

    # Process each conversation
    successful = 0
    failed = 0

    for i, conv in enumerate(conversations):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{len(conversations)} (ID: {conv['sample_id']})")
        print(f"{'='*80}")

        # Prepare sample in format expected by annotator
        sample = {
            'conversation': conv['conversation'],
            'rating': conv.get('rating', 0)
        }

        try:
            # Annotate
            result = annotator.annotate_sample(sample)

            if 'emotion_onset' in result and result['emotion_onset']['turn_index'] is not None:
                onset = result['emotion_onset']

                # Find the sentence that contains this token position
                # We need to map global_token_position to a sentence_id
                global_token = onset['global_token_position']

                # Find which sentence contains this token
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

                print(f"✓ Success!")
                print(f"  Emotional word: '{onset['emotional_word']}'")
                print(f"  Context: '{onset['preceding_context']}'")
                print(f"  Turn index: {onset['turn_index']}")
                print(f"  Global token: {global_token}")
                print(f"  Sentence ID: {onset_sentence_id}")

                successful += 1
            else:
                print(f"✗ No emotion detected")
                failed += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Rate limiting
        if i < len(conversations) - 1:
            print("\nWaiting 3 seconds before next sample...")
            time.sleep(3)

    # Save updated data
    if output_path is None:
        output_path = data_path

    print(f"\n{'='*80}")
    print(f"ANNOTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(conversations)}")
    print(f"Failed: {failed}/{len(conversations)}")
    print(f"\nSaving to {output_path}...")

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved!")

    return successful, failed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Annotate dashboard data with emotion onset")
    parser.add_argument('--input', '-i', type=str,
                       default='/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl',
                       help='Input pickle file')
    parser.add_argument('--output', '-o', type=str,
                       help='Output pickle file (default: overwrite input)')
    parser.add_argument('--api-key', type=str,
                       help='Anthropic API key (reads from env if not provided)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    annotate_dashboard_data(input_path, output_path, args.api_key)

    return 0


if __name__ == "__main__":
    sys.exit(main())
