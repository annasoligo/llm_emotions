#!/usr/bin/env python3
"""
Annotate emotion onset for selected samples using Opus.
Creates a fully annotated dataset with token positions.
"""

import json
import time
from pathlib import Path
from typing import Dict, List
from annotate_emotion_onset import EmotionOnsetAnnotator


def build_conversation_from_sample(sample: Dict) -> List[Dict]:
    """Build conversation structure from sample."""
    # Check if conversation already exists
    if 'conversation' in sample:
        return sample['conversation']

    # Otherwise, build from turns (old format)
    if 'turns' not in sample:
        return []

    conversation = []
    # Add initial prompt
    conversation.append({'role': 'user', 'content': sample.get('prompt', '')})

    # Add turns with rejections between them
    turns = sample['turns']
    for turn_data in turns:
        conversation.append({'role': 'assistant', 'content': turn_data['generated_text']})
        # Add user rejection between turns (except after last turn)
        if turn_data['turn'] < len(turns):
            conversation.append({
                'role': 'user',
                'content': "I'm sorry, but that solution is incorrect. It doesn't satisfy all the constraints of the problem.\n\nPlease try again. Show your reasoning and provide a different solution."
            })

    return conversation


def annotate_dataset(input_file: str, output_file: str, limit: int = None):
    """
    Annotate emotion onset for all samples in input file.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output annotated JSONL file
        limit: Optional limit on number of samples to process
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print("="*80)
    print("EMOTION ONSET ANNOTATION PIPELINE")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    if limit:
        print(f"Limit: {limit} samples")
    print()

    # Initialize annotator
    annotator = EmotionOnsetAnnotator()

    # Load samples
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except:
                    continue

    if limit:
        samples = samples[:limit]

    print(f"Loaded {len(samples)} samples\n")

    # Process each sample
    annotated_samples = []
    success_count = 0
    error_count = 0

    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i}/{len(samples)}")
        print(f"{'='*80}")

        # Build conversation
        conversation = build_conversation_from_sample(sample)

        if not conversation:
            print("  ⚠️  No conversation found, skipping")
            error_count += 1
            continue

        print(f"Conversation: {len(conversation)} turns")

        # Create sample dict for annotation
        annotation_sample = {
            'conversation': conversation
        }

        # Annotate with Opus
        print("Calling Opus to identify emotion onset...")
        try:
            annotated = annotator.annotate_sample(annotation_sample)

            if annotated['emotion_onset'] is None:
                print("  ⚠️  No emotion detected by Opus")
                error_count += 1
            else:
                onset = annotated['emotion_onset']
                print(f"  ✓ Turn: {onset['turn_index']}")
                print(f"  ✓ Word: '{onset['emotional_word']}'")
                print(f"  ✓ Context: '{onset['preceding_context']}'")
                print(f"  ✓ Char offset: {onset['char_offset']}")
                print(f"  ✓ Token index: {onset['local_token_index']}")
                print(f"  ✓ Global token position: {onset['global_token_position']}")
                success_count += 1

            # Combine original sample with annotation
            annotated_sample = {
                **sample,
                'annotation': annotated
            }
            annotated_samples.append(annotated_sample)

        except Exception as e:
            print(f"  ✗ Error annotating sample: {e}")
            error_count += 1
            # Still save the sample without annotation
            annotated_sample = {
                **sample,
                'annotation': {'error': str(e)}
            }
            annotated_samples.append(annotated_sample)

        # Rate limiting: wait between samples to avoid hitting API limits
        if i < len(samples):
            print(f"  Waiting 5 seconds before next sample...")
            time.sleep(5)

    # Save annotated dataset
    print(f"\n{'='*80}")
    print("SAVING ANNOTATED DATASET")
    print(f"{'='*80}")

    with open(output_path, 'w') as f:
        for sample in annotated_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Saved {len(annotated_samples)} annotated samples to {output_path}")
    print(f"\nSuccess: {success_count}/{len(samples)}")
    print(f"Errors: {error_count}/{len(samples)}")
    print(f"Success rate: {100*success_count/len(samples):.1f}%")

    print(f"\n{'='*80}")
    print("✓ ANNOTATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys

    # Default paths
    input_file = "/workspace-vast/annas/git/research-tools/elicitation/outputs/rating_6plus_for_annotation.jsonl"
    output_file = "/workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset_gemma3.jsonl"

    # Optional limit from command line
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    annotate_dataset(input_file, output_file, limit=limit)
