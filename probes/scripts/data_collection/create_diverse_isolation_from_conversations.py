#!/usr/bin/env python3
"""
Create improved isolation datasets using GLOBAL conversation activations.

KEY INSIGHT: We use global activations (where user+assistant are mixed) to test
if we can disentangle source-specific emotion representations from the same tokens.

IMPROVEMENT over controlled variation: In conversations2.h5, BOTH user and
assistant have varied emotions (not always neutral), removing the confound where
the probe just learns "is the other person neutral?"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm


def load_conversation_data(data_path: str) -> Dict:
    """Load global conversation activations."""
    print(f"Loading conversation data from {data_path}...")

    samples = {}

    with h5py.File(data_path, 'r') as f:
        metadata_list = json.loads(f['metadata'][()])

        for entry in tqdm(metadata_list):
            sample_id = entry['id']

            # Load GLOBAL activations (mixed user+assistant)
            global_acts = f[f'activations/{sample_id}/global'][:]  # (62, 5376)

            samples[sample_id] = {
                'activations': global_acts,
                'user_emotion': entry['user_emotion'],
                'asst_emotion': entry['asst_emotion'],
                'topic': entry.get('topic', 'unknown'),
            }

    print(f"Loaded {len(samples)} samples")
    return samples


def create_user_isolation_dataset(
    conversation_data: Dict,
    output_path: str,
):
    """
    Create user isolation dataset:
    - Label: user emotion
    - Activations: global (contains both user+assistant)
    - Key: assistant has VARIED emotions (not always neutral)
    """
    print("\nCreating USER isolation dataset from global activations...")

    samples = []

    for sample_id, data in tqdm(conversation_data.items()):
        user_emotion = data['user_emotion']
        asst_emotion = data['asst_emotion']

        # Only include samples where user is emotional
        if user_emotion == 'neutral':
            continue

        samples.append({
            'id': sample_id,
            'activations': data['activations'],  # GLOBAL activations
            'emotion': user_emotion,  # Target: user emotion
            'other_emotion': asst_emotion,  # Context: assistant emotion (VARIES!)
            'topic': data['topic'],
        })

    print(f"Created {len(samples)} samples")

    # Analyze diversity
    analyze_diversity(samples, "User", "Assistant")

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(samples)

    # Save to HDF5
    save_dataset(samples, output_path)


def create_assistant_isolation_dataset(
    conversation_data: Dict,
    output_path: str,
):
    """
    Create assistant isolation dataset:
    - Label: assistant emotion
    - Activations: global (contains both user+assistant)
    - Key: user has VARIED emotions (not always neutral)
    """
    print("\nCreating ASSISTANT isolation dataset from global activations...")

    samples = []

    for sample_id, data in tqdm(conversation_data.items()):
        user_emotion = data['user_emotion']
        asst_emotion = data['asst_emotion']

        # Only include samples where assistant is emotional
        if asst_emotion == 'neutral':
            continue

        samples.append({
            'id': sample_id,
            'activations': data['activations'],  # GLOBAL activations
            'emotion': asst_emotion,  # Target: assistant emotion
            'other_emotion': user_emotion,  # Context: user emotion (VARIES!)
            'topic': data['topic'],
        })

    print(f"Created {len(samples)} samples")

    # Analyze diversity
    analyze_diversity(samples, "Assistant", "User")

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(samples)

    # Save to HDF5
    save_dataset(samples, output_path)


def analyze_diversity(samples: List, target_name: str, other_name: str):
    """Analyze and print emotion distribution."""
    emotion_counts = {}
    other_emotion_counts = {}

    for sample in samples:
        emotion = sample['emotion']
        other = sample['other_emotion']

        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        other_emotion_counts[other] = other_emotion_counts.get(other, 0) + 1

    print(f"\n{target_name} emotion distribution (target):")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion}: {count}")

    print(f"\n{other_name} emotion distribution (other person - THE KEY!):")
    for emotion, count in sorted(other_emotion_counts.items()):
        pct = 100 * count / len(samples)
        print(f"  {emotion}: {count} ({pct:.1f}%)")

    # Count how many samples have non-neutral other person
    non_neutral_other = sum(1 for s in samples if s['other_emotion'] != 'neutral')
    pct_non_neutral = 100 * non_neutral_other / len(samples)

    print(f"\nDIVERSITY CHECK:")
    print(f"  Samples where {other_name} is NOT neutral: {non_neutral_other}/{len(samples)} ({pct_non_neutral:.1f}%)")
    print(f"  This is the improvement over controlled variation!")


def save_dataset(samples: List, output_path: str):
    """Save dataset to HDF5."""
    print(f"\nSaving to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Save activations by layer
        for layer_idx in tqdm(range(62), desc="Saving layers"):
            layer_acts = np.stack([s['activations'][layer_idx] for s in samples], axis=0)
            f.create_dataset(f'layer_{layer_idx}', data=layer_acts, compression='gzip')

        # Save metadata
        metadata_list = []
        for sample in samples:
            metadata_list.append({
                'id': sample['id'],
                'emotion': sample['emotion'],
                'other_emotion': sample['other_emotion'],
                'topic': sample['topic'],
            })

        f.create_dataset('metadata', data=json.dumps(metadata_list))

    print(f"✓ Saved {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Create diverse isolation datasets from conversation GLOBAL activations"
    )
    parser.add_argument(
        "--conversation-data",
        type=str,
        default="outputs/data/activations/conversations2.h5",
        help="Path to conversation data (with GLOBAL activations)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/activations/diverse_isolation",
        help="Output directory",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CREATING DIVERSE ISOLATION DATASETS FROM GLOBAL ACTIVATIONS")
    print("=" * 80)
    print()
    print("Key difference from controlled variation:")
    print("  - Controlled variation: Other person is ALWAYS neutral")
    print("  - This dataset: Other person has VARIED emotions")
    print()
    print("This tests if we can truly disentangle source-specific representations")
    print("from GLOBAL activations (where both user+assistant are mixed).")
    print("=" * 80)

    # Load data
    conversation_data = load_conversation_data(args.conversation_data)

    # Create user isolation dataset
    print("\n" + "=" * 80)
    create_user_isolation_dataset(
        conversation_data,
        output_path=str(output_dir / "user_isolation.h5"),
    )

    # Create assistant isolation dataset
    print("\n" + "=" * 80)
    create_assistant_isolation_dataset(
        conversation_data,
        output_path=str(output_dir / "assistant_isolation.h5"),
    )

    print("\n" + "=" * 80)
    print("✓ Done!")
    print(f"\nDatasets saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Compute orthogonal PCs for these datasets")
    print("2. Train probes with orthogonal regularization")
    print("3. Compare with controlled variation results")
    print("   - Hypothesis: Should see better performance since data is more realistic")
    print("   - Hypothesis: Source specificity might be harder (other person not always neutral)")


if __name__ == "__main__":
    main()
