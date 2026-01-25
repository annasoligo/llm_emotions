#!/usr/bin/env python3
"""
Quick patch script to add global_token_index field to existing preprocessed data.
"""
import pickle
from pathlib import Path

def main():
    input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl')
    output_path = input_path  # Overwrite in place

    print("="*80)
    print("PATCHING PICKLE FILE: Adding global_token_index")
    print("="*80)
    print(f"\nFile: {input_path}")

    # Load data
    print("\nLoading data...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    total_convs = len(data['conversations'])
    print(f"  Total conversations: {total_convs}")

    # Add global_token_index field to all sentences
    print("\nAdding global_token_index to sentences...")
    total_sentences = 0
    patched_sentences = 0

    for conv in data['conversations']:
        for sent in conv['sentences']:
            total_sentences += 1
            if 'global_token_index' not in sent:
                sent['global_token_index'] = sent['start_token']
                patched_sentences += 1

    print(f"  Total sentences: {total_sentences}")
    print(f"  Patched sentences: {patched_sentences}")
    print(f"  Already had field: {total_sentences - patched_sentences}")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved ({file_size_mb:.1f} MB)")

    print("\n" + "="*80)
    print("✓ PATCH COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
