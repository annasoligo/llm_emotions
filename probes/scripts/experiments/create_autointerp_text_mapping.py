#!/usr/bin/env python3
"""Create text mapping file for controlled variation auto-interpretation.

Maps pair_XXXX IDs (from H5) to conversation text (from JSONL).
"""

import argparse
import h5py
import json
from pathlib import Path


def create_text_mapping(
    h5_path: Path,
    jsonl_path: Path,
    output_path: Path,
):
    """Create text mapping file."""
    print(f"Loading H5 metadata from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        metadata_json = f['metadata'][()]
        if isinstance(metadata_json, bytes):
            metadata = json.loads(metadata_json)
        else:
            metadata = list(metadata_json)

    print(f"Loaded {len(metadata)} H5 metadata entries")

    # Load JSONL conversations
    print(f"Loading conversations from: {jsonl_path}")
    conv_data = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                conv_data[conv['id']] = conv

    print(f"Loaded {len(conv_data)} conversations")

    # Create mapping
    print("Creating pair_id -> text mapping...")
    output_data = []

    for meta in metadata:
        pair_id = meta['id']  # e.g., pair_0000
        emotional_id = meta.get('emotional_id', meta.get('id'))  # e.g., conv_0000

        if emotional_id not in conv_data:
            print(f"Warning: {emotional_id} not found in conversations")
            continue

        conv = conv_data[emotional_id]

        # Extract text from messages
        messages = conv.get('messages', [])
        user_texts = []
        asst_texts = []
        all_texts = []

        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', '')

            if role == 'user':
                user_texts.append(content)
            elif role == 'assistant':
                asst_texts.append(content)

            all_texts.append(content)

        # Create entry with pair_id as the ID
        entry = {
            'id': pair_id,  # Use pair_id so it matches H5
            'user_emotion': conv.get('user_emotion', meta.get('emotion', 'unknown')),
            'asst_emotion': conv.get('asst_emotion', 'neutral'),
            'topic': conv.get('topic', meta.get('topic', 'unknown')),
            'anchor_set': conv.get('anchor_set', meta.get('anchor_set')),
            'isolation_type': conv.get('isolation_type', 'unknown'),
            'neutral_text': '\n'.join(all_texts),  # For compatibility
            'emotional_text': '\n'.join(all_texts),
            'user_text': '\n'.join(user_texts),
            'asst_text': '\n'.join(asst_texts),
            'tier': 'conversation',  # For compatibility
            'intensity': None,
        }

        output_data.append(entry)

    # Save mapping
    print(f"Saving {len(output_data)} entries to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

    print("Done!")
    print(f"Sample entry IDs: {[e['id'] for e in output_data[:5]]}")


def main():
    parser = argparse.ArgumentParser(
        description="Create text mapping for auto-interpretation"
    )
    parser.add_argument(
        '--h5',
        type=Path,
        required=True,
        help='H5 file with pair_XXXX IDs'
    )
    parser.add_argument(
        '--jsonl',
        type=Path,
        required=True,
        help='JSONL file with conv_XXXX IDs'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file with pair_XXXX IDs'
    )

    args = parser.parse_args()

    create_text_mapping(args.h5, args.jsonl, args.output)


if __name__ == '__main__':
    main()
