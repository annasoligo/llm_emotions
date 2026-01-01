#!/usr/bin/env python3
"""Debug script to check data loading."""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import pickle
from pathlib import Path

def test_loading():
    data_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data")

    print("Testing data loading...")
    print(f"Data directory: {data_dir}")
    print(f"Exists: {data_dir.exists()}")
    print()

    test_file = data_dir / "high_emotion_6plus.pkl"
    print(f"Testing file: {test_file}")
    print(f"Exists: {test_file.exists()}")
    print(f"Size: {test_file.stat().st_size / 1024 / 1024:.2f} MB")
    print()

    try:
        with open(test_file, 'rb') as f:
            data = pickle.load(f)

        print("✓ File loaded successfully!")
        print(f"Keys: {data.keys()}")

        if 'conversations' in data:
            convs = data['conversations']
            print(f"\n✓ Found {len(convs)} conversations")

            if convs:
                conv = convs[0]
                print(f"\nFirst conversation keys: {conv.keys()}")
                print(f"Sample ID: {conv.get('sample_id')}")
                print(f"Number of sentences: {len(conv.get('sentences', []))}")
                print(f"Probe scores keys: {list(conv.get('probe_scores', {}).keys())}")

                if conv.get('probe_scores'):
                    probe_key = list(conv['probe_scores'].keys())[0]
                    print(f"\nFirst probe: {probe_key}")
                    scores = conv['probe_scores'][probe_key]
                    print(f"Number of sentence scores: {len(scores)}")
                    if scores:
                        first_score = list(scores.values())[0]
                        print(f"First score type: {type(first_score)}")
                        if isinstance(first_score, dict):
                            print(f"Score keys: {first_score.keys()}")
        else:
            print("✗ No 'conversations' key in data!")

    except Exception as e:
        print(f"✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_loading()
