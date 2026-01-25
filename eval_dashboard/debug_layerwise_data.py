#!/usr/bin/env python3
"""Debug script to check what layerwise data is actually in the pickle file."""
import pickle
from pathlib import Path

def main():
    input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')

    print("Loading data...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Total conversations: {len(data['conversations'])}")

    # Check first conversation
    conv = data['conversations'][0]
    print(f"\nFirst conversation keys: {conv.keys()}")

    # Check for layerwise data
    layerwise_keys = [k for k in conv.keys() if 'by_layer' in k]
    print(f"\nLayerwise keys found: {layerwise_keys}")

    if layerwise_keys:
        key = layerwise_keys[0]
        print(f"\nChecking key: {key}")
        per_layer_data = conv[key]
        print(f"  Type: {type(per_layer_data)}")
        print(f"  Sentence IDs: {list(per_layer_data.keys())[:5]}...")

        # Check first sentence
        first_sent_id = list(per_layer_data.keys())[0]
        sent_data = per_layer_data[first_sent_id]
        print(f"\n  First sentence ({first_sent_id}) layers: {list(sent_data.keys())}")
        print(f"    Layer range: {min(sent_data.keys())} - {max(sent_data.keys())}")

        # Check a layer's data
        first_layer = list(sent_data.keys())[0]
        layer_scores = sent_data[first_layer]
        print(f"\n    Layer {first_layer} scores shape: {len(layer_scores)}")
        print(f"    Layer {first_layer} scores: {layer_scores}")
    else:
        print("No layerwise data found!")

    # Check probe_configs
    if 'probe_configs' in data:
        print(f"\nProbe configs: {data['probe_configs']}")

if __name__ == "__main__":
    main()
