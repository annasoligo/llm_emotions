#!/usr/bin/env python3
"""Check what emotion values we're actually getting in the layerwise data."""
import pickle
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("CHECKING EMOTION VALUES IN LAYERWISE DATA")
    print("="*80)

    # Load the new layerwise data
    layerwise_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    print(f"\nLoading: {layerwise_path}")

    with open(layerwise_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Total conversations: {len(data['conversations'])}")

    # Check first conversation
    conv = data['conversations'][0]
    print(f"\n{'='*80}")
    print(f"CONVERSATION 0 - Sample ID: {conv.get('sample_id', 'unknown')}")
    print(f"{'='*80}")

    # Check first sentence with layerwise data
    for i, sent in enumerate(conv['sentences'][:3]):
        print(f"\nSentence {i}:")
        print(f"  Text: {sent['text'][:80]}...")
        print(f"  Dominant emotion: {sent.get('dominant_emotion', 'N/A')}")

        # Check if we have axes data
        if 'axes' in sent:
            print(f"  Axes data present: YES")
            print(f"    Axes keys: {list(sent['axes'].keys())}")
            # Sample an emotion
            for emotion in ['happiness', 'sadness', 'anger']:
                if emotion in sent['axes']:
                    vals = sent['axes'][emotion]
                    if isinstance(vals, list) and len(vals) > 0:
                        print(f"    {emotion} axes (first 3): {vals[:3]}")
                    break

        # Check if we have layerwise data
        if 'layerwise' in sent:
            print(f"  Layerwise data present: YES")
            lw = sent['layerwise']

            # Check structure
            if 'logit_lens_mean' in lw:
                print(f"    logit_lens_mean present: YES")
                llm = lw['logit_lens_mean']
                print(f"      Keys: {list(llm.keys())}")

                # Check emotion values for layer 40 (middle layer)
                for emotion in ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']:
                    if emotion in llm:
                        vals = llm[emotion]
                        if isinstance(vals, list):
                            print(f"      {emotion}: {len(vals)} layers")
                            # Show values for a few layers
                            if len(vals) >= 62:
                                layer_samples = [0, 20, 40, 60]
                                sample_vals = [vals[l] for l in layer_samples]
                                print(f"        Layers {layer_samples}: {sample_vals}")

                                # Check for NaN
                                nan_count = sum(1 for v in vals if v != v)  # NaN != NaN
                                print(f"        NaN count: {nan_count}/{len(vals)}")
                            else:
                                print(f"        Values: {vals}")
            else:
                print(f"    logit_lens_mean present: NO")
                print(f"    Available keys: {list(lw.keys())}")
        else:
            print(f"  Layerwise data present: NO")

    # Now compare with what we expect from axes
    print(f"\n{'='*80}")
    print(f"COMPARING AXES VS LAYERWISE")
    print(f"{'='*80}")

    sent = conv['sentences'][0]
    if 'axes' in sent and 'layerwise' in sent:
        axes_data = sent['axes']
        layerwise_data = sent['layerwise'].get('logit_lens_mean', {})

        for emotion in ['happiness', 'sadness', 'anger']:
            if emotion in axes_data and emotion in layerwise_data:
                axes_vals = axes_data[emotion]
                lw_vals = layerwise_data[emotion]

                if isinstance(axes_vals, list) and isinstance(lw_vals, list):
                    print(f"\n{emotion.upper()}:")
                    print(f"  Axes: {len(axes_vals)} values")
                    print(f"    First 5: {axes_vals[:5]}")
                    print(f"    Stats: min={min(axes_vals):.4f}, max={max(axes_vals):.4f}, mean={np.mean(axes_vals):.4f}")

                    print(f"  Layerwise: {len(lw_vals)} values")
                    if len(lw_vals) > 0:
                        # Check for NaN
                        valid_vals = [v for v in lw_vals if v == v]  # filter out NaN
                        if valid_vals:
                            print(f"    First 5: {lw_vals[:5]}")
                            print(f"    Stats: min={min(valid_vals):.4f}, max={max(valid_vals):.4f}, mean={np.mean(valid_vals):.4f}")
                            print(f"    Valid values: {len(valid_vals)}/{len(lw_vals)}")
                        else:
                            print(f"    ALL NaN!")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
