#!/usr/bin/env python3
"""
Analyze PC meanings (from autointerp) and their weights in emotion probes.

Creates a comprehensive report showing:
1. What each PC represents (from autointerp)
2. How much each PC contributes to each emotion probe
3. Which PCs are most important for which emotions
"""

import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def load_autointerp(autointerp_path: Path) -> dict:
    """Load autointerp results."""
    with open(autointerp_path, 'r') as f:
        return json.load(f)


def load_probe(probe_path: Path) -> dict:
    """Load trained probe."""
    with open(probe_path, 'rb') as f:
        return pickle.load(f)


def analyze_pc_weights(probe_results: dict, n_components: int) -> pd.DataFrame:
    """
    Analyze how much each PC contributes to each emotion.

    Returns DataFrame with columns: PC, emotion1_weight, emotion2_weight, ...
    """
    # Get probe weights: [n_emotions, n_components]
    weights = probe_results['model'].weight.detach().cpu().numpy()
    emotion_names = probe_results['label_names']

    # Create DataFrame
    data = {'PC': list(range(n_components))}

    for i, emotion in enumerate(emotion_names):
        # SIGNED weight for this emotion on each PC (positive or negative)
        data[f'{emotion}_weight'] = weights[i, :]

    return pd.DataFrame(data)


def create_comprehensive_report(layer: int, n_components_list: list,
                                autointerp_path: Path, output_path: Path):
    """
    Create comprehensive report for a given layer across multiple n_components.
    """

    # Load autointerp (only need to do this once - covers all PCs)
    print(f"Loading autointerp from {autointerp_path}")
    autointerp = load_autointerp(autointerp_path)

    # Extract interpretations list from new format
    if 'interpretations' in autointerp:
        interpretations = autointerp['interpretations']
    else:
        interpretations = autointerp  # Fallback for old format

    # Filter to target layer
    layer_interpretations = [item for item in interpretations if item['layer'] == layer]

    if not layer_interpretations:
        raise ValueError(f"No autointerp data found for layer {layer}")

    # Create PC meanings lookup
    pc_meanings = {}
    for pc_data in layer_interpretations:
        pc_idx = pc_data['pc_index']
        pc_meanings[pc_idx] = {
            'name': pc_data['dimension_name'],
            'confidence': pc_data['confidence'],
            'description': pc_data.get('reasoning', 'N/A')
        }

    # Process each n_components configuration
    all_sections = []

    for n_pcs in sorted(n_components_list):
        print(f"\nProcessing {n_pcs} PCs...")

        # Load probe
        if n_pcs == 50:
            probe_dir = Path('/workspace-vast/annas/git/research-tools/results/emotion_probes_high_alpha_cpca')
        else:
            probe_dir = Path(f'/workspace-vast/annas/git/research-tools/results/emotion_probes_top{n_pcs}')

        probe_path = probe_dir / f'probe_layer{layer}_all_cpca_top{n_pcs}.pkl' if n_pcs < 50 else probe_dir / f'probe_layer{layer}_all_cpca.pkl'

        if not probe_path.exists():
            print(f"  Warning: {probe_path} not found, skipping")
            continue

        probe_results = load_probe(probe_path)

        # Get weights
        weights_df = analyze_pc_weights(probe_results, n_pcs)
        emotion_names = probe_results['label_names']

        # Create section for this configuration
        section = []
        section.append(f"\n{'='*80}")
        section.append(f"LAYER {layer} - TOP {n_pcs} PCs")
        section.append(f"{'='*80}\n")

        # Overall statistics
        test_acc = probe_results['test_accuracy']
        section.append(f"Test Accuracy: {test_acc:.4f}\n")

        # For each PC
        for pc_idx in range(n_pcs):
            # PC meaning
            if pc_idx in pc_meanings:
                meaning = pc_meanings[pc_idx]
                section.append(f"\n--- PC {pc_idx}: {meaning['name']} (confidence: {meaning['confidence']}) ---")
                section.append(f"Description: {meaning['description'][:200]}...")
            else:
                section.append(f"\n--- PC {pc_idx}: [No autointerp available] ---")

            # Weights for each emotion
            section.append("\nWeights by emotion (signed - positive/negative contribution):")
            pc_weights = []
            for emotion in emotion_names:
                weight = weights_df[weights_df['PC'] == pc_idx][f'{emotion}_weight'].values[0]
                pc_weights.append((emotion, weight))

            # Sort by absolute weight (descending) but keep sign
            pc_weights.sort(key=lambda x: abs(x[1]), reverse=True)

            # Print sorted weights with sign
            for emotion, weight in pc_weights:
                # Visual bar that accounts for sign
                if weight > 0:
                    bar = '█' * int(abs(weight) * 50)
                    sign = '+'
                else:
                    bar = '▓' * int(abs(weight) * 50)  # Different pattern for negative
                    sign = '-'
                section.append(f"  {emotion:12s}: {sign}{abs(weight):.4f}  {bar}")

        # Summary: Which PCs matter most for each emotion
        section.append(f"\n\n{'='*80}")
        section.append(f"SUMMARY: Most Important PCs for Each Emotion (Layer {layer}, {n_pcs} PCs)")
        section.append(f"{'='*80}\n")

        for emotion in emotion_names:
            # Get top 3 PCs for this emotion (by absolute value)
            emotion_weights = [(pc, weights_df[weights_df['PC'] == pc][f'{emotion}_weight'].values[0])
                              for pc in range(n_pcs)]
            emotion_weights.sort(key=lambda x: abs(x[1]), reverse=True)
            top_pcs = emotion_weights[:3]

            section.append(f"\n{emotion.upper()}:")
            for rank, (pc, weight) in enumerate(top_pcs, 1):
                pc_name = pc_meanings.get(pc, {}).get('name', 'Unknown')
                sign = '+' if weight >= 0 else ''
                section.append(f"  {rank}. PC {pc} ({pc_name}): weight={sign}{weight:.4f}")

        all_sections.append('\n'.join(section))

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n\n'.join(all_sections))

    print(f"\n✓ Saved report to {output_path}")


def main():
    """Generate PC meaning and weight reports."""

    # Layer 30 (as specified)
    layer = 30

    # Only top 3, 5, 10 as requested
    n_components_list = [3, 5, 10]

    # Paths (absolute)
    autointerp_path = Path('/workspace-vast/annas/git/research-tools/probes/results/autointerp/gemma_layer30_top20.json')
    output_path = Path('/workspace-vast/annas/git/research-tools/results/pc_meanings_and_weights_layer30.txt')

    print("="*80)
    print("ANALYZING PC MEANINGS AND WEIGHTS")
    print("="*80)
    print(f"\nLayer: {layer}")
    print(f"Configurations: {n_components_list} PCs")
    print(f"Autointerp: {autointerp_path}")
    print(f"Output: {output_path}")

    if not autointerp_path.exists():
        print(f"\n❌ Error: Autointerp file not found: {autointerp_path}")
        return

    create_comprehensive_report(layer, n_components_list, autointerp_path, output_path)

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
