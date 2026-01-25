#!/usr/bin/env python3
"""
Compare PC-dimension alignments between opposite-pair and contrast-all methods.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

DIMENSION_NAMES = {
    'V': 'Valence',
    'A': 'Arousal',
    'D': 'Dominance',
    'AA': 'Approach-Avoidance'
}

def load_analysis_results(path: Path) -> dict:
    """Load layer analysis results JSON."""
    with open(path, 'r') as f:
        return json.load(f)

def extract_pc_dimension_sims(results: list, layer: int, entity: str) -> dict:
    """Extract PC-dimension similarities for a specific layer and entity."""
    layer_result = next((r for r in results if r['layer'] == layer), None)
    if not layer_result:
        return None

    return layer_result[f'{entity}_pc_dim_sims']

def compare_pc_alignments_at_layer(opposite_results: list, contrast_results: list, layer: int = 30):
    """Compare PC-dimension alignments at a specific layer."""

    print(f"\n{'='*80}")
    print(f"PC-DIMENSION ALIGNMENTS AT LAYER {layer}")
    print(f"{'='*80}")

    for entity in ['M', 'U']:
        entity_name = "Assistant (M)" if entity == 'M' else "User (U)"

        opp_sims = extract_pc_dimension_sims(opposite_results, layer, entity)
        con_sims = extract_pc_dimension_sims(contrast_results, layer, entity)

        if not opp_sims or not con_sims:
            print(f"⚠ Warning: No data for {entity} at layer {layer}")
            continue

        print(f"\n{entity_name} - PC to Dimension Alignments (Absolute Cosine Similarity)")
        print(f"\n{'PC':<6} {'Dimension':<20} {'Opposite-Pair':>15} {'Contrast-All':>15} {'Difference':>15}")
        print("-" * 80)

        for pc_idx in range(1, 5):
            pc_name = f"PC{pc_idx}"

            # Get similarities for this PC
            opp_pc = opp_sims.get(pc_name, {})
            con_pc = con_sims.get(pc_name, {})

            # Find strongest alignment in each method
            opp_max_dim = max(opp_pc.items(), key=lambda x: abs(x[1]))[0] if opp_pc else None
            con_max_dim = max(con_pc.items(), key=lambda x: abs(x[1]))[0] if con_pc else None

            # Print all dimensions for this PC
            for dim_code in ['V', 'A', 'D', 'AA']:
                if dim_code in opp_pc and dim_code in con_pc:
                    opp_val = abs(opp_pc[dim_code])
                    con_val = abs(con_pc[dim_code])
                    diff = con_val - opp_val

                    # Mark strongest alignment
                    marker = " ★" if dim_code == opp_max_dim or dim_code == con_max_dim else ""

                    dim_name = DIMENSION_NAMES[dim_code]
                    print(f"{pc_name:<6} {dim_name:<20}{marker} {opp_val:>15.4f} {con_val:>15.4f} {diff:>+15.4f}")

            # Print separator between PCs
            if pc_idx < 4:
                print()

def plot_pc_dimension_heatmap_comparison(opposite_results: list, contrast_results: list, layer: int, output_dir: Path):
    """Create side-by-side heatmaps comparing PC-dimension alignments."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'PC-Dimension Alignments at Layer {layer}\nOpposite-Pair vs Contrast-All',
                 fontsize=16, fontweight='bold')

    for idx, entity in enumerate(['M', 'U']):
        entity_name = "Assistant (M)" if entity == 'M' else "User (U)"

        opp_sims = extract_pc_dimension_sims(opposite_results, layer, entity)
        con_sims = extract_pc_dimension_sims(contrast_results, layer, entity)

        if not opp_sims or not con_sims:
            continue

        # Build matrices (4 PCs × 4 dimensions)
        dim_codes = ['V', 'A', 'D', 'AA']
        opp_matrix = np.zeros((4, 4))
        con_matrix = np.zeros((4, 4))

        for pc_idx in range(1, 5):
            pc_name = f"PC{pc_idx}"
            for dim_idx, dim_code in enumerate(dim_codes):
                if pc_name in opp_sims and dim_code in opp_sims[pc_name]:
                    opp_matrix[pc_idx-1, dim_idx] = abs(opp_sims[pc_name][dim_code])
                if pc_name in con_sims and dim_code in con_sims[pc_name]:
                    con_matrix[pc_idx-1, dim_idx] = abs(con_sims[pc_name][dim_code])

        # Plot opposite-pair
        ax_opp = axes[idx, 0]
        im_opp = ax_opp.imshow(opp_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax_opp.set_xticks(range(4))
        ax_opp.set_xticklabels([DIMENSION_NAMES[d] for d in dim_codes], rotation=45, ha='right')
        ax_opp.set_yticks(range(4))
        ax_opp.set_yticklabels([f'PC{i+1}' for i in range(4)])
        ax_opp.set_title(f'{entity_name}: Opposite-Pair', fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax_opp.text(j, i, f'{opp_matrix[i, j]:.3f}',
                                  ha="center", va="center", color="black", fontsize=9)

        # Plot contrast-all
        ax_con = axes[idx, 1]
        im_con = ax_con.imshow(con_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax_con.set_xticks(range(4))
        ax_con.set_xticklabels([DIMENSION_NAMES[d] for d in dim_codes], rotation=45, ha='right')
        ax_con.set_yticks(range(4))
        ax_con.set_yticklabels([f'PC{i+1}' for i in range(4)])
        ax_con.set_title(f'{entity_name}: Contrast-All', fontsize=12, fontweight='bold')

        # Add text annotations
        for i in range(4):
            for j in range(4):
                text = ax_con.text(j, i, f'{con_matrix[i, j]:.3f}',
                                  ha="center", va="center", color="black", fontsize=9)

    # Add colorbar
    fig.colorbar(im_con, ax=axes, orientation='horizontal', pad=0.05, fraction=0.05,
                 label='Absolute Cosine Similarity')

    plt.tight_layout()
    output_file = output_dir / f'pc_dimension_heatmap_comparison_layer{layer}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_file}")

def main():
    output_dir = Path("probes/ua_emotion_disentangle/method_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPARING PC-DIMENSION ALIGNMENTS")
    print("="*80)

    # Load analysis results
    opposite_file = Path("probes/ua_emotion_disentangle/layer_analysis/layer_analysis_results.json")
    contrast_file = Path("probes/ua_emotion_disentangle/layer_analysis_contrast_all/layer_analysis_results.json")

    opposite_data = load_analysis_results(opposite_file)
    contrast_data = load_analysis_results(contrast_file)

    opposite_results = opposite_data['results']
    contrast_results = contrast_data['results']

    # Compare at layer 30
    compare_pc_alignments_at_layer(opposite_results, contrast_results, layer=30)

    # Create heatmap comparison
    print("\n" + "="*80)
    print("CREATING HEATMAP COMPARISON")
    print("="*80)
    plot_pc_dimension_heatmap_comparison(opposite_results, contrast_results, 30, output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
