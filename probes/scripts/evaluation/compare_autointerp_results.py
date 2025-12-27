#!/usr/bin/env python3
"""Compare auto-interpretation results between ratio PCA and cPCA.

Usage:
    python scripts/compare_autointerp_results.py \
        --ratio-pca results/autointerp_ratio_pca_k10/layer30_top20.json \
        --cpca probes/results/autointerp/gemma_layer30_top20.json \
        --output results/autointerp_comparison.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_autointerp_results(json_path: Path) -> Dict:
    """Load autointerp results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def format_interpretation(interp: Dict, pc_idx: int, method_name: str) -> str:
    """Format a single PC interpretation."""
    lines = [
        f"{'='*80}",
        f"{method_name} - PC {pc_idx}",
        f"{'='*80}",
        "",
        "INTERPRETATION:",
        interp.get('interpretation', 'N/A'),
        "",
    ]

    # Add positive examples if available
    if 'positive_examples' in interp:
        lines.append("POSITIVE EXAMPLES (high activation):")
        for i, ex in enumerate(interp['positive_examples'][:3], 1):
            lines.append(f"  {i}. {ex.get('text', ex.get('emotional', 'N/A'))}")
        lines.append("")

    # Add negative examples if available
    if 'negative_examples' in interp:
        lines.append("NEGATIVE EXAMPLES (low activation):")
        for i, ex in enumerate(interp['negative_examples'][:3], 1):
            lines.append(f"  {i}. {ex.get('text', ex.get('emotional', 'N/A'))}")
        lines.append("")

    return "\n".join(lines)


def compare_interpretations(
    ratio_results: Dict,
    cpca_results: Dict,
    output_path: Path,
    layer: int = 30,
    max_pcs: int = 20,
):
    """Compare interpretations side by side."""

    # Get interpretations for the specified layer
    ratio_layer = None
    cpca_layer = None

    for layer_data in ratio_results.get('layers', []):
        if layer_data['layer'] == layer:
            ratio_layer = layer_data
            break

    for layer_data in cpca_results.get('layers', []):
        if layer_data['layer'] == layer:
            cpca_layer = layer_data
            break

    if not ratio_layer or not cpca_layer:
        print(f"ERROR: Layer {layer} not found in one or both results")
        return

    ratio_pcs = {pc['pc']: pc for pc in ratio_layer.get('pcs', [])}
    cpca_pcs = {pc['pc']: pc for pc in cpca_layer.get('pcs', [])}

    # Generate comparison
    lines = [
        "="*80,
        f"AUTO-INTERPRETATION COMPARISON: Layer {layer}",
        "="*80,
        "",
        f"Ratio PCA (k=10): {len(ratio_pcs)} PCs interpreted",
        f"cPCA: {len(cpca_pcs)} PCs interpreted",
        "",
    ]

    # Compare each PC
    for pc_idx in range(min(max_pcs, len(ratio_pcs), len(cpca_pcs))):
        if pc_idx in ratio_pcs and pc_idx in cpca_pcs:
            lines.append("")
            lines.append("")
            lines.append(f"{'#'*80}")
            lines.append(f"PC {pc_idx} COMPARISON")
            lines.append(f"{'#'*80}")
            lines.append("")

            # Ratio PCA interpretation
            lines.append(format_interpretation(ratio_pcs[pc_idx], pc_idx, "RATIO PCA"))

            # cPCA interpretation
            lines.append(format_interpretation(cpca_pcs[pc_idx], pc_idx, "cPCA"))

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Comparison saved to: {output_path}")
    print(f"Compared {min(max_pcs, len(ratio_pcs), len(cpca_pcs))} PCs")


def main():
    parser = argparse.ArgumentParser(description="Compare autointerp results")
    parser.add_argument(
        "--ratio-pca",
        type=Path,
        required=True,
        help="Ratio PCA autointerp JSON"
    )
    parser.add_argument(
        "--cpca",
        type=Path,
        required=True,
        help="cPCA autointerp JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output comparison text file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=30,
        help="Layer to compare (default: 30)"
    )
    parser.add_argument(
        "--max-pcs",
        type=int,
        default=20,
        help="Maximum PCs to compare (default: 20)"
    )

    args = parser.parse_args()

    print("Loading autointerp results...")
    ratio_results = load_autointerp_results(args.ratio_pca)
    cpca_results = load_autointerp_results(args.cpca)

    print(f"Comparing layer {args.layer}...")
    compare_interpretations(
        ratio_results,
        cpca_results,
        args.output,
        layer=args.layer,
        max_pcs=args.max_pcs,
    )


if __name__ == "__main__":
    main()
