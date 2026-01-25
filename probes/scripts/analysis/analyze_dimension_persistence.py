#!/usr/bin/env python3
"""Analyze dimension persistence across layers.

Identifies which dimensions (e.g., valence, arousal) persist across layers
and whether they represent the same underlying construct or evolve.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


def load_interpretations_csv(csv_path: Path) -> pd.DataFrame:
    """Load classified interpretations from CSV."""
    df = pd.read_csv(csv_path)
    return df


def load_raw_interpretations(raw_csv_path: Path) -> pd.DataFrame:
    """Load raw interpretations with full descriptions."""
    return pd.read_csv(raw_csv_path)


def compute_dimension_similarity(dim1: dict, dim2: dict) -> float:
    """Compute semantic similarity between two dimension interpretations."""
    # Simple keyword-based similarity
    name1 = set(dim1['dimension_name'].lower().split())
    name2 = set(dim2['dimension_name'].lower().split())

    pos1 = set(dim1['positive_description'].lower().split())
    pos2 = set(dim2['positive_description'].lower().split())

    neg1 = set(dim1['negative_description'].lower().split())
    neg2 = set(dim2['negative_description'].lower().split())

    # Compute overlap
    name_overlap = len(name1 & name2) / max(len(name1 | name2), 1)
    pos_overlap = len(pos1 & pos2) / max(len(pos1 | pos2), 1)
    neg_overlap = len(neg1 & neg2) / max(len(neg1 | neg2), 1)

    # Weighted average
    return 0.4 * name_overlap + 0.3 * pos_overlap + 0.3 * neg_overlap


def track_dimension_across_layers(
    df: pd.DataFrame,
    dimension_type: str,
    min_similarity: float = 0.3,
) -> List[List[Dict]]:
    """Track a specific dimension type across layers."""
    # Filter by dimension type
    relevant_dims = df[df['dimension_types'].str.contains(dimension_type, na=False)]

    if relevant_dims.empty:
        return []

    # Group by layer
    layers = sorted(relevant_dims['layer'].unique())

    # Track dimension chains across layers
    chains = []

    for start_layer in layers:
        layer_dims = relevant_dims[relevant_dims['layer'] == start_layer]

        for _, start_dim in layer_dims.iterrows():
            # Start a new chain
            chain = [start_dim.to_dict()]
            current_layer = start_layer

            # Try to extend chain to subsequent layers
            for next_layer in [l for l in layers if l > current_layer]:
                next_layer_dims = relevant_dims[relevant_dims['layer'] == next_layer]

                if next_layer_dims.empty:
                    break

                # Find best match in next layer
                best_match = None
                best_similarity = 0

                for _, next_dim in next_layer_dims.iterrows():
                    sim = compute_dimension_similarity(
                        chain[-1],
                        next_dim.to_dict()
                    )

                    if sim > best_similarity:
                        best_similarity = sim
                        best_match = next_dim.to_dict()

                # Add to chain if similar enough
                if best_match and best_similarity >= min_similarity:
                    chain.append(best_match)
                    current_layer = next_layer
                else:
                    break

            # Only keep chains spanning multiple layers
            if len(chain) >= 2:
                chains.append(chain)

    return chains


def analyze_dimension_stability(chain: List[Dict]) -> Dict:
    """Analyze how stable a dimension is across layers."""
    layers = [d['layer'] for d in chain]
    names = [d['dimension_name'] for d in chain]

    # Compute pairwise similarities
    similarities = []
    for i in range(len(chain) - 1):
        sim = compute_dimension_similarity(chain[i], chain[i + 1])
        similarities.append(sim)

    return {
        'start_layer': min(layers),
        'end_layer': max(layers),
        'span': max(layers) - min(layers),
        'n_layers': len(layers),
        'layers': layers,
        'dimension_names': names,
        'avg_similarity': np.mean(similarities) if similarities else 0,
        'min_similarity': np.min(similarities) if similarities else 0,
        'stable': np.mean(similarities) > 0.5 if similarities else False,
    }


def analyze_persistence(csv_path: Path, raw_csv_path: Path, output_dir: Path):
    """Analyze dimension persistence across layers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DIMENSION PERSISTENCE ANALYSIS")
    print("=" * 80)

    # Load both datasets and merge
    df_raw = load_raw_interpretations(raw_csv_path)
    df_classified = load_interpretations_csv(csv_path)

    # Merge on layer and pc_index
    df = df_raw.merge(
        df_classified[['layer', 'pc_index', 'dimension_types']],
        on=['layer', 'pc_index'],
        how='left'
    )

    print(f"\nLoaded {len(df)} interpretations across {len(df['layer'].unique())} layers")
    print(f"Layers: {sorted(df['layer'].unique())}")

    # Identify dimension types
    all_types = set()
    for types_str in df['dimension_types'].dropna():
        for t in types_str.split(', '):
            all_types.add(t)

    print(f"\nDimension types found: {sorted(all_types)}")

    # Analyze each dimension type
    results = []

    for dim_type in sorted(all_types):
        if dim_type == 'unclassified':
            continue

        print(f"\n{'=' * 80}")
        print(f"TRACKING: {dim_type.upper()}")
        print("=" * 80)

        chains = track_dimension_across_layers(df, dim_type, min_similarity=0.3)

        print(f"\nFound {len(chains)} dimension chains")

        # Analyze each chain
        for i, chain in enumerate(chains, 1):
            stability = analyze_dimension_stability(chain)

            if stability['n_layers'] >= 2:
                print(f"\nChain {i}:")
                print(f"  Layers: {stability['layers']}")
                print(f"  Span: {stability['span']} layers")
                print(f"  Avg similarity: {stability['avg_similarity']:.3f}")
                print(f"  Stable: {'✓' if stability['stable'] else '✗'}")
                print(f"  Names: {stability['dimension_names'][0]} → {stability['dimension_names'][-1]}")

                # Add to results
                result = {
                    'dimension_type': dim_type,
                    'chain_id': i,
                    **stability,
                    'representative_name': chain[0]['dimension_name'],
                    'representative_positive': chain[0]['positive_description'],
                    'representative_negative': chain[0]['negative_description'],
                }
                results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / 'dimension_persistence.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved persistence analysis: {results_csv}")

    # Generate summary report
    print("\n" + "=" * 80)
    print("PERSISTENCE SUMMARY")
    print("=" * 80)

    stable_chains = results_df[results_df['stable']]
    print(f"\nStable dimensions (avg similarity > 0.5): {len(stable_chains)}/{len(results_df)}")

    # Group by dimension type
    print("\nBy dimension type:")
    for dim_type in sorted(results_df['dimension_type'].unique()):
        type_chains = results_df[results_df['dimension_type'] == dim_type]
        stable_count = len(type_chains[type_chains['stable']])
        avg_span = type_chains['span'].mean()

        print(f"  {dim_type:20s}: {stable_count}/{len(type_chains)} stable, avg span: {avg_span:.1f} layers")

    # Find longest-spanning dimensions
    print("\nLongest-spanning dimensions:")
    longest = results_df.nlargest(5, 'span')
    for _, row in longest.iterrows():
        print(f"  {row['representative_name']:40s}")
        print(f"    Type: {row['dimension_type']}, Span: {row['span']} layers ({row['start_layer']}-{row['end_layer']})")
        print(f"    Stability: {row['avg_similarity']:.3f}")

    # Save markdown report
    report_path = output_dir / 'persistence_report.md'
    with open(report_path, 'w') as f:
        f.write("# Dimension Persistence Analysis\n\n")
        f.write(f"**Total Dimension Chains**: {len(results_df)}\n\n")
        f.write(f"**Stable Chains** (avg similarity > 0.5): {len(stable_chains)}\n\n")

        f.write("## Persistence by Dimension Type\n\n")
        for dim_type in sorted(results_df['dimension_type'].unique()):
            type_chains = results_df[results_df['dimension_type'] == dim_type]
            stable_count = len(type_chains[type_chains['stable']])
            avg_span = type_chains['span'].mean()

            f.write(f"### {dim_type.title()}\n\n")
            f.write(f"- **Total chains**: {len(type_chains)}\n")
            f.write(f"- **Stable chains**: {stable_count}\n")
            f.write(f"- **Average span**: {avg_span:.1f} layers\n\n")

        f.write("## Longest-Spanning Dimensions\n\n")
        for _, row in longest.iterrows():
            f.write(f"### {row['representative_name']}\n\n")
            f.write(f"- **Type**: {row['dimension_type']}\n")
            f.write(f"- **Layers**: {row['start_layer']} → {row['end_layer']} (span: {row['span']})\n")
            f.write(f"- **Stability**: {row['avg_similarity']:.3f}\n")
            f.write(f"- **Positive**: {row['representative_positive'][:200]}...\n")
            f.write(f"- **Negative**: {row['representative_negative'][:200]}...\n\n")

    print(f"\nSaved report: {report_path}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dimension persistence across layers"
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=Path('outputs/analysis/pc_synthesis_jan2026/dimensions_classified.csv'),
        help='Classified dimensions CSV file'
    )
    parser.add_argument(
        '--raw-csv',
        type=Path,
        default=Path('outputs/analysis/pc_synthesis_jan2026/all_interpretations_raw.csv'),
        help='Raw interpretations CSV with full descriptions'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/analysis/dimension_persistence'),
        help='Output directory for persistence analysis'
    )

    args = parser.parse_args()

    if not args.raw_csv.exists():
        print(f"ERROR: Raw CSV file not found: {args.raw_csv}")
        return

    analyze_persistence(args.csv, args.raw_csv, args.output)


if __name__ == '__main__':
    main()
