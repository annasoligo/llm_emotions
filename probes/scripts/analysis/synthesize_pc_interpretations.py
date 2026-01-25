#!/usr/bin/env python3
"""Synthesize PCA/cPCA interpretations across layers and entity types.

Analyzes auto-interpretation results to identify:
1. Consistent dimensions across layers
2. Entity-specific vs shared dimensions
3. Affective axes (valence, arousal, dominance, etc.)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional
import re

import pandas as pd


def load_interpretations(interp_path: Path) -> Dict:
    """Load interpretation results from JSON file."""
    with open(interp_path, 'r') as f:
        return json.load(f)


def extract_dimension_info(interp: Dict) -> Dict:
    """Extract key information from a single PC interpretation."""
    return {
        'source': interp.get('source', 'unknown'),
        'layer': interp.get('layer', -1),
        'pc_index': interp.get('pc_index', -1),
        'dimension_name': interp.get('dimension_name', 'unknown'),
        'positive_description': interp.get('positive_description', ''),
        'negative_description': interp.get('negative_description', ''),
        'confidence': interp.get('confidence', 'unknown'),
        'reasoning': interp.get('reasoning', ''),
        'tracks_user_emotion': interp.get('tracks_user_emotion'),
        'tracks_assistant_emotion': interp.get('tracks_assistant_emotion'),
        'tracks_interaction': interp.get('tracks_interaction'),
        'error': interp.get('error'),
    }


def classify_dimension_type(name: str, pos_desc: str, neg_desc: str) -> List[str]:
    """Classify dimension by affective type (valence, arousal, etc.)."""
    name_lower = name.lower()
    pos_lower = pos_desc.lower()
    neg_lower = neg_desc.lower()

    combined = f"{name_lower} {pos_lower} {neg_lower}"

    types = []

    # Valence keywords
    valence_keywords = [
        'positive', 'negative', 'pleasant', 'unpleasant',
        'happy', 'sad', 'success', 'failure', 'triumph', 'defeat',
        'joy', 'despair', 'delight', 'disgust', 'satisfaction', 'disappointment'
    ]
    if any(kw in combined for kw in valence_keywords):
        types.append('valence')

    # Arousal keywords
    arousal_keywords = [
        'arousal', 'activation', 'intensity', 'energy', 'explosive',
        'calm', 'intense', 'high-intensity', 'low-intensity',
        'energetic', 'withdrawn', 'reactive', 'passive'
    ]
    if any(kw in combined for kw in arousal_keywords):
        types.append('arousal')

    # Dominance/control keywords
    dominance_keywords = [
        'control', 'dominance', 'power', 'helpless', 'agency',
        'mastery', 'submission', 'confident', 'vulnerable'
    ]
    if any(kw in combined for kw in dominance_keywords):
        types.append('dominance')

    # Internalization/externalization
    direction_keywords = [
        'internaliz', 'externaliz', 'inward', 'outward',
        'self-focused', 'other-focused', 'introspective', 'reactive'
    ]
    if any(kw in combined for kw in direction_keywords):
        types.append('directionality')

    # Cognitive/appraisal
    cognitive_keywords = [
        'appraisal', 'cognitive', 'expectation', 'surprise',
        'anticipated', 'unexpected', 'certainty', 'uncertainty'
    ]
    if any(kw in combined for kw in cognitive_keywords):
        types.append('cognitive')

    return types if types else ['unclassified']


def group_similar_dimensions(dimensions: List[Dict], similarity_threshold: float = 0.7) -> List[List[Dict]]:
    """Group dimensions with similar names/descriptions across layers."""
    # Simple keyword-based grouping
    groups = []

    for dim in dimensions:
        name = dim['dimension_name'].lower()

        # Find matching group
        matched = False
        for group in groups:
            group_name = group[0]['dimension_name'].lower()

            # Check for keyword overlap
            name_words = set(re.findall(r'\w+', name))
            group_words = set(re.findall(r'\w+', group_name))

            overlap = len(name_words & group_words) / max(len(name_words), len(group_words))

            if overlap >= similarity_threshold:
                group.append(dim)
                matched = True
                break

        if not matched:
            groups.append([dim])

    return groups


def analyze_entity_specificity(dimension_groups: List[List[Dict]]) -> pd.DataFrame:
    """Analyze which dimensions are entity-specific vs shared."""
    results = []

    for group in dimension_groups:
        if not group:
            continue

        # Check entity tracking across group
        tracks_user = [d.get('tracks_user_emotion') for d in group if d.get('tracks_user_emotion') is not None]
        tracks_asst = [d.get('tracks_assistant_emotion') for d in group if d.get('tracks_assistant_emotion') is not None]
        tracks_inter = [d.get('tracks_interaction') for d in group if d.get('tracks_interaction') is not None]

        # Determine entity specificity
        if tracks_user and sum(tracks_user) / len(tracks_user) > 0.7:
            entity_type = 'user-specific'
        elif tracks_asst and sum(tracks_asst) / len(tracks_asst) > 0.7:
            entity_type = 'assistant-specific'
        elif tracks_inter and sum(tracks_inter) / len(tracks_inter) > 0.7:
            entity_type = 'interaction'
        else:
            entity_type = 'unknown'

        # Representative dimension
        rep = group[0]

        results.append({
            'dimension_name': rep['dimension_name'],
            'entity_type': entity_type,
            'n_layers': len(group),
            'layers': sorted([d['layer'] for d in group]),
            'confidence': rep['confidence'],
            'dimension_types': ', '.join(classify_dimension_type(
                rep['dimension_name'],
                rep['positive_description'],
                rep['negative_description']
            )),
        })

    return pd.DataFrame(results)


def synthesize_interpretations(interp_paths: List[Path], output_dir: Path):
    """Synthesize interpretations from multiple files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SYNTHESIZING PC INTERPRETATIONS")
    print("=" * 80)

    all_dims = []

    # Load all interpretations
    for path in interp_paths:
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue

        print(f"\nLoading: {path.name}")
        data = load_interpretations(path)

        interps = data.get('interpretations', [])
        valid_interps = [i for i in interps if 'error' not in i]
        error_interps = [i for i in interps if 'error' in i]

        print(f"  Valid: {len(valid_interps)}, Errors: {len(error_interps)}")

        for interp in valid_interps:
            dim_info = extract_dimension_info(interp)
            dim_info['source_file'] = path.name
            all_dims.append(dim_info)

    print(f"\nTotal dimensions loaded: {len(all_dims)}")

    if not all_dims:
        print("No valid interpretations found!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_dims)

    # Save raw data
    raw_csv = output_dir / 'all_interpretations_raw.csv'
    df.to_csv(raw_csv, index=False)
    print(f"\nSaved raw data: {raw_csv}")

    # Analyze by layer
    print("\n" + "=" * 80)
    print("DIMENSIONS BY LAYER")
    print("=" * 80)

    for layer in sorted(df['layer'].unique()):
        layer_dims = df[df['layer'] == layer]
        print(f"\nLayer {layer}: {len(layer_dims)} dimensions")
        for _, row in layer_dims.iterrows():
            print(f"  PC{row['pc_index']}: {row['dimension_name']} ({row['confidence']})")

    # Classify dimension types
    df['dimension_types'] = df.apply(
        lambda row: ', '.join(classify_dimension_type(
            row['dimension_name'],
            row['positive_description'],
            row['negative_description']
        )),
        axis=1
    )

    # Save classified
    classified_csv = output_dir / 'dimensions_classified.csv'
    df[['layer', 'pc_index', 'dimension_name', 'dimension_types', 'confidence', 'source_file']].to_csv(
        classified_csv, index=False
    )
    print(f"\nSaved classified dimensions: {classified_csv}")

    # Analyze dimension type distribution
    print("\n" + "=" * 80)
    print("DIMENSION TYPE DISTRIBUTION")
    print("=" * 80)

    type_counts = defaultdict(int)
    for types_str in df['dimension_types']:
        for t in types_str.split(', '):
            type_counts[t] += 1

    print("\nAffective dimension types found:")
    for dim_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(df)
        print(f"  {dim_type:20s}: {count:3d} ({pct:5.1f}%)")

    # Group similar dimensions across layers
    print("\n" + "=" * 80)
    print("DIMENSION CONSISTENCY ACROSS LAYERS")
    print("=" * 80)

    dimension_groups = group_similar_dimensions(all_dims)
    consistent_dims = [g for g in dimension_groups if len(g) >= 3]

    print(f"\nFound {len(consistent_dims)} dimensions appearing in 3+ layers:")
    for group in sorted(consistent_dims, key=lambda g: -len(g)):
        rep = group[0]
        layers = sorted([d['layer'] for d in group])
        print(f"\n  {rep['dimension_name']}")
        print(f"    Layers: {layers}")
        print(f"    Types: {', '.join(classify_dimension_type(rep['dimension_name'], rep['positive_description'], rep['negative_description']))}")
        print(f"    Positive: {rep['positive_description'][:100]}...")
        print(f"    Negative: {rep['negative_description'][:100]}...")

    # Analyze entity specificity if data available
    if df['tracks_user_emotion'].notna().any():
        print("\n" + "=" * 80)
        print("ENTITY SPECIFICITY ANALYSIS")
        print("=" * 80)

        entity_df = analyze_entity_specificity(dimension_groups)
        entity_csv = output_dir / 'entity_specificity.csv'
        entity_df.to_csv(entity_csv, index=False)
        print(f"\nSaved entity analysis: {entity_csv}")

        print("\nEntity-specific dimensions:")
        for _, row in entity_df.iterrows():
            if row['entity_type'] != 'unknown':
                print(f"  {row['dimension_name']:40s} -> {row['entity_type']:20s} (layers: {row['n_layers']})")

    # Generate summary report
    summary_path = output_dir / 'synthesis_summary.md'
    with open(summary_path, 'w') as f:
        f.write("# PC Interpretation Synthesis\n\n")
        f.write(f"**Total Dimensions Analyzed**: {len(all_dims)}\n\n")
        f.write(f"**Layers Covered**: {sorted(df['layer'].unique())}\n\n")
        f.write(f"**Source Files**: {len(interp_paths)}\n\n")

        f.write("## Dimension Type Distribution\n\n")
        for dim_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(df)
            f.write(f"- **{dim_type}**: {count} ({pct:.1f}%)\n")

        f.write("\n## Consistent Cross-Layer Dimensions\n\n")
        f.write(f"Found {len(consistent_dims)} dimensions appearing in 3+ layers\n\n")
        for group in sorted(consistent_dims, key=lambda g: -len(g)):
            rep = group[0]
            layers = sorted([d['layer'] for d in group])
            f.write(f"### {rep['dimension_name']}\n\n")
            f.write(f"- **Layers**: {layers}\n")
            f.write(f"- **Types**: {', '.join(classify_dimension_type(rep['dimension_name'], rep['positive_description'], rep['negative_description']))}\n")
            f.write(f"- **Positive**: {rep['positive_description']}\n")
            f.write(f"- **Negative**: {rep['negative_description']}\n")
            f.write(f"- **Confidence**: {rep['confidence']}\n\n")

        if df['tracks_user_emotion'].notna().any() and not entity_df.empty:
            f.write("\n## Entity Specificity\n\n")
            for _, row in entity_df.iterrows():
                if row['entity_type'] != 'unknown':
                    f.write(f"### {row['dimension_name']}\n\n")
                    f.write(f"- **Entity Type**: {row['entity_type']}\n")
                    f.write(f"- **Layers**: {row['layers']}\n")
                    f.write(f"- **Dimension Types**: {row['dimension_types']}\n\n")

    print(f"\nSaved summary: {summary_path}")

    print("\n" + "=" * 80)
    print("SYNTHESIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - {raw_csv.name}")
    print(f"  - {classified_csv.name}")
    if df['tracks_user_emotion'].notna().any():
        print(f"  - {entity_csv.name}")
    print(f"  - {summary_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize PCA/cPCA auto-interpretation results"
    )
    parser.add_argument(
        '--interp-dir',
        type=Path,
        default=Path('outputs/interpretations/autointerp'),
        help='Directory containing interpretation JSON files'
    )
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        help='Specific interpretation files to analyze (relative to interp-dir)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('outputs/analysis/pc_synthesis'),
        help='Output directory for synthesis results'
    )

    args = parser.parse_args()

    # Find interpretation files
    if args.files:
        interp_paths = [args.interp_dir / f for f in args.files]
    else:
        # Find all JSON files recursively
        interp_paths = list(args.interp_dir.rglob('*.json'))
        interp_paths = [p for p in interp_paths if 'checkpoint' not in p.name]

    if not interp_paths:
        print(f"ERROR: No interpretation files found in {args.interp_dir}")
        return

    print(f"Found {len(interp_paths)} interpretation files")
    for p in interp_paths:
        print(f"  - {p.relative_to(args.interp_dir)}")

    synthesize_interpretations(interp_paths, args.output)


if __name__ == '__main__':
    main()
