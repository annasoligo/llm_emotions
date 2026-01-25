#!/usr/bin/env python3
"""Compare user vs assistant entity-specific dimensions from auto-interpretation.

This script analyzes whether discovered affective dimensions are:
1. Entity-specific (different for user vs assistant)
2. Shared (similar across both entities)
3. How dimensions relate to valence, arousal, dominance axes
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import re


def load_entity_interpretations(filepath: Path) -> List[Dict]:
    """Load interpretations from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data['interpretations']


def extract_key_terms(text: str) -> set:
    """Extract key emotional/affective terms from description."""
    # Common affective dimension keywords
    emotional_terms = {
        # Valence
        'positive', 'negative', 'enthusiastic', 'excited', 'happy', 'angry', 'frustrated',
        'cynical', 'confident', 'anxious', 'fear', 'disgust', 'joy', 'sadness',
        # Arousal
        'energetic', 'calm', 'activated', 'passive', 'intense', 'subdued',
        # Dominance/Agency
        'confident', 'vulnerable', 'assertive', 'helpless', 'empowered', 'powerless',
        'agency', 'control', 'dominance',
        # Cognitive
        'learning', 'understanding', 'discovery', 'confusion', 'clarity',
        # Interpersonal
        'empathetic', 'directive', 'supportive', 'dismissive',
    }

    text_lower = text.lower()
    found_terms = set()
    for term in emotional_terms:
        if term in text_lower:
            found_terms.add(term)

    return found_terms


def classify_vad_dimension(dim_name: str, pos_desc: str, neg_desc: str) -> Dict[str, bool]:
    """Classify whether dimension relates to Valence, Arousal, Dominance."""
    combined = (dim_name + ' ' + pos_desc + ' ' + neg_desc).lower()

    valence_keywords = ['positive', 'negative', 'enthusiasm', 'cynical', 'happy', 'angry',
                        'frustrated', 'excited', 'disgust', 'joy', 'optimistic', 'pessimistic']
    arousal_keywords = ['energy', 'calm', 'activated', 'intense', 'arousal', 'excitement',
                       'subdued', 'vigorous', 'lethargic']
    dominance_keywords = ['confident', 'vulnerable', 'assertive', 'helpless', 'agency',
                         'control', 'dominance', 'power', 'submission', 'empowered']

    return {
        'valence': any(kw in combined for kw in valence_keywords),
        'arousal': any(kw in combined for kw in arousal_keywords),
        'dominance': any(kw in combined for kw in dominance_keywords),
    }


def compute_dimension_similarity(user_dim: Dict, asst_dim: Dict) -> float:
    """Compute similarity between user and assistant dimensions (0-1)."""
    # Extract terms from both
    user_terms = extract_key_terms(
        user_dim['standard_interpretation']['dimension_name'] + ' ' +
        user_dim['standard_interpretation']['positive_description'] + ' ' +
        user_dim['standard_interpretation']['negative_description']
    )

    asst_terms = extract_key_terms(
        asst_dim['standard_interpretation']['dimension_name'] + ' ' +
        asst_dim['standard_interpretation']['positive_description'] + ' ' +
        asst_dim['standard_interpretation']['negative_description']
    )

    if not user_terms and not asst_terms:
        return 0.0

    # Jaccard similarity
    intersection = user_terms & asst_terms
    union = user_terms | asst_terms

    return len(intersection) / len(union) if union else 0.0


def main():
    user_file = Path('outputs/interpretations/autointerp/controlled_variation/user_isolation_layers_20-40.json')
    asst_file = Path('outputs/interpretations/autointerp/controlled_variation/assistant_isolation_layers_20-40.json')

    user_interps = load_entity_interpretations(user_file)
    asst_interps = load_entity_interpretations(asst_file)

    print("=" * 80)
    print("ENTITY-SPECIFIC DIMENSION COMPARISON")
    print("=" * 80)

    # Organize by layer and PC
    user_by_layer_pc = {}
    asst_by_layer_pc = {}

    for interp in user_interps:
        key = (interp['layer'], interp['pc_index'])
        user_by_layer_pc[key] = interp

    for interp in asst_interps:
        key = (interp['layer'], interp['pc_index'])
        asst_by_layer_pc[key] = interp

    # Compare each layer
    for layer in sorted(set(k[0] for k in user_by_layer_pc.keys())):
        print(f"\n{'='*80}")
        print(f"LAYER {layer}")
        print('='*80)

        # Compare corresponding PCs
        for pc in range(5):
            key = (layer, pc)
            if key not in user_by_layer_pc or key not in asst_by_layer_pc:
                continue

            user_dim = user_by_layer_pc[key]
            asst_dim = asst_by_layer_pc[key]

            similarity = compute_dimension_similarity(user_dim, asst_dim)

            user_std = user_dim['standard_interpretation']
            asst_std = asst_dim['standard_interpretation']

            # Classify VAD
            user_vad = classify_vad_dimension(
                user_std['dimension_name'],
                user_std['positive_description'],
                user_std['negative_description']
            )
            asst_vad = classify_vad_dimension(
                asst_std['dimension_name'],
                asst_std['positive_description'],
                asst_std['negative_description']
            )

            print(f"\n--- PC{pc} Comparison ---")
            print(f"Similarity: {similarity:.2f}")
            print(f"\nUSER: {user_std['dimension_name']}")
            print(f"  (+) {user_std['positive_description'][:100]}...")
            print(f"  (-) {user_std['negative_description'][:100]}...")
            print(f"  VAD: V={user_vad['valence']} A={user_vad['arousal']} D={user_vad['dominance']}")

            print(f"\nASSISTANT: {asst_std['dimension_name']}")
            print(f"  (+) {asst_std['positive_description'][:100]}...")
            print(f"  (-) {asst_std['negative_description'][:100]}...")
            print(f"  VAD: V={asst_vad['valence']} A={asst_vad['arousal']} D={asst_vad['dominance']}")

            # Classification
            if similarity > 0.5:
                print(f"  → SHARED dimension (similarity={similarity:.2f})")
            elif similarity > 0.25:
                print(f"  → PARTIALLY SHARED (similarity={similarity:.2f})")
            else:
                print(f"  → ENTITY-SPECIFIC (similarity={similarity:.2f})")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_similarities = []
    shared_count = 0
    partial_count = 0
    specific_count = 0

    for layer in sorted(set(k[0] for k in user_by_layer_pc.keys())):
        for pc in range(5):
            key = (layer, pc)
            if key in user_by_layer_pc and key in asst_by_layer_pc:
                sim = compute_dimension_similarity(user_by_layer_pc[key], asst_by_layer_pc[key])
                all_similarities.append(sim)

                if sim > 0.5:
                    shared_count += 1
                elif sim > 0.25:
                    partial_count += 1
                else:
                    specific_count += 1

    total = len(all_similarities)
    avg_sim = sum(all_similarities) / total if total > 0 else 0

    print(f"\nTotal dimension pairs analyzed: {total}")
    print(f"Average similarity: {avg_sim:.3f}")
    print(f"\nClassification:")
    print(f"  SHARED (>0.5 similarity):          {shared_count:2d} ({100*shared_count/total:.1f}%)")
    print(f"  PARTIALLY SHARED (0.25-0.5):       {partial_count:2d} ({100*partial_count/total:.1f}%)")
    print(f"  ENTITY-SPECIFIC (<0.25 similarity): {specific_count:2d} ({100*specific_count/total:.1f}%)")

    # VAD analysis
    print(f"\n{'='*80}")
    print("DIMENSION TYPE ANALYSIS")
    print('='*80)

    user_vad_counts = {'valence': 0, 'arousal': 0, 'dominance': 0}
    asst_vad_counts = {'valence': 0, 'arousal': 0, 'dominance': 0}

    for interp in user_interps:
        std = interp['standard_interpretation']
        vad = classify_vad_dimension(std['dimension_name'], std['positive_description'], std['negative_description'])
        for dim_type, present in vad.items():
            if present:
                user_vad_counts[dim_type] += 1

    for interp in asst_interps:
        std = interp['standard_interpretation']
        vad = classify_vad_dimension(std['dimension_name'], std['positive_description'], std['negative_description'])
        for dim_type, present in vad.items():
            if present:
                asst_vad_counts[dim_type] += 1

    print(f"\nUSER dimensions:")
    for dim_type, count in user_vad_counts.items():
        print(f"  {dim_type.capitalize():12s}: {count:2d} / 15 ({100*count/15:.1f}%)")

    print(f"\nASSISTANT dimensions:")
    for dim_type, count in asst_vad_counts.items():
        print(f"  {dim_type.capitalize():12s}: {count:2d} / 15 ({100*count/15:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
