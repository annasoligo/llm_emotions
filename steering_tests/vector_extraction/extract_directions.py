#!/usr/bin/env python3
"""
Extract steering directions from collected activations.

Supports four methods:
- emotion_vs_others: mean(emotion) - mean(all_other_emotions)
- emotion_vs_neutral: mean(emotional) - mean(neutral) [text_pairs only]
- emotion_vs_opposite: mean(emotion) - mean(opposite_emotion) [text_pairs only]
- emotion_vs_opposite_unique: (emotion - opposite) - mean(all bipolar vectors) [text_pairs only]
  Isolates what's unique about each bipolar axis vs the average bipolar direction.

Usage:
    # For text pairs - emotion vs neutral
    python extract_directions.py \
        --activations steering_tests/activations/test_gemma3_27b_text_pairs \
        --output steering_tests/vectors/gemma3_27b/text_pairs_emotion_vs_neutral \
        --method emotion_vs_neutral

    # For base/high - emotion vs others
    python extract_directions.py \
        --activations steering_tests/activations/test_gemma3_27b \
        --output steering_tests/vectors/gemma3_27b/base_emotion_vs_others \
        --method emotion_vs_others \
        --representation last_token
"""

import argparse
import json
import pickle
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm


# Psychological opposites mapping (from available 24 emotions)
EMOTION_OPPOSITES = {
    # Negative valence → Positive
    "fear": "calm",
    "anxiety": "relief",
    "anger": "calm",
    "frustration": "contentment",
    "sadness": "joy",
    "guilt": "pride",
    "shame": "pride",
    "disgust": "admiration",
    "contempt": "admiration",
    "boredom": "excitement",
    "despair": "hope",
    "confusion": "calm",

    # Positive valence → Negative (symmetric)
    "calm": "anxiety",
    "relief": "anxiety",
    "contentment": "frustration",
    "joy": "sadness",
    "pride": "shame",
    "admiration": "contempt",
    "excitement": "boredom",
    "hope": "despair",

    # Neutral/ambiguous
    "surprise": "calm",
    "curiosity": "boredom",
    "interest": "boredom",
    "gratitude": "contempt",
}


def parse_emotion_from_key(key: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract emotion and condition from activation key.

    Returns:
        (emotion, condition) where condition is 'neutral'/'emotional' for text_pairs,
        or None for chat mode datasets.
    """
    if mode == "text":
        # Text pairs: set_0_third_person_fear_neutral or set_0_third_person_fear_emotional
        match = re.match(r'set_\d+_\w+_(\w+)_(neutral|emotional)$', key)
        if match:
            return match.group(1), match.group(2)
    else:
        # Chat mode: career_decision_fear_0
        # Pattern: {topic}_{emotion}_{idx}
        # Emotions are single words, idx is a number
        match = re.match(r'.+_(\w+)_\d+$', key)
        if match:
            emotion = match.group(1)
            # Validate it's a known emotion (not part of topic)
            if emotion in EMOTION_OPPOSITES or emotion in EMOTION_OPPOSITES.values():
                return emotion, None

    return None, None


def load_layer_activations(layer_path: Path, mode: str, representation: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Load activations from a layer pickle file.

    Args:
        layer_path: Path to layer_XX.pkl
        mode: 'chat' or 'text'
        representation: For chat mode, 'last_token' or 'special_mean'

    Returns:
        Dict mapping key -> activation vector
    """
    with open(layer_path, 'rb') as f:
        data = pickle.load(f)

    if mode == "text":
        # Text mode: {key: ndarray}
        return data
    else:
        # Chat mode: {key: {'last_token': ndarray, 'special_mean': ndarray}}
        if representation is None:
            raise ValueError("representation required for chat mode")
        return {k: v[representation] for k, v in data.items()}


def group_by_emotion(
    activations: Dict[str, np.ndarray],
    mode: str,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """
    Group activations by emotion.

    Returns:
        For text mode: (emotional_by_emotion, neutral_by_emotion)
        For chat mode: (all_by_emotion, {})
    """
    emotional_by_emotion = defaultdict(list)
    neutral_by_emotion = defaultdict(list)

    for key, activation in activations.items():
        emotion, condition = parse_emotion_from_key(key, mode)
        if emotion is None:
            continue

        if mode == "text":
            if condition == "emotional":
                emotional_by_emotion[emotion].append(activation)
            else:
                neutral_by_emotion[emotion].append(activation)
        else:
            emotional_by_emotion[emotion].append(activation)

    return dict(emotional_by_emotion), dict(neutral_by_emotion)


def compute_emotion_vs_others(
    emotional_by_emotion: Dict[str, List[np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Compute emotion - all_others direction for each emotion.

    Returns:
        Dict mapping emotion -> (unit_vector, norm)
    """
    # Compute mean for each emotion
    emotion_means = {}
    for emotion, acts in emotional_by_emotion.items():
        emotion_means[emotion] = np.mean(acts, axis=0)

    # Compute global mean of all emotions (weighted equally per emotion, not per sample)
    all_emotion_means = list(emotion_means.values())

    results = {}
    for emotion, mean_act in emotion_means.items():
        # Mean of all OTHER emotions
        other_means = [m for e, m in emotion_means.items() if e != emotion]
        mean_others = np.mean(other_means, axis=0)

        # Direction: this emotion - others
        direction = mean_act - mean_others
        norm = float(np.linalg.norm(direction))
        unit_vector = direction / norm if norm > 0 else direction

        results[emotion] = (unit_vector, norm)

    return results


def compute_emotion_vs_neutral(
    emotional_by_emotion: Dict[str, List[np.ndarray]],
    neutral_by_emotion: Dict[str, List[np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Compute emotion - neutral direction using paired differences.

    Returns:
        Dict mapping emotion -> (unit_vector, norm)
    """
    results = {}

    for emotion in emotional_by_emotion:
        if emotion not in neutral_by_emotion:
            continue

        emotional_acts = emotional_by_emotion[emotion]
        neutral_acts = neutral_by_emotion[emotion]

        # Paired differences (assumes same order)
        n = min(len(emotional_acts), len(neutral_acts))
        diffs = [emotional_acts[i] - neutral_acts[i] for i in range(n)]

        # Mean difference
        direction = np.mean(diffs, axis=0)
        norm = float(np.linalg.norm(direction))
        unit_vector = direction / norm if norm > 0 else direction

        results[emotion] = (unit_vector, norm)

    return results


def compute_emotion_vs_opposite(
    emotional_by_emotion: Dict[str, List[np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Compute emotion - opposite_emotion direction.

    Returns:
        Dict mapping emotion -> (unit_vector, norm)
    """
    # Compute mean for each emotion
    emotion_means = {}
    for emotion, acts in emotional_by_emotion.items():
        emotion_means[emotion] = np.mean(acts, axis=0)

    results = {}
    for emotion, mean_act in emotion_means.items():
        opposite = EMOTION_OPPOSITES.get(emotion)
        if opposite is None or opposite not in emotion_means:
            continue

        mean_opposite = emotion_means[opposite]

        direction = mean_act - mean_opposite
        norm = float(np.linalg.norm(direction))
        unit_vector = direction / norm if norm > 0 else direction

        results[emotion] = (unit_vector, norm)

    return results


def compute_emotion_vs_opposite_unique(
    emotional_by_emotion: Dict[str, List[np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Compute unique bipolar direction: (emotion - opposite) - mean(all bipolar vectors).

    This isolates what's unique about each emotion's bipolar axis compared to
    the average bipolar direction shared across all emotion pairs.

    Formula: unique(emotion) = bipolar(emotion) - mean(all bipolar vectors)
    where bipolar(emotion) = mean(emotion) - mean(opposite)

    Returns:
        Dict mapping emotion -> (unit_vector, norm)
    """
    # Compute mean for each emotion
    emotion_means = {}
    for emotion, acts in emotional_by_emotion.items():
        emotion_means[emotion] = np.mean(acts, axis=0)

    # Step 1: Compute all bipolar vectors (emotion - opposite)
    bipolar_vectors = {}
    for emotion, mean_act in emotion_means.items():
        opposite = EMOTION_OPPOSITES.get(emotion)
        if opposite is None or opposite not in emotion_means:
            continue
        bipolar_vectors[emotion] = mean_act - emotion_means[opposite]

    if not bipolar_vectors:
        return {}

    # Step 2: Compute mean of all bipolar vectors
    mean_bipolar = np.mean(list(bipolar_vectors.values()), axis=0)

    # Step 3: Subtract mean bipolar to get unique component for each emotion
    results = {}
    for emotion, bipolar_vec in bipolar_vectors.items():
        unique_direction = bipolar_vec - mean_bipolar
        norm = float(np.linalg.norm(unique_direction))
        unit_vector = unique_direction / norm if norm > 0 else unique_direction
        results[emotion] = (unit_vector, norm)

    return results


def extract_directions(
    activations_dir: Path,
    output_dir: Path,
    method: str,
    representation: Optional[str] = None,
):
    """
    Extract steering directions from activations.

    Args:
        activations_dir: Directory containing layer_XX.pkl files and metadata.json
        output_dir: Output directory for vectors
        method: 'emotion_vs_others', 'emotion_vs_neutral', or 'emotion_vs_opposite'
        representation: For chat mode, 'last_token' or 'special_mean'
    """
    # Load source metadata
    metadata_path = activations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {activations_dir}")

    with open(metadata_path) as f:
        source_metadata = json.load(f)

    mode = source_metadata["mode"]
    num_layers = source_metadata["num_layers"]
    layers = source_metadata["layers"]

    # Validate method vs mode
    # Only emotion_vs_neutral requires text mode (needs neutral/emotional pairing)
    # emotion_vs_opposite and emotion_vs_opposite_unique work with any emotion data
    if method == "emotion_vs_neutral" and mode != "text":
        raise ValueError(f"Method {method} requires text mode activations (needs neutral/emotional pairs)")

    # For chat mode, require representation
    if mode == "chat" and representation is None:
        raise ValueError("--representation required for chat mode (last_token or special_mean)")

    # Create output directory structure
    if mode == "chat":
        vectors_dir = output_dir / representation
    else:
        vectors_dir = output_dir / "layers"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    # Track stats
    all_norms = {}
    n_samples = None
    emotions_found = set()

    # Process each layer
    print(f"Extracting {method} directions from {activations_dir}")
    print(f"Mode: {mode}, Layers: {num_layers}")
    if representation:
        print(f"Representation: {representation}")

    for layer_idx in tqdm(layers, desc="Processing layers"):
        layer_path = activations_dir / f"layer_{layer_idx:02d}.pkl"
        if not layer_path.exists():
            print(f"Warning: {layer_path} not found, skipping")
            continue

        # Load activations
        activations = load_layer_activations(layer_path, mode, representation)

        # Group by emotion
        emotional_by_emotion, neutral_by_emotion = group_by_emotion(activations, mode)

        # Track sample counts (once)
        if n_samples is None:
            n_samples = {e: len(acts) for e, acts in emotional_by_emotion.items()}
            if mode == "text":
                n_samples = {e: len(acts) for e, acts in emotional_by_emotion.items()}

        # Compute directions based on method
        if method == "emotion_vs_others":
            results = compute_emotion_vs_others(emotional_by_emotion)
        elif method == "emotion_vs_neutral":
            results = compute_emotion_vs_neutral(emotional_by_emotion, neutral_by_emotion)
        elif method == "emotion_vs_opposite":
            results = compute_emotion_vs_opposite(emotional_by_emotion)
        elif method == "emotion_vs_opposite_unique":
            results = compute_emotion_vs_opposite_unique(emotional_by_emotion)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Extract vectors and norms
        layer_vectors = {}
        layer_norms = {}
        for emotion, (unit_vector, norm) in results.items():
            layer_vectors[emotion] = unit_vector.astype(np.float32)
            layer_norms[emotion] = norm
            emotions_found.add(emotion)

        # Save layer vectors
        layer_output_path = vectors_dir / f"layer_{layer_idx:02d}.pkl"
        with open(layer_output_path, 'wb') as f:
            pickle.dump(layer_vectors, f)

        all_norms[f"layer_{layer_idx:02d}"] = layer_norms

    # Build metadata for this representation
    rep_key = representation if mode == "chat" else "layers"
    rep_metadata = {
        "n_samples": n_samples,
        "norms": all_norms,
    }

    # Load existing metadata if present (for merging multiple representations)
    metadata_output_path = output_dir / "metadata.json"
    if metadata_output_path.exists():
        with open(metadata_output_path) as f:
            metadata = json.load(f)
        # Update representations dict
        if "representations" not in metadata:
            metadata["representations"] = {}
        metadata["representations"][rep_key] = rep_metadata
        metadata["updated"] = datetime.now().isoformat()
    else:
        # Create new metadata
        metadata = {
            "source_metadata": {
                "model_name": source_metadata["model_name"],
                "mode": source_metadata["mode"],
                "source_representations": source_metadata["representations"],
                "start_token": source_metadata.get("start_token"),
                "hidden_dim": source_metadata["hidden_dim"],
                "num_layers": source_metadata["num_layers"],
            },
            "method": method,
            "normalized": True,
            "emotions": sorted(emotions_found),
            "representations": {
                rep_key: rep_metadata
            },
            "source_dir": str(activations_dir),
            "created": datetime.now().isoformat(),
            "script": "steering_tests/vector_extraction/extract_directions.py",
        }

    # Add opposites mapping if relevant
    if method in ["emotion_vs_opposite", "emotion_vs_opposite_unique"]:
        metadata["opposites"] = {e: EMOTION_OPPOSITES[e] for e in emotions_found if e in EMOTION_OPPOSITES}

    # Save metadata
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Saved {len(emotions_found)} emotion directions across {num_layers} layers")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract steering directions from activations")
    parser.add_argument(
        "--activations", "-a",
        type=Path,
        required=True,
        help="Directory containing activation layer_XX.pkl files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for vectors"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["emotion_vs_others", "emotion_vs_neutral", "emotion_vs_opposite", "emotion_vs_opposite_unique"],
        required=True,
        help="Extraction method"
    )
    parser.add_argument(
        "--representation", "-r",
        choices=["last_token", "special_mean"],
        help="Which representation to use (required for chat mode)"
    )

    args = parser.parse_args()

    extract_directions(
        activations_dir=args.activations,
        output_dir=args.output,
        method=args.method,
        representation=args.representation,
    )


if __name__ == "__main__":
    main()
