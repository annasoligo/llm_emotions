#!/usr/bin/env python3
"""Train multiclass emotion probe on texts.h5 data.

This script:
1. Loads emotional and neutral activations from texts.h5
2. Creates a 7-class dataset (6 emotions + neutral)
3. Optionally projects onto cPCA components for dimensionality reduction
4. Splits 80/20 train/test
5. Trains a linear probe with early stopping
6. Saves the trained probe and results

Usage:
    # Train on raw activations (5376-dim)
    python scripts/train_emotion_probe.py --layer 20 --data data/activations/texts.h5

    # Train on cPCA projections (50-dim)
    python scripts/train_emotion_probe.py --layer 20 --data data/activations/texts.h5 \
        --use-cpca --cpca-results probes/results/cpca_tier_data/google/gemma-3-27b-it_cpca.npz
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from probes.methods import train_multiclass_probe
from probes.core import load_cpca_results


def load_texts_data(
    h5_path: Path,
    layer: int,
    tiers: List[str] = None,
    max_samples_per_emotion: int = None,
    cpca_components: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load activations and labels from texts.h5.

    Args:
        h5_path: Path to texts.h5 file
        layer: Layer index to extract
        tiers: List of tier names to include (e.g., ['direct_address', 'third_person'])
               If None, includes all tiers
        max_samples_per_emotion: Max samples per emotion (for balancing)
        cpca_components: Optional cPCA components [n_components, hidden_dim] to project onto
        n_components: If specified, only use first N components (requires cpca_components)

    Returns:
        activations: [n_samples, hidden_dim] or [n_samples, n_components] if using cPCA
        labels: [n_samples] integer labels
        label_names: List of emotion names corresponding to label integers
    """
    print(f"Loading data from {h5_path}")
    print(f"Layer: {layer}")
    print(f"Tiers: {tiers if tiers else 'all'}")

    f = h5py.File(h5_path, "r")
    acts_group = f["activations"]

    # Define emotion mapping
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    # Collect all activations
    all_activations = []
    all_labels = []

    # First, collect emotional activations
    for key in acts_group.keys():
        # Parse key: set_X_TIER_EMOTION
        parts = key.split("_")
        if len(parts) < 4:
            continue

        # Extract tier and emotion
        tier = "_".join(parts[2:-1])  # Everything between set_X and emotion
        emotion = parts[-1]

        # Filter by tier if specified
        if tiers is not None and tier not in tiers:
            continue

        # Filter by emotion (exclude neutral for now, we'll add it separately)
        if emotion not in emotions[:-1]:  # Exclude "neutral" from list
            continue

        # Load emotional activations
        emotional_acts = acts_group[key]["emotional"][layer]  # [hidden_dim]

        all_activations.append(emotional_acts)
        all_labels.append(emotion_to_idx[emotion])

    # Now add neutral activations (using the same keys but from "neutral" field)
    for key in acts_group.keys():
        parts = key.split("_")
        if len(parts) < 4:
            continue

        tier = "_".join(parts[2:-1])
        emotion = parts[-1]

        # Filter by tier if specified
        if tiers is not None and tier not in tiers:
            continue

        # Only process emotions we care about
        if emotion not in emotions[:-1]:
            continue

        # Load neutral activations
        neutral_acts = acts_group[key]["neutral"][layer]  # [hidden_dim]

        all_activations.append(neutral_acts)
        all_labels.append(emotion_to_idx["neutral"])

    f.close()

    # Convert to arrays
    activations = np.array(all_activations)
    labels = np.array(all_labels)

    print(f"Loaded {len(activations)} samples")
    print(f"Raw activation shape: {activations.shape}")

    # Project onto cPCA components if provided
    if cpca_components is not None:
        # Optionally limit to first N components
        if n_components is not None and n_components < cpca_components.shape[0]:
            print(f"\nUsing only first {n_components} components (out of {cpca_components.shape[0]})")
            cpca_components = cpca_components[:n_components]

        print(f"\nProjecting onto cPCA components...")
        print(f"cPCA components shape: {cpca_components.shape}")
        print(f"Projection: [{activations.shape[0]}, {activations.shape[1]}] @ [{cpca_components.shape[1]}, {cpca_components.shape[0]}]^T")
        print(f"           -> [{activations.shape[0]}, {cpca_components.shape[0]}]")

        # Project: [n_samples, hidden_dim] @ [hidden_dim, n_components] -> [n_samples, n_components]
        activations = activations.astype(np.float32) @ cpca_components.T.astype(np.float32)
        print(f"Projected activation shape: {activations.shape}")

    # Class distribution before balancing
    print("\nClass distribution (before balancing):")
    for i, emotion in enumerate(emotions):
        count = np.sum(labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    # Auto-balance: use min count across emotions (excluding neutral) or max_samples if specified
    emotion_counts = [np.sum(labels == i) for i in range(len(emotions) - 1)]  # Exclude neutral
    if max_samples_per_emotion is None:
        max_samples_per_emotion = min(emotion_counts)
        print(f"\nAuto-balancing to {max_samples_per_emotion} samples per class (min emotion count)")

    # Balance dataset
    balanced_acts = []
    balanced_labels = []

    for i, emotion in enumerate(emotions):
        mask = labels == i
        emotion_acts = activations[mask]
        emotion_labels = labels[mask]

        if len(emotion_acts) > max_samples_per_emotion:
            # Randomly sample
            indices = np.random.choice(
                len(emotion_acts), max_samples_per_emotion, replace=False
            )
            emotion_acts = emotion_acts[indices]
            emotion_labels = emotion_labels[indices]

        balanced_acts.append(emotion_acts)
        balanced_labels.append(emotion_labels)

    activations = np.vstack(balanced_acts)
    labels = np.concatenate(balanced_labels)

    print(f"\nBalanced dataset: {len(activations)} samples")
    print("\nBalanced class distribution:")
    for i, emotion in enumerate(emotions):
        count = np.sum(labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    return activations, labels, emotions


def main():
    parser = argparse.ArgumentParser(description="Train multiclass emotion probe")
    parser.add_argument(
        "--data",
        type=str,
        default="data/activations/texts.h5",
        help="Path to texts.h5 file",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to train probe on",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=None,
        help="Tiers to include (e.g., direct_address third_person). If not specified, uses all tiers.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per emotion. If not specified, auto-balances to min emotion count (default: auto)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=500,
        help="Max epochs (default: 500)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0,
        help="L2 regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.0,
        help="L1 regularization strength (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Use only first N cPCA components (requires --use-cpca)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/emotion_probes",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--use-cpca",
        action="store_true",
        help="Use cPCA-projected activations instead of raw activations",
    )
    parser.add_argument(
        "--cpca-results",
        type=str,
        default="probes/results/cpca_tier_data/google/gemma-3-27b-it_cpca.npz",
        help="Path to cPCA results NPZ file (only used if --use-cpca is set)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Load cPCA components if requested
    cpca_components = None
    if args.use_cpca:
        print("=" * 80)
        print("LOADING cPCA COMPONENTS")
        print("=" * 80)
        cpca_path = Path(args.cpca_results)
        if not cpca_path.exists():
            raise FileNotFoundError(f"cPCA results not found: {cpca_path}")

        cpca_data = load_cpca_results(cpca_path)
        print(f"Loaded cPCA results from {cpca_path}")
        print(f"Components shape: {cpca_data['components'].shape}")
        print(f"  -> [n_layers={cpca_data['components'].shape[0]}, "
              f"n_components={cpca_data['components'].shape[1]}, "
              f"hidden_dim={cpca_data['components'].shape[2]}]")

        # Extract components for this layer: [n_components, hidden_dim]
        cpca_components = cpca_data["components"][args.layer]
        print(f"\nUsing components for layer {args.layer}: {cpca_components.shape}")
        print(f"Dimensionality reduction: {cpca_components.shape[1]} -> {cpca_components.shape[0]}")
        print()

    # Load data
    print("=" * 80)
    print("LOADING ACTIVATION DATA")
    print("=" * 80)
    activations, labels, label_names = load_texts_data(
        Path(args.data),
        args.layer,
        tiers=args.tiers,
        max_samples_per_emotion=args.max_samples,
        cpca_components=cpca_components,
        n_components=args.n_components,
    )

    # Train/test split
    print(f"\nSplitting data: {1-args.test_size:.0%} train, {args.test_size:.0%} test")
    train_acts, test_acts, train_labels, test_labels = train_test_split(
        activations,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    print(f"Train samples: {len(train_acts)}")
    print(f"Test samples: {len(test_acts)}")

    # Train probe
    print("\n" + "=" * 80)
    print("TRAINING PROBE")
    print("=" * 80)

    results = train_multiclass_probe(
        train_activations=train_acts,
        train_labels=train_labels,
        test_activations=test_acts,
        test_labels=test_labels,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        device=args.device,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    tier_str = "_".join(args.tiers) if args.tiers else "all"
    cpca_str = "_cpca" if args.use_cpca else ""
    n_comp_str = f"_top{args.n_components}" if args.n_components else ""
    l1_str = f"_l1" if args.l1_lambda > 0 else ""
    filename = f"probe_layer{args.layer}_{tier_str}{cpca_str}{n_comp_str}{l1_str}.pkl"
    output_path = output_dir / filename

    # Add metadata
    results["label_names"] = label_names
    results["layer"] = args.layer
    results["tiers"] = args.tiers
    results["data_path"] = str(args.data)
    results["use_cpca"] = args.use_cpca
    results["cpca_results_path"] = str(args.cpca_results) if args.use_cpca else None
    results["args"] = vars(args)

    # Save
    print(f"\nSaving results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    # Also save a text summary
    summary_path = output_dir / filename.replace(".pkl", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EMOTION PROBE TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Tiers: {args.tiers if args.tiers else 'all'}\n")
        f.write(f"Max samples per emotion: {args.max_samples}\n")
        f.write(f"Using cPCA: {args.use_cpca}\n")
        if args.use_cpca:
            f.write(f"cPCA results: {args.cpca_results}\n")
            if args.n_components:
                f.write(f"Using top {args.n_components} components\n")
            f.write(f"Input dimensionality: {activations.shape[1]} (cPCA-projected)\n")
        else:
            f.write(f"Input dimensionality: {activations.shape[1]} (raw)\n")
        f.write("\n")
        f.write(f"Train samples: {len(train_acts)}\n")
        f.write(f"Test samples: {len(test_acts)}\n\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Patience: {args.patience}\n")
        f.write(f"Weight decay (L2): {args.weight_decay}\n")
        f.write(f"L1 lambda: {args.l1_lambda}\n\n")
        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best epoch: {results['best_epoch'] + 1}\n")
        f.write(f"Total epochs: {results['total_epochs']}\n\n")
        f.write(f"Train accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Test accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"Best test accuracy: {results['best_test_accuracy']:.4f}\n\n")
        f.write("=" * 80 + "\n")
        f.write("LABEL MAPPING\n")
        f.write("=" * 80 + "\n\n")
        for i, name in enumerate(label_names):
            f.write(f"{i}: {name}\n")

    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
