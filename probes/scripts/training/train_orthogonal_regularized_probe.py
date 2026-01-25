#!/usr/bin/env python3
"""Train orthogonality-regularized emotion probe on text data.

This script trains a linear emotion classifier with an orthogonality penalty
that encourages probe weights to be orthogonal to neutral-dominated principal
components.

Workflow:
1. Load emotional activations from texts.h5
2. Load pre-computed neutral PCs (from compute_neutral_pcs_for_regularization.py)
3. Split data into train/test
4. Train probe with orthogonality regularization: L = L_emotion + λ * ||U.T @ w||²
5. Save trained probe and results

Usage:
    # Train with orthogonality regularization
    python train_orthogonal_regularized_probe.py \
        --data data/activations/texts.h5 \
        --layer 20 \
        --neutral-pcs results/neutral_pcs/layer_20_neutral_pcs_k20.npy \
        --lambda-ortho 1.0

    # Train baseline (no orthogonality regularization)
    python train_orthogonal_regularized_probe.py \
        --data data/activations/texts.h5 \
        --layer 20 \
        --lambda-ortho 0.0

    # Sweep lambda values
    for lambda in 0.01 0.1 1.0 10.0 100.0; do
        python train_orthogonal_regularized_probe.py \
            --layer 20 \
            --neutral-pcs results/neutral_pcs/layer_20_neutral_pcs_k20.npy \
            --lambda-ortho $lambda \
            --output-dir results/ortho_probes/lambda_sweep
    done
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from probes.methods.orthogonal_regularized_probe import train_orthogonal_regularized_probe


def load_emotion_activations(
    h5_path: Path,
    layer: int,
    tiers: list[str] = None,
    max_samples_per_emotion: int = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load emotion activations from texts.h5.

    Args:
        h5_path: Path to texts.h5 file
        layer: Layer index to extract
        tiers: List of tier names to include. If None, includes all tiers
        max_samples_per_emotion: Max samples per emotion (for balancing)

    Returns:
        activations: [n_samples, hidden_dim] emotional activations
        labels: [n_samples] integer labels (0-6: 6 emotions + neutral)
        label_names: List of emotion names
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
        tier = "_".join(parts[2:-1])
        emotion = parts[-1]

        # Filter by tier if specified
        if tiers is not None and tier not in tiers:
            continue

        # Filter by emotion (exclude neutral for now, we'll add it separately)
        if emotion not in emotions[:-1]:  # Exclude "neutral" from list
            continue

        # Load emotional activations
        emotional_acts = acts_group[key]["emotional"][layer]

        all_activations.append(emotional_acts)
        all_labels.append(emotion_to_idx[emotion])

    # Now add neutral activations
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
        neutral_acts = acts_group[key]["neutral"][layer]

        all_activations.append(neutral_acts)
        all_labels.append(emotion_to_idx["neutral"])

    f.close()

    # Convert to arrays
    activations = np.array(all_activations)
    labels = np.array(all_labels)

    print(f"Loaded {len(activations)} samples")
    print(f"Activation shape: {activations.shape}")

    # Class distribution before balancing
    print("\nClass distribution (before balancing):")
    for i, emotion in enumerate(emotions):
        count = np.sum(labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    # Auto-balance: use min count across all classes
    class_counts = [np.sum(labels == i) for i in range(len(emotions))]
    if max_samples_per_emotion is None:
        max_samples_per_emotion = min(class_counts)
        print(f"\nAuto-balancing to {max_samples_per_emotion} samples per class")

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
    print("Balanced class distribution:")
    for i, emotion in enumerate(emotions):
        count = np.sum(labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    return activations, labels, emotions


def main():
    parser = argparse.ArgumentParser(
        description="Train orthogonality-regularized emotion probe"
    )
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
        "--neutral-pcs",
        type=str,
        default=None,
        help="Path to neutral PCs .npy file. If not provided, trains standard probe.",
    )
    parser.add_argument(
        "--lambda-ortho",
        type=float,
        default=1.0,
        help="Orthogonality penalty weight (default: 1.0). Set to 0 for baseline.",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=None,
        help="Tiers to include (e.g., direct_address third_person). If not specified, uses all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per emotion. If not specified, auto-balances to min count.",
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
        "--output-dir",
        type=str,
        default="results/orthogonal_regularized_probes",
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

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print("=" * 80)
    print("ORTHOGONALITY-REGULARIZED EMOTION PROBE TRAINING")
    print("=" * 80)

    # Load neutral PCs if provided
    neutral_pcs = None
    if args.neutral_pcs is not None:
        print("\nLoading neutral PCs...")
        neutral_pcs_path = Path(args.neutral_pcs)
        if not neutral_pcs_path.exists():
            raise FileNotFoundError(f"Neutral PCs not found: {neutral_pcs_path}")

        neutral_pcs = np.load(neutral_pcs_path)
        print(f"Loaded neutral PCs from: {neutral_pcs_path}")
        print(f"Neutral PCs shape: {neutral_pcs.shape}")
        print(f"  -> [hidden_dim={neutral_pcs.shape[0]}, k={neutral_pcs.shape[1]}]")
    else:
        print("\nNo neutral PCs provided - training standard probe")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING ACTIVATION DATA")
    print("=" * 80)
    activations, labels, label_names = load_emotion_activations(
        Path(args.data),
        args.layer,
        tiers=args.tiers,
        max_samples_per_emotion=args.max_samples,
    )

    # Validate dimensions match
    if neutral_pcs is not None:
        if activations.shape[1] != neutral_pcs.shape[0]:
            raise ValueError(
                f"Activation dimension ({activations.shape[1]}) must match "
                f"neutral PCs dimension ({neutral_pcs.shape[0]})"
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

    results = train_orthogonal_regularized_probe(
        train_activations=train_acts,
        train_labels=train_labels,
        test_activations=test_acts,
        test_labels=test_labels,
        neutral_pcs=neutral_pcs,
        lambda_ortho=args.lambda_ortho,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        device=args.device,
        verbose=True,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    tier_str = "_".join(args.tiers) if args.tiers else "all"
    if neutral_pcs is not None:
        k = neutral_pcs.shape[1]
        filename = f"probe_layer{args.layer}_{tier_str}_ortho_k{k}_lambda{args.lambda_ortho}.pkl"
    else:
        filename = f"probe_layer{args.layer}_{tier_str}_baseline.pkl"

    output_path = output_dir / filename

    # Add metadata
    results["label_names"] = label_names
    results["layer"] = args.layer
    results["tiers"] = args.tiers
    results["data_path"] = str(args.data)
    results["neutral_pcs_path"] = str(args.neutral_pcs) if args.neutral_pcs else None
    results["args"] = vars(args)

    # Save
    print(f"\nSaving results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    # Also save a text summary
    summary_path = output_dir / filename.replace(".pkl", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ORTHOGONALITY-REGULARIZED PROBE TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Tiers: {args.tiers if args.tiers else 'all'}\n")
        f.write(f"Max samples per emotion: {args.max_samples}\n")
        f.write(f"Input dimensionality: {activations.shape[1]}\n")
        f.write(f"Train samples: {len(train_acts)}\n")
        f.write(f"Test samples: {len(test_acts)}\n\n")

        f.write("ORTHOGONALITY REGULARIZATION\n")
        f.write("-" * 80 + "\n")
        if neutral_pcs is not None:
            f.write(f"Neutral PCs: {args.neutral_pcs}\n")
            f.write(f"Neutral PC dimensions: {neutral_pcs.shape}\n")
            f.write(f"Lambda orthogonality: {args.lambda_ortho}\n")
        else:
            f.write("No orthogonality regularization (baseline)\n")
        f.write("\n")

        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Max epochs: {args.max_epochs}\n")
        f.write(f"Patience: {args.patience}\n")
        f.write(f"Weight decay (L2): {args.weight_decay}\n\n")

        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best epoch: {results['best_epoch'] + 1}\n")
        f.write(f"Total epochs: {results['total_epochs']}\n\n")
        f.write(f"Train accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Test accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"Best test accuracy: {results['best_test_accuracy']:.4f}\n\n")

        if neutral_pcs is not None:
            f.write("ORTHOGONALITY METRICS\n")
            f.write("-" * 80 + "\n")
            om = results['ortho_metrics']
            f.write(f"Frobenius norm: {om['ortho_frobenius']:.4f}\n")
            f.write(f"Normalized overlap: {om['ortho_normalized']:.4f}\n")
            f.write(f"Neutral fraction: {om['neutral_fraction']:.4f}\n")
            f.write("\n")
            f.write("Interpretation:\n")
            f.write("  - Normalized overlap: 0 = fully orthogonal, 1 = fully aligned\n")
            f.write("  - Neutral fraction: fraction of probe weight in neutral subspace\n")
            f.write("\n")

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
