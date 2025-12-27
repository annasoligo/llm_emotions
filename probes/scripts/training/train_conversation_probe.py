#!/usr/bin/env python3
"""Train multiclass emotion probe on conversation data.

Supports:
- Raw mean activations (global or regional)
- Global cPCA projections (top N components)
- Regional cPCA projections (top N per region, concatenated)
- High L1 sparsity for feature selection
- Dual probe training: separate probes for user and assistant emotions

Usage:
    # Train both user and assistant probes (default)
    python scripts/train_conversation_probe.py --layer 20 --representation raw --target both

    # Train only user probe
    python scripts/train_conversation_probe.py --layer 20 --representation raw --target user

    # Global cPCA top 10 with both probes
    python scripts/train_conversation_probe.py --layer 20 --representation global_cpca --n-components 10 --target both

    # Regional cPCA top 3 per region (= 12 dims total)
    python scripts/train_conversation_probe.py --layer 20 --representation regional_cpca --n-components 3 --target both
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from probes.methods import train_multiclass_probe
from probes.core import load_cpca_results


def load_conversation_data(
    h5_path: Path,
    layer: int,
    representation: str = "raw",
    global_cpca_path: Optional[Path] = None,
    regional_cpca_paths: Optional[dict] = None,
    n_components: Optional[int] = None,
    max_samples_per_emotion: Optional[int] = None,
    target: str = "user",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load conversation activations and labels.

    Args:
        h5_path: Path to conversations2_combined.h5
        layer: Layer index
        representation: 'raw', 'global_cpca', or 'regional_cpca'
        global_cpca_path: Path to global cPCA results
        regional_cpca_paths: Dict mapping region name to cPCA result path
        n_components: Number of components to use
        max_samples_per_emotion: Max samples per emotion for balancing
        target: 'user', 'assistant', or 'both' (for dual probe training)

    Returns:
        activations: [n_samples, feature_dim]
        user_labels: [n_samples] integer labels for user emotions
        asst_labels: [n_samples] integer labels for assistant emotions
        label_names: List of emotion names
    """
    print(f"Loading conversation data from {h5_path}")
    print(f"Layer: {layer}")
    print(f"Representation: {representation}")

    # Define emotions
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    with h5py.File(h5_path, "r") as f:
        metadata = json.loads(f["metadata"][()])
        acts_group = f["activations"]

        all_activations = []
        all_user_labels = []
        all_asst_labels = []

        for entry in metadata:
            conv_id = entry["id"]
            user_emotion = entry["user_emotion"]
            asst_emotion = entry["asst_emotion"]

            # Filter: both emotions must be in our set
            if user_emotion not in emotions or asst_emotion not in emotions:
                continue

            conv_group = acts_group[conv_id]

            if representation == "raw":
                # Use global mean activations (emotional only)
                emotional = conv_group["emotional"][layer]
                acts = emotional

            elif representation == "global_cpca":
                # Project emotional activations onto global cPCA (trained on diffs)
                if global_cpca_path is None:
                    raise ValueError("global_cpca_path required for global_cpca representation")

                emotional = conv_group["emotional"][layer]
                # Use emotional activations directly (cPCA was trained on diffs)
                acts = emotional

            elif representation == "regional_cpca":
                # Concatenate regional cPCA projections
                if regional_cpca_paths is None:
                    raise ValueError("regional_cpca_paths required for regional_cpca representation")

                # Will concatenate regional projections after loading all data
                acts = {}
                for region in ["user", "asst", "special1", "special2"]:
                    regional_group = conv_group["regional"][region]
                    emotional = regional_group["emotional"][layer]
                    # Use emotional activations directly (cPCA was trained on diffs)
                    acts[region] = emotional

            else:
                raise ValueError(f"Unknown representation: {representation}")

            all_activations.append(acts)
            all_user_labels.append(emotion_to_idx[user_emotion])
            all_asst_labels.append(emotion_to_idx[asst_emotion])

    print(f"Loaded {len(all_activations)} conversations")

    # Project onto cPCA if needed
    if representation == "global_cpca":
        print(f"\nProjecting onto global cPCA (top {n_components} components)...")
        cpca_data = load_cpca_results(global_cpca_path)
        components = cpca_data["components"][layer]  # [n_components, hidden_dim]

        if n_components is not None:
            components = components[:n_components]

        print(f"Using {components.shape[0]} components")
        activations = np.array(all_activations).astype(np.float32)  # [n_samples, hidden_dim]
        activations = activations @ components.T  # [n_samples, n_components]

    elif representation == "regional_cpca":
        print(f"\nProjecting onto regional cPCA (top {n_components} per region)...")

        # Load cPCA for each region
        regional_projections = []
        for region in ["user", "asst", "special1", "special2"]:
            cpca_data = load_cpca_results(regional_cpca_paths[region])
            components = cpca_data["components"][layer]  # [n_components, hidden_dim]

            if n_components is not None:
                components = components[:n_components]

            print(f"  {region}: {components.shape[0]} components")

            # Project each sample's regional acts
            region_acts = np.array([acts[region] for acts in all_activations]).astype(np.float32)
            projected = region_acts @ components.T
            regional_projections.append(projected)

        # Concatenate all regional projections
        activations = np.concatenate(regional_projections, axis=1)
        print(f"Total feature dim: {activations.shape[1]} (= {n_components} × 4 regions)")

    else:  # raw
        activations = np.array(all_activations).astype(np.float32)

    user_labels = np.array(all_user_labels)
    asst_labels = np.array(all_asst_labels)

    print(f"Activation shape: {activations.shape}")
    print(f"User labels shape: {user_labels.shape}")
    print(f"Assistant labels shape: {asst_labels.shape}")

    # Class distribution
    print("\nUser emotion distribution (before balancing):")
    for i, emotion in enumerate(emotions):
        count = np.sum(user_labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    print("\nAssistant emotion distribution (before balancing):")
    for i, emotion in enumerate(emotions):
        count = np.sum(asst_labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    # Balance dataset based on target
    if target == "user":
        labels_for_balancing = user_labels
    elif target == "assistant":
        labels_for_balancing = asst_labels
    else:  # both
        # For dual probes, balance based on user emotions
        labels_for_balancing = user_labels

    if max_samples_per_emotion is None:
        emotion_counts = [np.sum(labels_for_balancing == i) for i in range(len(emotions))]
        max_samples_per_emotion = min(emotion_counts)
        print(f"\nAuto-balancing to {max_samples_per_emotion} samples per class")

    balanced_acts = []
    balanced_user_labels = []
    balanced_asst_labels = []

    for i, emotion in enumerate(emotions):
        mask = labels_for_balancing == i
        emotion_acts = activations[mask]
        emotion_user_labels = user_labels[mask]
        emotion_asst_labels = asst_labels[mask]

        if len(emotion_acts) > max_samples_per_emotion:
            indices = np.random.choice(
                len(emotion_acts), max_samples_per_emotion, replace=False
            )
            emotion_acts = emotion_acts[indices]
            emotion_user_labels = emotion_user_labels[indices]
            emotion_asst_labels = emotion_asst_labels[indices]

        balanced_acts.append(emotion_acts)
        balanced_user_labels.append(emotion_user_labels)
        balanced_asst_labels.append(emotion_asst_labels)

    activations = np.vstack(balanced_acts)
    user_labels = np.concatenate(balanced_user_labels)
    asst_labels = np.concatenate(balanced_asst_labels)

    print(f"\nBalanced dataset: {len(activations)} samples")
    print("\nBalanced user emotion distribution:")
    for i, emotion in enumerate(emotions):
        count = np.sum(user_labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    print("\nBalanced assistant emotion distribution:")
    for i, emotion in enumerate(emotions):
        count = np.sum(asst_labels == i)
        print(f"  {emotion:12s}: {count:6d} samples")

    return activations, user_labels, asst_labels, emotions


def main():
    parser = argparse.ArgumentParser(description="Train conversation emotion probe")
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace-vast/annas/git/research-tools/data/activations/conversations2_combined.h5",
        help="Path to conversations2_combined.h5",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to train probe on",
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["raw", "global_cpca", "regional_cpca"],
        default="raw",
        help="Feature representation",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of cPCA components (per region for regional_cpca)",
    )
    parser.add_argument(
        "--global-cpca",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results/cpca_conversations_global/google/gemma-3-27b-it_cpca.npz",
        help="Path to global cPCA results",
    )
    parser.add_argument(
        "--regional-cpca-dir",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results",
        help="Directory containing regional cPCA results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per emotion (default: auto-balance)",
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
        help="L2 regularization (default: 1.0)",
    )
    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=0.1,
        help="L1 regularization for sparsity (default: 0.1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes",
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
        "--target",
        type=str,
        choices=["user", "assistant", "both"],
        default="both",
        help="Train probe on user emotions, assistant emotions, or both",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Prepare regional cPCA paths if needed
    regional_cpca_paths = None
    if args.representation == "regional_cpca":
        base_dir = Path(args.regional_cpca_dir)
        regional_cpca_paths = {
            "user": base_dir / "cpca_conversations_regional_user/google/gemma-3-27b-it_cpca.npz",
            "asst": base_dir / "cpca_conversations_regional_asst/google/gemma-3-27b-it_cpca.npz",
            "special1": base_dir / "cpca_conversations_regional_special1/google/gemma-3-27b-it_cpca.npz",
            "special2": base_dir / "cpca_conversations_regional_special2/google/gemma-3-27b-it_cpca.npz",
        }

    # Load data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Target: {args.target}")
    activations, user_labels, asst_labels, label_names = load_conversation_data(
        Path(args.data),
        args.layer,
        representation=args.representation,
        global_cpca_path=Path(args.global_cpca) if args.representation == "global_cpca" else None,
        regional_cpca_paths=regional_cpca_paths,
        n_components=args.n_components,
        max_samples_per_emotion=args.max_samples,
        target=args.target,
    )

    # Train/test split
    print(f"\nSplitting data: {1-args.test_size:.0%} train, {args.test_size:.0%} test")

    # Use user_labels for stratification (same splits for user and assistant)
    train_acts, test_acts, train_user_labels, test_user_labels, train_asst_labels, test_asst_labels = train_test_split(
        activations,
        user_labels,
        asst_labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=user_labels,
    )

    print(f"Train samples: {len(train_acts)}")
    print(f"Test samples: {len(test_acts)}")

    # Train probe(s)
    print("\n" + "=" * 80)
    print("TRAINING PROBE(S)")
    print("=" * 80)
    print(f"L1 lambda: {args.l1_lambda} (high sparsity)")

    results = {}

    if args.target in ["user", "both"]:
        print("\n" + "-" * 80)
        print("TRAINING USER EMOTION PROBE")
        print("-" * 80)
        user_results = train_multiclass_probe(
            train_activations=train_acts,
            train_labels=train_user_labels,
            test_activations=test_acts,
            test_labels=test_user_labels,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            l1_lambda=args.l1_lambda,
            device=args.device,
        )
        results["user"] = user_results
        print(f"\n✓ User probe - Test accuracy: {user_results['test_accuracy']:.4f}")

    if args.target in ["assistant", "both"]:
        print("\n" + "-" * 80)
        print("TRAINING ASSISTANT EMOTION PROBE")
        print("-" * 80)
        asst_results = train_multiclass_probe(
            train_activations=train_acts,
            train_labels=train_asst_labels,
            test_activations=test_acts,
            test_labels=test_asst_labels,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            weight_decay=args.weight_decay,
            l1_lambda=args.l1_lambda,
            device=args.device,
        )
        results["assistant"] = asst_results
        print(f"\n✓ Assistant probe - Test accuracy: {asst_results['test_accuracy']:.4f}")

    # Add cross-accuracy metrics if training both
    if args.target == "both":
        print("\n" + "-" * 80)
        print("CROSS-ACCURACY (Disentanglement Metrics)")
        print("-" * 80)

        import torch
        from sklearn.metrics import accuracy_score

        # User probe on assistant labels
        user_probe = results["user"]["model"]
        user_probe.eval()
        with torch.no_grad():
            test_acts_tensor = torch.from_numpy(test_acts).float().to(args.device)
            user_logits = user_probe(test_acts_tensor)
            user_preds_on_asst = user_logits.argmax(dim=1).cpu().numpy()
        user_on_asst = accuracy_score(test_asst_labels, user_preds_on_asst)
        results["user_on_asst_accuracy"] = user_on_asst
        print(f"User probe on assistant labels: {user_on_asst:.4f}")

        # Assistant probe on user labels
        asst_probe = results["assistant"]["model"]
        asst_probe.eval()
        with torch.no_grad():
            asst_logits = asst_probe(test_acts_tensor)
            asst_preds_on_user = asst_logits.argmax(dim=1).cpu().numpy()
        asst_on_user = accuracy_score(test_user_labels, asst_preds_on_user)
        results["asst_on_user_accuracy"] = asst_on_user
        print(f"Assistant probe on user labels: {asst_on_user:.4f}")

        # Disentanglement quality
        user_self_acc = results["user"]["test_accuracy"]
        asst_self_acc = results["assistant"]["test_accuracy"]
        user_disentangle = user_self_acc - user_on_asst
        asst_disentangle = asst_self_acc - asst_on_user

        print(f"\nDisentanglement scores:")
        print(f"  User: {user_disentangle:.4f} (self {user_self_acc:.4f} - cross {user_on_asst:.4f})")
        print(f"  Assistant: {asst_disentangle:.4f} (self {asst_self_acc:.4f} - cross {asst_on_user:.4f})")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    rep_str = args.representation
    n_comp_str = f"_top{args.n_components}" if args.n_components else ""
    l1_str = f"_l1{args.l1_lambda}" if args.l1_lambda > 0 else ""
    target_str = f"_{args.target}" if args.target != "both" else ""
    filename = f"probe_layer{args.layer}_{rep_str}{n_comp_str}{l1_str}{target_str}.pkl"
    output_path = output_dir / filename

    # Add metadata
    results["label_names"] = label_names
    results["layer"] = args.layer
    results["representation"] = args.representation
    results["n_components"] = args.n_components
    results["target"] = args.target
    results["args"] = vars(args)

    # Save
    print(f"\nSaving results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    # Save summary
    summary_path = output_dir / filename.replace(".pkl", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CONVERSATION EMOTION PROBE TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Target: {args.target}\n")
        f.write(f"Layer: {args.layer}\n")
        f.write(f"Representation: {args.representation}\n")
        if args.n_components:
            if args.representation == "regional_cpca":
                f.write(f"Components: {args.n_components} per region (= {args.n_components * 4} total)\n")
            else:
                f.write(f"Components: {args.n_components}\n")
        f.write(f"Input dimensionality: {activations.shape[1]}\n\n")
        f.write(f"Train samples: {len(train_acts)}\n")
        f.write(f"Test samples: {len(test_acts)}\n\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"L1 lambda: {args.l1_lambda}\n")
        f.write(f"L2 weight decay: {args.weight_decay}\n\n")
        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n\n")

        if args.target in ["user", "both"]:
            user_res = results["user"]
            f.write("USER EMOTION PROBE:\n")
            f.write(f"  Best epoch: {user_res['best_epoch'] + 1}\n")
            f.write(f"  Train accuracy: {user_res['train_accuracy']:.4f}\n")
            f.write(f"  Test accuracy: {user_res['test_accuracy']:.4f}\n")
            f.write(f"  Best test accuracy: {user_res['best_test_accuracy']:.4f}\n\n")

        if args.target in ["assistant", "both"]:
            asst_res = results["assistant"]
            f.write("ASSISTANT EMOTION PROBE:\n")
            f.write(f"  Best epoch: {asst_res['best_epoch'] + 1}\n")
            f.write(f"  Train accuracy: {asst_res['train_accuracy']:.4f}\n")
            f.write(f"  Test accuracy: {asst_res['test_accuracy']:.4f}\n")
            f.write(f"  Best test accuracy: {asst_res['best_test_accuracy']:.4f}\n\n")

        if args.target == "both":
            f.write("CROSS-ACCURACY (Disentanglement):\n")
            f.write(f"  User probe on assistant labels: {results['user_on_asst_accuracy']:.4f}\n")
            f.write(f"  Assistant probe on user labels: {results['asst_on_user_accuracy']:.4f}\n\n")
            user_disentangle = results["user"]["test_accuracy"] - results["user_on_asst_accuracy"]
            asst_disentangle = results["assistant"]["test_accuracy"] - results["asst_on_user_accuracy"]
            f.write(f"  User disentanglement score: {user_disentangle:.4f}\n")
            f.write(f"  Assistant disentanglement score: {asst_disentangle:.4f}\n")

    print(f"Saved summary to {summary_path}")

    if args.target == "both":
        print(f"\n✓ Training complete!")
        print(f"  User probe test accuracy: {results['user']['test_accuracy']:.4f}")
        print(f"  Assistant probe test accuracy: {results['assistant']['test_accuracy']:.4f}")
    elif args.target == "user":
        print(f"\n✓ Training complete! User probe test accuracy: {results['user']['test_accuracy']:.4f}")
    else:
        print(f"\n✓ Training complete! Assistant probe test accuracy: {results['assistant']['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
