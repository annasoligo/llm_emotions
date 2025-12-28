#!/usr/bin/env python3
"""Train emotion probes with soft orthogonality constraints between user and assistant directions.

Enforces soft orthogonality: all user_probes ⊥ all assistant_probes (global subspace separation)

Usage:
    # Raw mean activations
    python scripts/train_orthogonal_conversation_probe.py --layer 20 --representation raw --ortho-weight 1.0

    # Global cPCA
    python scripts/train_orthogonal_conversation_probe.py --layer 20 --representation global_cpca --n-components 10 --ortho-weight 1.0

    # Regional cPCA
    python scripts/train_orthogonal_conversation_probe.py --layer 20 --representation regional_cpca --n-components 5 --ortho-weight 1.0
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from probes.core import load_cpca_results


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


class OrthogonalEmotionProbes(nn.Module):
    """Emotion probes with orthogonality constraints between user and assistant subspaces."""

    def __init__(self, hidden_dim, n_emotions=6):
        super().__init__()
        # User and assistant probe directions
        self.user_probes = nn.Parameter(torch.randn(n_emotions, hidden_dim))
        self.assistant_probes = nn.Parameter(torch.randn(n_emotions, hidden_dim))

        # Initialize with small values for stability
        nn.init.xavier_normal_(self.user_probes, gain=0.1)
        nn.init.xavier_normal_(self.assistant_probes, gain=0.1)

    def forward(self, activations, probe_type='user'):
        """Project activations onto probe directions"""
        probes = self.user_probes if probe_type == 'user' else self.assistant_probes
        # Normalize probes
        probes_norm = probes / probes.norm(dim=1, keepdim=True)
        return activations @ probes_norm.T  # [batch, n_emotions]

    def orthogonality_loss(self):
        """Penalize non-orthogonality between user and assistant subspaces.

        Returns the mean squared dot product across ALL pairs of user/assistant probes.
        This enforces that the entire user emotion subspace is orthogonal to the
        entire assistant emotion subspace.
        """
        user_norm = self.user_probes / self.user_probes.norm(dim=1, keepdim=True)
        asst_norm = self.assistant_probes / self.assistant_probes.norm(dim=1, keepdim=True)

        # Cross-group orthogonality: user probes ⊥ assistant probes
        # Shape: [n_emotions, n_emotions] - all pairwise dot products
        cross_dots = user_norm @ asst_norm.T
        cross_loss = (cross_dots ** 2).mean()

        return cross_loss

    def get_normalized_probes(self, probe_type='user'):
        """Get normalized probe vectors"""
        probes = self.user_probes if probe_type == 'user' else self.assistant_probes
        return (probes / probes.norm(dim=1, keepdim=True)).detach().cpu().numpy()


class EmotionDataset(Dataset):
    """Dataset of emotion-labeled activations"""

    def __init__(self, activations, labels, speaker_types):
        """
        Args:
            activations: [n_samples, hidden_dim] tensor
            labels: [n_samples] integer labels (not one-hot)
            speaker_types: [n_samples] list of 'user' or 'assistant'
        """
        self.activations = activations
        self.labels = labels
        self.speaker_types = speaker_types

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return {
            'activation': self.activations[idx],
            'label': self.labels[idx],
            'speaker_type': self.speaker_types[idx],
        }


def load_conversation_data(
    h5_path: Path,
    layer: int,
    representation: str = "raw",
    global_cpca_path: Optional[Path] = None,
    regional_cpca_paths: Optional[dict] = None,
    n_components: Optional[int] = None,
    max_samples_per_emotion: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load conversation activations and labels.

    Returns:
        activations: [n_samples, feature_dim]
        user_labels: [n_samples] integer labels for user emotions
        asst_labels: [n_samples] integer labels for assistant emotions
        label_names: List of emotion names
    """
    print(f"Loading conversation data from {h5_path}")
    print(f"Layer: {layer}")
    print(f"Representation: {representation}")

    emotions = EMOTIONS
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
                # Load regional activations
                if regional_cpca_paths is None:
                    raise ValueError("regional_cpca_paths required for regional_cpca representation")

                acts = {}
                for region in ["user", "asst", "special1", "special2"]:
                    # Note: HDF5 structure is conv_group["regional"][region], not conv_group[f"regional_{region}"]
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

        if n_components:
            components = components[:n_components]

        print(f"Components shape: {components.shape}")

        # Project all activations
        activations = np.array(all_activations).astype(np.float32)  # [n_conv, hidden_dim]
        activations = activations @ components.T  # [n_conv, n_components]
        print(f"Projected shape: {activations.shape}")

    elif representation == "regional_cpca":
        print(f"\nProjecting onto regional cPCA (top {n_components} per region)...")

        regional_projections = []
        for region in ["user", "asst", "special1", "special2"]:
            cpca_path = regional_cpca_paths[region]
            cpca_data = load_cpca_results(cpca_path)
            components = cpca_data["components"][layer]  # [n_components, hidden_dim]

            if n_components:
                components = components[:n_components]

            print(f"  {region}: {components.shape}")

            # Extract regional activations and project
            regional_acts = np.array([conv[region] for conv in all_activations]).astype(np.float32)
            projected = regional_acts @ components.T  # [n_conv, n_components]
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

    # Balance dataset based on user emotions
    if max_samples_per_emotion is None:
        emotion_counts = [np.sum(user_labels == i) for i in range(len(emotions))]
        max_samples_per_emotion = min(emotion_counts)
        print(f"\nAuto-balancing to {max_samples_per_emotion} samples per class")

    balanced_acts = []
    balanced_user_labels = []
    balanced_asst_labels = []

    for i, emotion in enumerate(emotions):
        mask = user_labels == i
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

    return activations, user_labels, asst_labels, emotions


def train_epoch(model, dataloader, optimizer, ortho_weight, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_ortho_loss = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        activations = batch['activation'].to(device)
        user_labels = batch['label'][0].to(device)  # [batch_size]
        asst_labels = batch['label'][1].to(device)  # [batch_size]

        optimizer.zero_grad()

        # Forward pass for both user and assistant
        user_preds = model(activations, probe_type='user')
        asst_preds = model(activations, probe_type='assistant')

        # Task loss
        user_loss = F.cross_entropy(user_preds, user_labels)
        asst_loss = F.cross_entropy(asst_preds, asst_labels)
        task_loss = user_loss + asst_loss

        # Orthogonality loss
        ortho_loss = model.orthogonality_loss()

        # Total loss
        loss = task_loss + ortho_weight * ortho_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_ortho_loss += ortho_loss.item()

    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'ortho_loss': total_ortho_loss / n_batches,
    }


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_user_preds = []
    all_user_labels = []
    all_asst_preds = []
    all_asst_labels = []

    # For cross-accuracy
    user_on_asst_correct = 0
    asst_on_user_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            activations = batch['activation'].to(device)
            user_labels = batch['label'][0].to(device)
            asst_labels = batch['label'][1].to(device)

            # Forward pass
            user_preds = model(activations, probe_type='user')
            asst_preds = model(activations, probe_type='assistant')

            # Task loss
            user_loss = F.cross_entropy(user_preds, user_labels)
            asst_loss = F.cross_entropy(asst_preds, asst_labels)
            total_loss += (user_loss + asst_loss).item()

            # Store predictions
            all_user_preds.append(user_preds.cpu())
            all_user_labels.append(user_labels.cpu())
            all_asst_preds.append(asst_preds.cpu())
            all_asst_labels.append(asst_labels.cpu())

            # Cross-accuracy
            user_on_asst_correct += (user_preds.argmax(dim=1) == asst_labels).sum().item()
            asst_on_user_correct += (asst_preds.argmax(dim=1) == user_labels).sum().item()
            total_samples += len(activations)

    # Compute metrics
    user_preds = torch.cat(all_user_preds)
    user_labels = torch.cat(all_user_labels)
    asst_preds = torch.cat(all_asst_preds)
    asst_labels = torch.cat(all_asst_labels)

    user_acc = (user_preds.argmax(dim=1) == user_labels).float().mean().item()
    asst_acc = (asst_preds.argmax(dim=1) == asst_labels).float().mean().item()

    metrics = {
        'val_loss': total_loss / len(dataloader),
        'user_accuracy': user_acc,
        'asst_accuracy': asst_acc,
        'user_on_asst_accuracy': user_on_asst_correct / total_samples,
        'asst_on_user_accuracy': asst_on_user_correct / total_samples,
        'ortho_loss': model.orthogonality_loss().item(),
    }

    return metrics


def compute_orthogonality_metrics(model):
    """Compute detailed orthogonality metrics"""
    user_probes = model.get_normalized_probes('user')
    asst_probes = model.get_normalized_probes('assistant')

    # Cross-group dot products (all pairs)
    cross_dots = user_probes @ asst_probes.T  # [n_emotions, n_emotions]

    metrics = {
        'cross_dots_mean': np.abs(cross_dots).mean(),
        'cross_dots_max': np.abs(cross_dots).max(),
        'cross_dots_std': np.abs(cross_dots).std(),
        'cross_dots_matrix': cross_dots.tolist(),
    }

    return metrics


def collate_fn(batch):
    """Custom collate function to handle dual labels"""
    activations = torch.stack([item['activation'] for item in batch])
    user_labels = torch.tensor([item['label'][0] for item in batch])
    asst_labels = torch.tensor([item['label'][1] for item in batch])

    return {
        'activation': activations,
        'label': (user_labels, asst_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Train orthogonal conversation emotion probes")
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace-vast/annas/git/research-tools/data/activations/conversations2_combined.h5",
        help="Path to conversations2_combined.h5",
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer to train probe on")
    parser.add_argument(
        "--representation",
        type=str,
        choices=["raw", "global_cpca", "regional_cpca"],
        default="raw",
        help="Representation type",
    )
    parser.add_argument("--n-components", type=int, help="Number of cPCA components")
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
        help="Base directory for regional cPCA results",
    )
    parser.add_argument("--max-samples", type=int, help="Max samples per emotion")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max epochs")
    parser.add_argument("--ortho-weight", type=float, default=1.0, help="Orthogonality loss weight")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes_orthogonal",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare regional cPCA paths if needed
    regional_cpca_paths = None
    if args.representation == "regional_cpca":
        base_dir = Path(args.regional_cpca_dir)
        regional_cpca_paths = {
            "user": base_dir / "regional_user/google/gemma-3-27b-it_cpca.npz",
            "asst": base_dir / "regional_asst/google/gemma-3-27b-it_cpca.npz",
            "special1": base_dir / "regional_special1/google/gemma-3-27b-it_cpca.npz",
            "special2": base_dir / "regional_special2/google/gemma-3-27b-it_cpca.npz",
        }

    # Load data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    activations, user_labels, asst_labels, label_names = load_conversation_data(
        Path(args.data),
        args.layer,
        representation=args.representation,
        global_cpca_path=Path(args.global_cpca) if args.representation == "global_cpca" else None,
        regional_cpca_paths=regional_cpca_paths,
        n_components=args.n_components,
        max_samples_per_emotion=args.max_samples,
    )

    # Convert to tensors
    activations = torch.from_numpy(activations).float()

    # Create dual-label dataset (each sample has both user and asst labels)
    dual_labels = list(zip(user_labels, asst_labels))

    # Train/test split (stratify by user labels)
    train_idx, test_idx = train_test_split(
        np.arange(len(activations)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=user_labels,
    )

    train_dataset = EmotionDataset(
        activations[train_idx],
        [dual_labels[i] for i in train_idx],
        ['dual'] * len(train_idx)  # Dummy speaker type
    )
    test_dataset = EmotionDataset(
        activations[test_idx],
        [dual_labels[i] for i in test_idx],
        ['dual'] * len(test_idx)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    print("\n" + "=" * 80)
    print("TRAINING ORTHOGONAL PROBES")
    print("=" * 80)
    print(f"Orthogonality weight: {args.ortho_weight}")

    hidden_dim = activations.shape[1]
    model = OrthogonalEmotionProbes(hidden_dim=hidden_dim, n_emotions=len(EMOTIONS))
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    history = []

    for epoch in range(args.max_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args.ortho_weight, args.device)
        val_metrics = evaluate(model, test_loader, args.device)
        ortho_metrics = compute_orthogonality_metrics(model)

        # Combine metrics
        metrics = {
            'epoch': epoch,
            **train_metrics,
            **val_metrics,
            **ortho_metrics,
        }
        history.append(metrics)

        # Print progress
        if epoch % 10 == 0 or epoch == args.max_epochs - 1:
            print(f"Epoch {epoch:3d} | "
                  f"Train: {train_metrics['total_loss']:.4f} "
                  f"(task={train_metrics['task_loss']:.4f}, ortho={train_metrics['ortho_loss']:.4f}) | "
                  f"Val: {val_metrics['val_loss']:.4f} | "
                  f"User: {val_metrics['user_accuracy']:.4f} | "
                  f"Asst: {val_metrics['asst_accuracy']:.4f} | "
                  f"Cross-dots: {ortho_metrics['cross_dots_mean']:.4f}")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_epoch = epoch

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    rep_str = args.representation
    n_comp_str = f"_top{args.n_components}" if args.n_components else ""
    ortho_str = f"_ortho{args.ortho_weight}"
    filename = f"probe_layer{args.layer}_{rep_str}{n_comp_str}{ortho_str}.pkl"
    output_path = output_dir / filename

    results = {
        'args': vars(args),
        'history': history,
        'best_epoch': best_epoch,
        'final_user_probes': model.get_normalized_probes('user'),
        'final_asst_probes': model.get_normalized_probes('assistant'),
        'final_ortho_metrics': ortho_metrics,
        'final_val_metrics': val_metrics,
        'label_names': label_names,
        'layer': args.layer,
        'representation': args.representation,
        'n_components': args.n_components,
        'ortho_weight': args.ortho_weight,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"User accuracy: {val_metrics['user_accuracy']:.4f}")
    print(f"Assistant accuracy: {val_metrics['asst_accuracy']:.4f}")
    print(f"User→Asst cross-accuracy: {val_metrics['user_on_asst_accuracy']:.4f}")
    print(f"Asst→User cross-accuracy: {val_metrics['asst_on_user_accuracy']:.4f}")
    print(f"\nOrthogonality metrics:")
    print(f"  Mean |cross-dot|: {ortho_metrics['cross_dots_mean']:.4f}")
    print(f"  Max |cross-dot|: {ortho_metrics['cross_dots_max']:.4f}")
    print(f"  Std |cross-dot|: {ortho_metrics['cross_dots_std']:.4f}")
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
