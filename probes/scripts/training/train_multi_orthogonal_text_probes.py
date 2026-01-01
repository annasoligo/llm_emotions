#!/usr/bin/env python3
"""Train K orthogonal emotion probe sets on text data to discover intrinsic dimensionality.

This script discovers how many independent emotion subspaces exist in the representation
by training K orthogonal probe sets, where each set is constrained to be orthogonal to
all other sets.

Key differences from conversation probes:
- Not tied to user/assistant - these are general emotion probes
- K is a hyperparameter (can search for optimal K)
- All K sets trained on the same labeled data
- Orthogonality enforced between ALL pairs of sets (O(K²) constraints)

Usage:
    # Train with fixed K
    python train_multi_orthogonal_text_probes.py --layer 30 --n-sets 3 --ortho-weight 10.0

    # Search for optimal K
    python train_multi_orthogonal_text_probes.py --layer 30 --search-k --max-sets 10
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from probes.core import load_cpca_results
from probes.output_config import get_output_path


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


class MultiOrthogonalEmotionProbes(nn.Module):
    """K orthogonal emotion probe sets with all-pairs orthogonality constraints."""

    def __init__(self, hidden_dim: int, n_emotions: int = 6, n_sets: int = 2):
        """Initialize K orthogonal probe sets.

        Args:
            hidden_dim: Dimensionality of activations
            n_emotions: Number of emotion classes (default: 6)
            n_sets: Number of independent probe sets (K)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_emotions = n_emotions
        self.n_sets = n_sets

        # All probe sets: [K, n_emotions, hidden_dim]
        self.probe_sets = nn.Parameter(torch.randn(n_sets, n_emotions, hidden_dim))

        # Initialize with small values for stability, then normalize
        for i in range(n_sets):
            nn.init.xavier_normal_(self.probe_sets[i], gain=0.1)

        # Initialize to unit vectors to start
        with torch.no_grad():
            self.probe_sets.data = F.normalize(self.probe_sets.data, dim=2)

    def forward(self, activations: torch.Tensor, set_idx: Optional[int] = None) -> torch.Tensor:
        """Project activations onto probe directions.

        Args:
            activations: [batch, hidden_dim] activation vectors
            set_idx: Which probe set to use (0 to K-1). If None, returns all sets.

        Returns:
            If set_idx is None: [batch, K, n_emotions] logits for all sets
            If set_idx is int: [batch, n_emotions] logits for that set
        """
        if set_idx is not None:
            # Single set - use normalized probes directly
            probes = self.probe_sets[set_idx]  # [n_emotions, hidden_dim]
            probes_norm = F.normalize(probes, dim=1)
            return activations @ probes_norm.T  # [batch, n_emotions]
        else:
            # All sets
            all_logits = []
            for i in range(self.n_sets):
                probes = self.probe_sets[i]
                probes_norm = F.normalize(probes, dim=1)
                logits = activations @ probes_norm.T  # [batch, n_emotions]
                all_logits.append(logits)
            return torch.stack(all_logits, dim=1)  # [batch, K, n_emotions]

    def project_gradients_to_tangent_space(self):
        """Project gradients onto tangent space of unit sphere.

        For each probe vector v with gradient g, the tangent space projection is:
        g_tangent = g - (g · v)v

        This ensures gradients don't change the norm, only the direction.
        Should be called after backward() but before optimizer.step().
        """
        if self.probe_sets.grad is None:
            return

        with torch.no_grad():
            # Normalize probe vectors
            probes_norm = F.normalize(self.probe_sets, dim=2)  # [K, n_emotions, hidden_dim]

            # Compute dot product between gradients and normalized probes
            # Shape: [K, n_emotions, 1]
            grad_dot_probe = (self.probe_sets.grad * probes_norm).sum(dim=2, keepdim=True)

            # Project gradient onto tangent space: g - (g·v)v
            self.probe_sets.grad = self.probe_sets.grad - grad_dot_probe * probes_norm

    def gram_schmidt_orthogonalize(self):
        """Apply Gram-Schmidt orthogonalization to all probe sets.

        This enforces strict orthogonality after each gradient step by:
        1. Flattening all K * n_emotions probe vectors
        2. Applying Gram-Schmidt to make them mutually orthogonal
        3. Reshaping back to [K, n_emotions, hidden_dim]

        Should be called after optimizer.step() and renormalization.
        """
        with torch.no_grad():
            # Reshape to [K * n_emotions, hidden_dim]
            n_total_probes = self.n_sets * self.n_emotions
            probes_flat = self.probe_sets.data.reshape(n_total_probes, self.hidden_dim)

            # Gram-Schmidt orthogonalization
            orthogonal_probes = torch.zeros_like(probes_flat)

            for i in range(n_total_probes):
                # Start with current vector
                vec = probes_flat[i].clone()

                # Subtract projections onto all previous orthogonal vectors
                for j in range(i):
                    # Project vec onto orthogonal_probes[j]
                    projection = torch.dot(vec, orthogonal_probes[j]) * orthogonal_probes[j]
                    vec = vec - projection

                # Normalize
                vec_norm = vec.norm()
                if vec_norm > 1e-8:  # Avoid division by zero
                    orthogonal_probes[i] = vec / vec_norm
                else:
                    # If vector became zero (linear dependent), use random orthogonal direction
                    orthogonal_probes[i] = torch.randn_like(vec)
                    for j in range(i):
                        projection = torch.dot(orthogonal_probes[i], orthogonal_probes[j]) * orthogonal_probes[j]
                        orthogonal_probes[i] = orthogonal_probes[i] - projection
                    orthogonal_probes[i] = F.normalize(orthogonal_probes[i], dim=0)

            # Reshape back to [K, n_emotions, hidden_dim]
            self.probe_sets.data = orthogonal_probes.reshape(self.n_sets, self.n_emotions, self.hidden_dim)

    def orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality loss between all pairs of probe sets.

        Returns:
            Mean squared dot product across all O(K²) set pairs and emotion pairs.
        """
        # Normalize all probe sets
        probe_sets_norm = self.probe_sets / self.probe_sets.norm(dim=2, keepdim=True)
        # Shape: [K, n_emotions, hidden_dim]

        total_loss = 0.0
        n_pairs = 0

        # For each pair of sets (i, j) where i < j
        for i in range(self.n_sets):
            for j in range(i + 1, self.n_sets):
                # Get probe sets
                set_i = probe_sets_norm[i]  # [n_emotions, hidden_dim]
                set_j = probe_sets_norm[j]  # [n_emotions, hidden_dim]

                # Compute cross-emotion dot products between sets
                cross_dots = set_i @ set_j.T  # [n_emotions, n_emotions]

                # Penalize all dot products
                total_loss += (cross_dots ** 2).mean()
                n_pairs += 1

        # Average over all pairs
        # Ensure we return a tensor (even when K=1 and total_loss=0)
        if n_pairs == 0:
            return torch.tensor(0.0, device=self.probe_sets.device)
        return total_loss / n_pairs

    def get_normalized_probes(self, set_idx: int) -> np.ndarray:
        """Get normalized probe vectors for a specific set.

        Args:
            set_idx: Which probe set (0 to K-1)

        Returns:
            Normalized probe vectors [n_emotions, hidden_dim]
        """
        probes = self.probe_sets[set_idx]
        probes_norm = probes / probes.norm(dim=1, keepdim=True)
        return probes_norm.detach().cpu().numpy()

    def get_all_probes(self) -> np.ndarray:
        """Get all normalized probe sets.

        Returns:
            All probe sets [K, n_emotions, hidden_dim]
        """
        probe_sets_norm = self.probe_sets / self.probe_sets.norm(dim=2, keepdim=True)
        return probe_sets_norm.detach().cpu().numpy()


class EmotionDataset(Dataset):
    """Dataset of emotion-labeled text activations."""

    def __init__(self, activations: np.ndarray, labels: np.ndarray):
        """
        Args:
            activations: [n_samples, hidden_dim] tensor
            labels: [n_samples] integer labels (0-5 for 6 emotions)
        """
        self.activations = torch.from_numpy(activations).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'activation': self.activations[idx],
            'label': self.labels[idx],
        }


def load_text_data(
    h5_path: Path,
    layer: int,
    cpca_path: Optional[Path] = None,
    n_components: Optional[int] = None,
    max_samples_per_emotion: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load text activations and labels.

    Returns:
        activations: [n_samples, feature_dim]
        labels: [n_samples] integer labels
        label_names: List of emotion names
    """
    print(f"Loading text data from {h5_path}")
    print(f"Layer: {layer}")

    emotions = EMOTIONS
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    with h5py.File(h5_path, "r") as f:
        all_activations = []
        all_labels = []

        activations_group = f["activations"]

        # Iterate through all keys that contain emotion variants
        for key in activations_group.keys():
            # Keys are like: batch1_set_0_direct_address_anger
            parts = key.split('_')
            if len(parts) < 4:
                continue

            # Extract emotion from the end of the key
            emotion_part = parts[-1]

            # Map to standard emotion names if needed
            if emotion_part in emotion_to_idx:
                emotion = emotion_part
            else:
                continue

            # Load activations for this text
            text_group = activations_group[key]

            if "emotional" not in text_group:
                continue

            # Shape is [n_layers, hidden_dim] where layer index = layer number
            emotional_data = text_group["emotional"][:]

            if layer >= len(emotional_data):
                continue

            # Get activation for the specified layer
            layer_act = emotional_data[layer]  # [hidden_dim]

            all_activations.append(layer_act)
            all_labels.append(emotion_to_idx[emotion])

        activations = np.stack(all_activations).astype(np.float32)  # [n_samples, hidden_dim]
        labels = np.array(all_labels, dtype=np.int64)

    print(f"Loaded {len(activations)} samples")
    print(f"Activation shape: {activations.shape}")

    # Apply cPCA projection if needed
    if cpca_path is not None:
        print(f"\nProjecting onto cPCA (top {n_components} components)...")
        cpca_data = load_cpca_results(cpca_path)
        components = cpca_data["components"][layer]

        if n_components:
            components = components[:n_components]

        print(f"Components shape: {components.shape}")
        activations = activations @ components.T
        print(f"Projected shape: {activations.shape}")

    # Balance dataset
    if max_samples_per_emotion is None:
        emotion_counts = [np.sum(labels == i) for i in range(len(emotions))]
        max_samples_per_emotion = min(emotion_counts)
        print(f"\nAuto-balancing to {max_samples_per_emotion} samples per class")

    balanced_acts = []
    balanced_labels = []

    for i, emotion in enumerate(emotions):
        mask = labels == i
        emotion_acts = activations[mask]
        emotion_labels = labels[mask]

        if len(emotion_acts) > max_samples_per_emotion:
            indices = np.random.choice(
                len(emotion_acts), max_samples_per_emotion, replace=False
            )
            emotion_acts = emotion_acts[indices]
            emotion_labels = emotion_labels[indices]

        balanced_acts.append(emotion_acts)
        balanced_labels.append(emotion_labels)

    activations = np.vstack(balanced_acts)
    labels = np.concatenate(balanced_labels)

    print(f"Balanced dataset: {len(activations)} samples")

    return activations, labels, emotions


def train_epoch(
    model: MultiOrthogonalEmotionProbes,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ortho_weight: float,
    device: str,
    use_gram_schmidt: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_ortho_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        activations = batch['activation'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        # Forward pass for all K sets
        all_logits = model(activations)  # [batch, K, n_emotions]

        # Task loss: average cross-entropy across all K sets
        task_loss = 0.0
        for k in range(model.n_sets):
            logits_k = all_logits[:, k, :]  # [batch, n_emotions]
            task_loss += F.cross_entropy(logits_k, labels)
        task_loss = task_loss / model.n_sets

        # Orthogonality loss
        ortho_loss = model.orthogonality_loss()

        # Total loss
        loss = task_loss + ortho_weight * ortho_loss

        loss.backward()

        # Project gradients to tangent space to maintain unit norm
        model.project_gradients_to_tangent_space()

        optimizer.step()

        # Renormalize probes after step to ensure they stay on unit sphere
        with torch.no_grad():
            model.probe_sets.data = F.normalize(model.probe_sets.data, dim=2)

        # Apply Gram-Schmidt orthogonalization if requested
        if use_gram_schmidt:
            model.gram_schmidt_orthogonalize()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_ortho_loss += ortho_loss.item()

    n_batches = len(dataloader)
    return {
        'total_loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'ortho_loss': total_ortho_loss / n_batches,
    }


def evaluate(
    model: MultiOrthogonalEmotionProbes,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()

    # Per-set metrics
    set_correct = [0] * model.n_sets
    total_samples = 0

    # Cross-set agreement metrics
    cross_agreements = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            activations = batch['activation'].to(device)
            labels = batch['label'].to(device)

            # Get predictions from all sets
            all_logits = model(activations)  # [batch, K, n_emotions]
            all_preds = all_logits.argmax(dim=2)  # [batch, K]

            # Accuracy per set
            for k in range(model.n_sets):
                set_correct[k] += (all_preds[:, k] == labels).sum().item()

            # Cross-set agreement (should be low if truly orthogonal/independent)
            for i in range(model.n_sets):
                for j in range(i + 1, model.n_sets):
                    agreement = (all_preds[:, i] == all_preds[:, j]).float().mean().item()
                    cross_agreements.append(agreement)

            total_samples += len(labels)

    # Compute metrics
    accuracies = [correct / total_samples for correct in set_correct]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    mean_cross_agreement = np.mean(cross_agreements) if cross_agreements else 0.0

    metrics = {
        'accuracies': accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'mean_cross_agreement': mean_cross_agreement,
        'ortho_loss': model.orthogonality_loss().item(),
    }

    return metrics


def compute_orthogonality_metrics(model: MultiOrthogonalEmotionProbes) -> Dict:
    """Compute detailed orthogonality metrics."""
    probe_sets = model.get_all_probes()  # [K, n_emotions, hidden_dim]

    all_dots = []

    # Compute dot products between all pairs of sets
    for i in range(model.n_sets):
        for j in range(i + 1, model.n_sets):
            # Cross-emotion dot products
            cross_dots = probe_sets[i] @ probe_sets[j].T  # [n_emotions, n_emotions]
            all_dots.extend(np.abs(cross_dots).flatten())

    # Handle case when K=1 (no pairs)
    if len(all_dots) == 0:
        metrics = {
            'cross_dots_mean': 0.0,
            'cross_dots_max': 0.0,
            'cross_dots_std': 0.0,
            'cross_dots_median': 0.0,
            'n_pairs': 0,
        }
    else:
        metrics = {
            'cross_dots_mean': np.mean(all_dots),
            'cross_dots_max': np.max(all_dots),
            'cross_dots_std': np.std(all_dots),
            'cross_dots_median': np.median(all_dots),
            'n_pairs': len(all_dots),
        }

    return metrics


def train_single_config(
    activations: np.ndarray,
    labels: np.ndarray,
    n_sets: int,
    ortho_weight: float,
    args: argparse.Namespace,
) -> Dict:
    """Train probes for a single configuration.

    Returns:
        Dictionary with model, metrics, and convergence info
    """
    print(f"\n{'='*80}")
    print(f"Training with K={n_sets} probe sets, ortho_weight={ortho_weight}")
    print(f"{'='*80}")

    # Train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(activations)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    train_dataset = EmotionDataset(activations[train_idx], labels[train_idx])
    test_dataset = EmotionDataset(activations[test_idx], labels[test_idx])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    hidden_dim = activations.shape[1]
    model = MultiOrthogonalEmotionProbes(
        hidden_dim=hidden_dim,
        n_emotions=len(EMOTIONS),
        n_sets=n_sets,
    )
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_mean_acc = 0.0
    best_epoch = 0
    best_model_state = None
    history = []
    epochs_without_improvement = 0

    for epoch in range(args.max_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer, ortho_weight, args.device, args.gram_schmidt
        )
        val_metrics = evaluate(model, test_loader, args.device)
        ortho_metrics = compute_orthogonality_metrics(model)

        # Combine metrics
        metrics = {
            'epoch': epoch,
            'n_sets': n_sets,
            **train_metrics,
            **val_metrics,
            **ortho_metrics,
        }
        history.append(metrics)

        # Print progress
        if epoch % 10 == 0 or epoch == args.max_epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {train_metrics['total_loss']:.4f} "
                f"(task={train_metrics['task_loss']:.4f}, "
                f"ortho={train_metrics['ortho_loss']:.4f}) | "
                f"Mean Acc: {val_metrics['mean_accuracy']:.4f} ± {val_metrics['std_accuracy']:.4f} | "
                f"Cross-dots: {ortho_metrics['cross_dots_mean']:.4f}"
            )

        # Early stopping based on mean accuracy
        if val_metrics['mean_accuracy'] > best_mean_acc:
            best_mean_acc = val_metrics['mean_accuracy']
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(args.device)

    # Final evaluation
    final_val_metrics = evaluate(model, test_loader, args.device)
    final_ortho_metrics = compute_orthogonality_metrics(model)

    # Check convergence quality
    converged = (
        final_ortho_metrics['cross_dots_mean'] < args.convergence_threshold and
        final_val_metrics['mean_accuracy'] > args.min_accuracy_threshold
    )

    results = {
        'model': model,
        'history': history,
        'best_epoch': best_epoch,
        'n_sets': n_sets,
        'ortho_weight': ortho_weight,
        'final_metrics': {**final_val_metrics, **final_ortho_metrics},
        'converged': converged,
        'all_probe_sets': model.get_all_probes(),
    }

    print(f"\nFinal Results:")
    print(f"  Mean accuracy: {final_val_metrics['mean_accuracy']:.4f} ± {final_val_metrics['std_accuracy']:.4f}")
    print(f"  Accuracy range: [{final_val_metrics['min_accuracy']:.4f}, {final_val_metrics['max_accuracy']:.4f}]")
    print(f"  Cross-set agreement: {final_val_metrics['mean_cross_agreement']:.4f}")
    print(f"  Orthogonality (mean |dot|): {final_ortho_metrics['cross_dots_mean']:.4f}")
    print(f"  Converged: {converged}")

    return results


def search_optimal_k(
    activations: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict]:
    """Search for optimal K by incrementally testing K=1,2,3,...,max_sets.

    Stops when:
    - Accuracy degrades significantly
    - Orthogonality loss won't converge
    - max_sets is reached

    Returns:
        List of results for each K tested
    """
    print(f"\n{'='*80}")
    print(f"SEARCHING FOR OPTIMAL K (max_sets={args.max_sets})")
    print(f"{'='*80}")

    all_results = []
    prev_mean_acc = None

    for k in range(1, args.max_sets + 1):
        # Train with K sets
        results = train_single_config(
            activations, labels, n_sets=k, ortho_weight=args.ortho_weight, args=args
        )

        all_results.append(results)

        current_mean_acc = results['final_metrics']['mean_accuracy']
        converged = results['converged']

        # Check stopping criteria
        if not converged:
            print(f"\n⚠ K={k} did not converge. Stopping search.")
            break

        if prev_mean_acc is not None:
            acc_degradation = prev_mean_acc - current_mean_acc
            if acc_degradation > args.accuracy_degradation_threshold:
                print(
                    f"\n⚠ K={k} shows significant accuracy degradation "
                    f"({acc_degradation:.4f}). Stopping search."
                )
                break

        prev_mean_acc = current_mean_acc

        print(f"\n✓ K={k} converged successfully with mean accuracy {current_mean_acc:.4f}")

    # Summary
    print(f"\n{'='*80}")
    print("SEARCH SUMMARY")
    print(f"{'='*80}")
    for i, res in enumerate(all_results):
        k = res['n_sets']
        metrics = res['final_metrics']
        status = "✓ converged" if res['converged'] else "✗ failed"
        print(
            f"K={k}: mean_acc={metrics['mean_accuracy']:.4f}, "
            f"ortho={metrics['cross_dots_mean']:.4f} - {status}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple orthogonal emotion probe sets on text data"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/data/activations/texts_combined.h5",
        help="Path to text activations HDF5 file",
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer to train on")
    parser.add_argument(
        "--cpca-path",
        type=str,
        help="Optional path to cPCA results for dimensionality reduction",
    )
    parser.add_argument(
        "--n-components", type=int, help="Number of cPCA components to use"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Max samples per emotion (for balancing)"
    )

    # Model architecture
    parser.add_argument(
        "--n-sets",
        type=int,
        default=2,
        help="Number of orthogonal probe sets K (ignored if --search-k)",
    )
    parser.add_argument(
        "--search-k",
        action="store_true",
        help="Search for optimal K instead of using fixed --n-sets",
    )
    parser.add_argument(
        "--max-sets",
        type=int,
        default=10,
        help="Maximum K to test in search mode",
    )

    # Training
    parser.add_argument(
        "--ortho-weight",
        type=float,
        default=1000.0,
        help="Weight for orthogonality loss",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gram-schmidt",
        action="store_true",
        help="Apply Gram-Schmidt re-orthogonalization after each gradient step"
    )

    # Convergence criteria
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.001,
        help="Max mean |cross-dot| for convergence",
    )
    parser.add_argument(
        "--min-accuracy-threshold",
        type=float,
        default=0.25,
        help="Minimum mean accuracy to consider converged",
    )
    parser.add_argument(
        "--accuracy-degradation-threshold",
        type=float,
        default=0.05,
        help="Max accuracy drop between K and K+1 before stopping search",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (auto-determined if not specified)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    print("="*80)
    print("LOADING DATA")
    print("="*80)

    data_path = Path(args.data)
    if not data_path.is_absolute() and not str(data_path).startswith("outputs/"):
        base_data = Path(get_output_path("data"))
        data_path = base_data / "activations" / args.data.split('/')[-1]

    cpca_path = None
    if args.cpca_path:
        cpca_path = Path(args.cpca_path)
        if not cpca_path.is_absolute():
            base_cpca = Path(get_output_path("dimensionality_reduction"))
            cpca_path = base_cpca / "cpca" / "text_based" / args.cpca_path.split('/')[-1]

    activations, labels, label_names = load_text_data(
        data_path,
        args.layer,
        cpca_path=cpca_path,
        n_components=args.n_components,
        max_samples_per_emotion=args.max_samples,
    )

    # Train
    if args.search_k:
        # Search for optimal K
        all_results = search_optimal_k(activations, labels, args)

        # Save all results
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            base_output = Path(get_output_path("probes"))
            output_dir = base_output / "emotion_probes" / "text_based" / "multi_orthogonal"
        output_dir.mkdir(parents=True, exist_ok=True)

        search_filename = f"search_layer{args.layer}_ortho{args.ortho_weight}_seed{args.seed}.pkl"
        search_path = output_dir / search_filename

        # Save without torch models (too large)
        save_results = []
        for res in all_results:
            save_results.append({
                'n_sets': res['n_sets'],
                'ortho_weight': res['ortho_weight'],
                'converged': res['converged'],
                'best_epoch': res['best_epoch'],
                'final_metrics': res['final_metrics'],
                'history': res['history'],
                'all_probe_sets': res['all_probe_sets'],
            })

        with open(search_path, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'label_names': label_names,
                'layer': args.layer,
                'all_results': save_results,
            }, f)

        print(f"\nSaved search results to {search_path}")

        # Also save best model
        best_k = None
        best_acc = 0.0
        for i, res in enumerate(all_results):
            if res['converged'] and res['final_metrics']['mean_accuracy'] > best_acc:
                best_k = i
                best_acc = res['final_metrics']['mean_accuracy']

        if best_k is not None:
            best_res = all_results[best_k]
            best_filename = f"best_k{best_res['n_sets']}_layer{args.layer}_ortho{args.ortho_weight}.pkl"
            best_path = output_dir / best_filename

            with open(best_path, 'wb') as f:
                pickle.dump({
                    'args': vars(args),
                    'label_names': label_names,
                    'layer': args.layer,
                    'n_sets': best_res['n_sets'],
                    'model_state_dict': best_res['model'].state_dict(),
                    'final_metrics': best_res['final_metrics'],
                    'history': best_res['history'],
                    'all_probe_sets': best_res['all_probe_sets'],
                }, f)

            print(f"Saved best model (K={best_res['n_sets']}) to {best_path}")

    else:
        # Train single configuration
        results = train_single_config(
            activations, labels, n_sets=args.n_sets, ortho_weight=args.ortho_weight, args=args
        )

        # Save results
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            base_output = Path(get_output_path("probes"))
            output_dir = base_output / "emotion_probes" / "text_based" / "multi_orthogonal"
        output_dir.mkdir(parents=True, exist_ok=True)

        nc_str = f"_nc{args.n_components}" if args.n_components else ""
        filename = f"probe_k{args.n_sets}_layer{args.layer}{nc_str}_ortho{args.ortho_weight}.pkl"
        output_path = output_dir / filename

        with open(output_path, 'wb') as f:
            pickle.dump({
                'args': vars(args),
                'label_names': label_names,
                'layer': args.layer,
                'n_sets': results['n_sets'],
                'model_state_dict': results['model'].state_dict(),
                'final_metrics': results['final_metrics'],
                'history': results['history'],
                'all_probe_sets': results['all_probe_sets'],
            }, f)

        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
