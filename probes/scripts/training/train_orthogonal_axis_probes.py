#!/usr/bin/env python3
"""Train soft orthogonal probes for emotion axes using ordinal regression.

Trains 4 separate probe directions (one per axis) that output continuous scores
along each emotion axis, with soft orthogonality constraints. Uses ordinal regression
to respect the ordering: low < neutral < high.
"""

import argparse
import json
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple
import time

# Add data module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))
from constants import AXES


class OrdinalProbe(nn.Module):
    """Single ordinal regression probe with learnable ordered thresholds."""

    def __init__(self, hidden_dim: int, n_classes: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # Linear projection to get score
        self.linear = nn.Linear(hidden_dim, 1)

        # Parameterize thresholds to guarantee ordering: θ₀ < θ₁ < ...
        # Base threshold + cumulative positive deltas
        # Initialize with reasonable spacing
        self.threshold_base = nn.Parameter(torch.tensor(-1.0))  # Start lower
        self.threshold_deltas = nn.Parameter(torch.ones(n_classes - 2) * 0.5)  # Initial spacing ~1.0 apart

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0)  # Larger gain
        nn.init.zeros_(self.linear.bias)

    def get_thresholds(self) -> torch.Tensor:
        """Get ordered thresholds: θ₀ < θ₁ < ... < θ_{K-2}"""
        # First threshold is the base
        # Subsequent thresholds = previous + softplus(delta)
        thresholds = [self.threshold_base]
        for delta in self.threshold_deltas:
            thresholds.append(thresholds[-1] + F.softplus(delta))
        return torch.stack(thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ordinal log probabilities.

        Args:
            x: [batch_size, hidden_dim]

        Returns:
            log_probs: [batch_size, n_classes] log probabilities
        """
        # Get continuous score
        score = self.linear(x).squeeze(-1)  # [batch_size]

        # Get ordered thresholds
        thresholds = self.get_thresholds()  # [n_classes - 1]

        # Compute cumulative probabilities: P(y <= k) = sigmoid(θ_k - score)
        # Shape: [batch_size, n_classes-1]
        cum_probs = torch.sigmoid(thresholds - score.unsqueeze(-1))

        # Convert to class probabilities using padding trick
        # P(y=0) = cum_prob_0
        # P(y=k) = cum_prob_k - cum_prob_{k-1}
        # P(y=K-1) = 1 - cum_prob_{K-2}

        # Pad with 0 on left, 1 on right for easy differencing
        padded = torch.cat([
            torch.zeros(score.shape[0], 1, device=score.device),
            cum_probs,
            torch.ones(score.shape[0], 1, device=score.device),
        ], dim=1)  # [batch_size, n_classes + 1]

        probs = padded[:, 1:] - padded[:, :-1]  # [batch_size, n_classes]

        # Small epsilon for numerical stability
        probs = probs.clamp(min=1e-7)

        return torch.log(probs)  # Return log probabilities for NLLLoss


class OrthogonalAxisProbes(nn.Module):
    """4 separate ordinal probes with soft orthogonality constraints."""

    def __init__(self, hidden_dim: int, n_classes: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_axes = 4

        # Create 4 separate ordinal probes
        self.valence_probe = OrdinalProbe(hidden_dim, n_classes)
        self.arousal_probe = OrdinalProbe(hidden_dim, n_classes)
        self.dominance_probe = OrdinalProbe(hidden_dim, n_classes)
        self.trust_probe = OrdinalProbe(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, axis: str) -> torch.Tensor:
        """Forward pass for specific axis.

        Args:
            x: [batch_size, hidden_dim] activations
            axis: Which axis probe to use

        Returns:
            log_probs: [batch_size, n_classes] log probabilities
        """
        if axis == 'valence':
            return self.valence_probe(x)
        elif axis == 'arousal':
            return self.arousal_probe(x)
        elif axis == 'dominance':
            return self.dominance_probe(x)
        elif axis == 'trust':
            return self.trust_probe(x)
        else:
            raise ValueError(f"Unknown axis: {axis}")

    def get_all_probes(self) -> List[OrdinalProbe]:
        """Get list of all probe modules."""
        return [
            self.valence_probe,
            self.arousal_probe,
            self.dominance_probe,
            self.trust_probe,
        ]

    def orthogonality_loss(self) -> torch.Tensor:
        """Compute soft orthogonality loss between all probe pairs.

        Returns:
            Mean squared cosine similarity between all pairs of normalized probe weights
        """
        probes = self.get_all_probes()

        # Get normalized weight vectors from the linear layers
        weights = [F.normalize(p.linear.weight.squeeze(0), dim=0) for p in probes]

        # Compute pairwise similarities
        total_loss = 0.0
        n_pairs = 0

        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                # Dot product of normalized vectors
                dot = torch.dot(weights[i], weights[j])
                # Penalize squared dot product
                total_loss += dot ** 2
                n_pairs += 1

        return total_loss / n_pairs if n_pairs > 0 else torch.tensor(0.0)


def load_data(
    h5_file: Path,
    layer_idx: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Load activations and labels from HDF5 file.

    Args:
        h5_file: Path to HDF5 file
        layer_idx: Which layer to use (0-indexed within saved layers)

    Returns:
        activations: [n_samples, hidden_dim]
        labels_by_axis: Dict mapping axis name to [n_samples] ordinal labels (0=low, 1=neutral, 2=high)
        sample_ids: List of sample IDs
    """
    with h5py.File(h5_file, 'r') as f:
        metadata = json.loads(f.attrs['metadata'])

        activations_list = []
        labels_dict = {axis: [] for axis in AXES}
        sample_ids = []

        for sample_id in f['activations'].keys():
            # Load activation for this layer
            acts = f['activations'][sample_id][layer_idx]  # [hidden_dim]
            activations_list.append(acts)

            # Get labels for each axis
            meta = metadata[sample_id]

            # Map level strings to ordinal integers
            level_map = {'low': 0, 'neutral': 1, 'high': 2}

            for axis in AXES:
                level = meta[axis]
                labels_dict[axis].append(level_map[level])

            sample_ids.append(sample_id)

        # Stack into arrays
        activations = np.stack(activations_list, axis=0)
        labels_by_axis = {
            axis: np.array(labels, dtype=np.int64)
            for axis, labels in labels_dict.items()
        }

    return activations, labels_by_axis, sample_ids


def train_epoch(
    model: OrthogonalAxisProbes,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    ortho_weight: float,
    device: str,
) -> Tuple[float, float, Dict[str, float]]:
    """Train for one epoch.

    Returns:
        avg_total_loss, avg_ortho_loss, accuracies_by_axis
    """
    model.train()

    total_loss_sum = 0.0
    ortho_loss_sum = 0.0
    n_batches = 0

    # Track predictions for accuracy
    all_preds = {axis: [] for axis in AXES}
    all_labels = {axis: [] for axis in AXES}

    for batch in dataloader:
        # Unpack: [batch_size, hidden_dim], [batch_size] x 4
        x = batch[0].to(device)
        y_valence = batch[1].to(device)
        y_arousal = batch[2].to(device)
        y_dominance = batch[3].to(device)
        y_trust = batch[4].to(device)

        labels_dict = {
            'valence': y_valence,
            'arousal': y_arousal,
            'dominance': y_dominance,
            'trust': y_trust,
        }

        optimizer.zero_grad()

        # Compute ordinal classification loss for each axis
        classification_losses = []

        for axis in AXES:
            log_probs = model(x, axis)
            loss = criterion(log_probs, labels_dict[axis])
            classification_losses.append(loss)

            # Track predictions
            preds = log_probs.argmax(dim=1).cpu().numpy()
            all_preds[axis].extend(preds)
            all_labels[axis].extend(labels_dict[axis].cpu().numpy())

        # Average classification loss
        class_loss = torch.stack(classification_losses).mean()

        # Orthogonality loss
        ortho_loss = model.orthogonality_loss()

        # Total loss
        total_loss = class_loss + ortho_weight * ortho_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        ortho_loss_sum += ortho_loss.item()
        n_batches += 1

    # Compute accuracies
    accuracies = {
        axis: accuracy_score(all_labels[axis], all_preds[axis])
        for axis in AXES
    }

    avg_total_loss = total_loss_sum / n_batches
    avg_ortho_loss = ortho_loss_sum / n_batches

    return avg_total_loss, avg_ortho_loss, accuracies


def ordinal_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy allowing off-by-one errors."""
    return np.mean(np.abs(preds - labels) <= 1)


def evaluate(
    model: OrthogonalAxisProbes,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on test set.

    Returns:
        Dict with accuracies and ordinal_accuracies by axis
    """
    model.eval()

    all_preds = {axis: [] for axis in AXES}
    all_labels = {axis: [] for axis in AXES}

    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            y_valence = batch[1]
            y_arousal = batch[2]
            y_dominance = batch[3]
            y_trust = batch[4]

            labels_dict = {
                'valence': y_valence,
                'arousal': y_arousal,
                'dominance': y_dominance,
                'trust': y_trust,
            }

            for axis in AXES:
                log_probs = model(x, axis)
                preds = log_probs.argmax(dim=1).cpu().numpy()
                all_preds[axis].extend(preds)
                all_labels[axis].extend(labels_dict[axis].numpy())

    # Compute exact and ordinal accuracies
    metrics = {}
    for axis in AXES:
        preds_arr = np.array(all_preds[axis])
        labels_arr = np.array(all_labels[axis])
        metrics[f'{axis}_acc'] = accuracy_score(labels_arr, preds_arr)
        metrics[f'{axis}_ord_acc'] = ordinal_accuracy(preds_arr, labels_arr)

    return metrics


def compute_probe_orthogonality(model: OrthogonalAxisProbes) -> Dict[str, float]:
    """Compute pairwise cosine similarities between probe weights.

    Returns:
        Dict with mean, min, max cosine similarity
    """
    probes = model.get_all_probes()
    weights = [F.normalize(p.linear.weight.squeeze(0), dim=0).detach().cpu() for p in probes]

    similarities = []

    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            sim = torch.dot(weights[i], weights[j]).item()
            similarities.append(abs(sim))

    return {
        'mean': np.mean(similarities),
        'min': np.min(similarities),
        'max': np.max(similarities),
    }


def main():
    parser = argparse.ArgumentParser(description="Train soft orthogonal axis probes with ordinal regression")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="HDF5 file with activations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for trained probes",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Layer index to use (0-indexed within saved layers)",
    )
    parser.add_argument(
        "--ortho-weight",
        type=float,
        required=True,
        help="Orthogonality loss weight (0 = no orthogonality constraint)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="L2 regularization weight",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("TRAINING SOFT ORTHOGONAL AXIS PROBES (ORDINAL REGRESSION)")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Layer: {args.layer}")
    print(f"Orthogonality weight: {args.ortho_weight}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.n_epochs}")
    print(f"Patience: {args.patience}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print()

    # Load data
    print("Loading data...")
    activations, labels_by_axis, sample_ids = load_data(args.data, args.layer)

    print(f"✓ Loaded {len(activations)} samples")
    print(f"  Activation shape: {activations.shape}")
    print(f"  Hidden dim: {activations.shape[1]}")
    print()

    # Check class distribution
    print("Class distribution:")
    level_names = ['low', 'neutral', 'high']
    for axis in AXES:
        counts = np.bincount(labels_by_axis[axis], minlength=3)
        print(f"  {axis:20s}: {' / '.join(f'{level_names[i]}={counts[i]}' for i in range(3))}")
    print()

    # Train/test split (stratified by valence to ensure balance)
    X_train, X_test, idx_train, idx_test = train_test_split(
        activations,
        np.arange(len(activations)),
        test_size=args.test_size,
        stratify=labels_by_axis['valence'],  # Stratify on one axis
        random_state=args.seed,
    )

    # Split labels
    y_train = {axis: labels_by_axis[axis][idx_train] for axis in AXES}
    y_test = {axis: labels_by_axis[axis][idx_test] for axis in AXES}

    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    print()

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train['valence']),
        torch.LongTensor(y_train['arousal']),
        torch.LongTensor(y_train['dominance']),
        torch.LongTensor(y_train['trust']),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test['valence']),
        torch.LongTensor(y_test['arousal']),
        torch.LongTensor(y_test['dominance']),
        torch.LongTensor(y_test['trust']),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    hidden_dim = activations.shape[1]
    model = OrthogonalAxisProbes(hidden_dim=hidden_dim, n_classes=3)
    model = model.to(args.device)

    print("Model initialized:")
    print(f"  4 ordinal probes × {hidden_dim} dimensions → 3 ordered classes each")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.NLLLoss()

    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()

    best_avg_acc = 0.0
    patience_counter = 0
    history = []
    model_path = None

    for epoch in range(args.n_epochs):
        # Train
        train_loss, train_ortho, train_accs = train_epoch(
            model, train_loader, optimizer, criterion, args.ortho_weight, args.device
        )

        # Evaluate
        test_metrics = evaluate(model, test_loader, args.device)

        # Extract accuracies
        test_accs = {axis: test_metrics[f'{axis}_acc'] for axis in AXES}
        test_ord_accs = {axis: test_metrics[f'{axis}_ord_acc'] for axis in AXES}

        # Compute metrics
        avg_train_acc = np.mean(list(train_accs.values()))
        avg_test_acc = np.mean(list(test_accs.values()))
        avg_test_ord_acc = np.mean(list(test_ord_accs.values()))

        # Probe orthogonality
        ortho_metrics = compute_probe_orthogonality(model)

        # Print progress
        print(f"Epoch {epoch + 1:3d}/{args.n_epochs}")
        print(f"  Train loss: {train_loss:.4f} (ortho: {train_ortho:.4f})")
        print(f"  Train acc:  {avg_train_acc:.3f} (V:{train_accs['valence']:.3f} "
              f"A:{train_accs['arousal']:.3f} D:{train_accs['dominance']:.3f} "
              f"AA:{train_accs['trust']:.3f})")
        print(f"  Test acc:   {avg_test_acc:.3f} (V:{test_accs['valence']:.3f} "
              f"A:{test_accs['arousal']:.3f} D:{test_accs['dominance']:.3f} "
              f"AA:{test_accs['trust']:.3f})")
        print(f"  Ord acc:    {avg_test_ord_acc:.3f} (V:{test_ord_accs['valence']:.3f} "
              f"A:{test_ord_accs['arousal']:.3f} D:{test_ord_accs['dominance']:.3f} "
              f"AA:{test_ord_accs['trust']:.3f})")
        print(f"  Ortho:      mean={ortho_metrics['mean']:.4f} "
              f"min={ortho_metrics['min']:.4f} max={ortho_metrics['max']:.4f}")

        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_ortho_loss': train_ortho,
            'train_accuracies': train_accs,
            'test_accuracies': test_accs,
            'test_ordinal_accuracies': test_ord_accs,
            'avg_train_acc': avg_train_acc,
            'avg_test_acc': avg_test_acc,
            'avg_test_ord_acc': avg_test_ord_acc,
            'orthogonality': ortho_metrics,
        })

        # Early stopping
        if avg_test_acc > best_avg_acc:
            best_avg_acc = avg_test_acc
            patience_counter = 0

            # Save best model
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.output_dir / f"probe_layer{args.layer}_ortho{args.ortho_weight}_seed{args.seed}.pt"

            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'best_epoch': epoch + 1,
                'best_avg_acc': best_avg_acc,
                'test_accuracies': test_accs,
                'test_ordinal_accuracies': test_ord_accs,
            }, model_path)

            print(f"  → Best model saved (avg_acc={best_avg_acc:.3f})")

        else:
            patience_counter += 1

        print()

        # Early stopping
        if patience_counter >= args.patience:
            print(f"✓ Early stopping at epoch {epoch + 1}")
            break

    # Save history (convert numpy types to Python types for JSON serialization)
    def convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj

    history_serializable = convert_to_json_serializable(history)

    history_path = args.output_dir / f"history_layer{args.layer}_ortho{args.ortho_weight}_seed{args.seed}.json"
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best test accuracy: {best_avg_acc:.3f}")
    if model_path:
        print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print()


if __name__ == "__main__":
    main()
