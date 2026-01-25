#!/usr/bin/env python3
"""Train contrastive probes using paired neutral/emotional activations.

Uses activation differences: Δ = activation(paraphrase) - activation(neutral)
This isolates the emotional signal and removes content-specific variance.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add data module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))
from constants import AXES


@dataclass
class PairedSample:
    """A neutral statement and its emotional paraphrase."""
    neutral_id: str
    paraphrase_id: str
    delta_activation: np.ndarray  # paraphrase - neutral
    labels: Dict[str, int]  # axis -> {-1, 0, 1}


class ContrastiveProbe(nn.Module):
    """Probe that operates on activation differences."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """Predict score from activation difference."""
        return self.linear(delta).squeeze(-1)

    def get_steering_vector(self) -> np.ndarray:
        """Extract the steering direction."""
        return self.linear.weight.squeeze(0).detach().cpu().numpy()


class OrthogonalContrastiveProbes(nn.Module):
    """Four contrastive probes with soft orthogonality constraint."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.probes = nn.ModuleDict({
            axis: ContrastiveProbe(hidden_dim) for axis in AXES
        })

    def forward(self, delta: torch.Tensor, axis: str) -> torch.Tensor:
        return self.probes[axis](delta)

    def get_steering_vectors(self) -> Dict[str, np.ndarray]:
        return {axis: probe.get_steering_vector() for axis, probe in self.probes.items()}

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize correlation between probe directions."""
        weights = [
            F.normalize(probe.linear.weight.squeeze(0), dim=0)
            for probe in self.probes.values()
        ]

        loss = 0.0
        n_pairs = 0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                cos_sim = torch.dot(weights[i], weights[j])
                loss += cos_sim ** 2
                n_pairs += 1

        return loss / n_pairs if n_pairs > 0 else torch.tensor(0.0)


def load_paired_data(
    h5_file: Path,
    layer_idx: int,
) -> List[PairedSample]:
    """Load data as paired (neutral, paraphrase) samples.

    Uses paraphrases where all axes are 'neutral' as the baseline for each neutral_idx group.
    """

    with h5py.File(h5_file, 'r') as f:
        metadata = json.loads(f.attrs['metadata'])

        # Group samples by neutral_idx and find the neutral baseline for each group
        samples_by_neutral_idx = {}
        for sample_id, meta in metadata.items():
            if meta.get('is_original_neutral', False):
                continue  # Skip original neutrals

            neutral_idx = meta['neutral_idx']
            if neutral_idx not in samples_by_neutral_idx:
                samples_by_neutral_idx[neutral_idx] = []
            samples_by_neutral_idx[neutral_idx].append((sample_id, meta))

        print(f"Found {len(samples_by_neutral_idx)} neutral_idx groups")

        # Find neutral baseline for each group (where all axes are 'neutral')
        neutral_activations = {}
        for neutral_idx, samples in samples_by_neutral_idx.items():
            # Find a sample where all axes are neutral
            for sample_id, meta in samples:
                if all(meta[axis] == 'neutral' for axis in AXES):
                    neutral_activations[neutral_idx] = f['activations'][sample_id][layer_idx]
                    break

        print(f"Found {len(neutral_activations)} neutral baselines")

        # Now create paired samples
        paired_samples = []
        level_map = {'low': -1, 'neutral': 0, 'high': 1}
        skipped = 0

        for neutral_idx, samples in samples_by_neutral_idx.items():
            if neutral_idx not in neutral_activations:
                skipped += len(samples)
                continue

            neutral_act = neutral_activations[neutral_idx]

            for sample_id, meta in samples:
                # Skip samples that are all neutral (they're our baseline)
                if all(meta[axis] == 'neutral' for axis in AXES):
                    continue

                paraphrase_act = f['activations'][sample_id][layer_idx]

                # Compute delta
                delta = paraphrase_act - neutral_act

                # Get labels
                labels = {
                    axis: level_map[meta[axis]]
                    for axis in AXES
                }

                paired_samples.append(PairedSample(
                    neutral_id=f"neutral_{neutral_idx}",
                    paraphrase_id=sample_id,
                    delta_activation=delta,
                    labels=labels,
                ))

        print(f"Created {len(paired_samples)} paired samples (skipped {skipped})")
        return paired_samples


def train_epoch(
    model: OrthogonalContrastiveProbes,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ortho_weight: float,
    device: str,
) -> Tuple[float, float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()

    total_loss_sum = 0.0
    ortho_loss_sum = 0.0
    n_batches = 0

    # Track predictions for correlation
    all_preds = {axis: [] for axis in AXES}
    all_labels = {axis: [] for axis in AXES}

    for batch in dataloader:
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

        # Regression loss for each axis
        reg_losses = []
        for axis in AXES:
            pred = model(x, axis)
            loss = F.mse_loss(pred, labels_dict[axis])
            reg_losses.append(loss)

            # Track predictions
            all_preds[axis].extend(pred.detach().cpu().numpy())
            all_labels[axis].extend(labels_dict[axis].cpu().numpy())

        reg_loss = torch.stack(reg_losses).mean()

        # Orthogonality loss
        ortho_loss = model.orthogonality_loss()

        # Total loss
        total_loss = reg_loss + ortho_weight * ortho_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum += total_loss.item()
        ortho_loss_sum += ortho_loss.item()
        n_batches += 1

    # Compute correlations
    correlations = {}
    for axis in AXES:
        if len(set(all_labels[axis])) > 1:
            corr, _ = pearsonr(all_labels[axis], all_preds[axis])
            correlations[axis] = corr
        else:
            correlations[axis] = 0.0

    return total_loss_sum / n_batches, ortho_loss_sum / n_batches, correlations


def evaluate(
    model: OrthogonalContrastiveProbes,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model."""
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
                pred = model(x, axis)
                all_preds[axis].extend(pred.cpu().numpy())
                all_labels[axis].extend(labels_dict[axis].numpy())

    # Compute metrics
    metrics = {}
    for axis in AXES:
        preds = np.array(all_preds[axis])
        labels = np.array(all_labels[axis])

        # Correlation
        if len(set(labels)) > 1:
            corr, _ = pearsonr(labels, preds)
            metrics[f'{axis}_corr'] = corr
        else:
            metrics[f'{axis}_corr'] = 0.0

        # Classification accuracy (binning predictions)
        pred_class = np.sign(preds).astype(int)  # -1, 0, 1
        true_class = labels.astype(int)
        metrics[f'{axis}_acc'] = accuracy_score(true_class, pred_class)

        # MSE
        metrics[f'{axis}_mse'] = np.mean((preds - labels) ** 2)

    return metrics


def compute_probe_orthogonality(model: OrthogonalContrastiveProbes) -> Dict[str, float]:
    """Compute pairwise cosine similarities between probe weights."""
    vecs = model.get_steering_vectors()

    similarities = []
    axes_list = list(vecs.keys())

    for i in range(len(axes_list)):
        for j in range(i + 1, len(axes_list)):
            v1, v2 = vecs[axes_list[i]], vecs[axes_list[j]]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            similarities.append(abs(cos))

    return {
        'mean': np.mean(similarities) if similarities else 0.0,
        'min': np.min(similarities) if similarities else 0.0,
        'max': np.max(similarities) if similarities else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train contrastive axis probes")
    parser.add_argument("--data", type=Path, required=True, help="HDF5 file with activations")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=5, help="Layer index")
    parser.add_argument("--ortho-weight", type=float, required=True, help="Orthogonality weight")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("TRAINING CONTRASTIVE AXIS PROBES")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Layer: {args.layer}")
    print(f"Ortho weight: {args.ortho_weight}")
    print()

    # Load paired data
    print("Loading paired data...")
    paired_samples = load_paired_data(args.data, args.layer)
    print()

    # Prepare data
    deltas = np.stack([s.delta_activation for s in paired_samples]).astype(np.float32)
    labels_by_axis = {
        axis: np.array([s.labels[axis] for s in paired_samples], dtype=np.float32)
        for axis in AXES
    }

    print(f"Delta activations shape: {deltas.shape}")
    print(f"Delta stats: mean={deltas.mean():.4f}, std={deltas.std():.4f}")
    print()

    # Class distribution
    print("Label distribution:")
    for axis in AXES:
        labels = labels_by_axis[axis]
        counts = {
            'low (-1)': np.sum(labels == -1),
            'neutral (0)': np.sum(labels == 0),
            'high (1)': np.sum(labels == 1),
        }
        print(f"  {axis:20s}: {counts}")
    print()

    # Normalize deltas
    scaler = StandardScaler()
    deltas_norm = scaler.fit_transform(deltas)

    # Train/test split
    valence_classes = (labels_by_axis['valence'] + 1).astype(int)  # -1,0,1 -> 0,1,2
    idx_train, idx_test = train_test_split(
        np.arange(len(deltas_norm)),
        test_size=args.test_size,
        stratify=valence_classes,
        random_state=args.seed,
    )

    X_train = deltas_norm[idx_train]
    X_test = deltas_norm[idx_test]
    y_train = {axis: labels_by_axis[axis][idx_train] for axis in AXES}
    y_test = {axis: labels_by_axis[axis][idx_test] for axis in AXES}

    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    print()

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train['valence']),
        torch.FloatTensor(y_train['arousal']),
        torch.FloatTensor(y_train['dominance']),
        torch.FloatTensor(y_train['trust']),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test['valence']),
        torch.FloatTensor(y_test['arousal']),
        torch.FloatTensor(y_test['dominance']),
        torch.FloatTensor(y_test['trust']),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    hidden_dim = deltas.shape[1]
    model = OrthogonalContrastiveProbes(hidden_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print("Model initialized:")
    print(f"  4 contrastive probes × {hidden_dim} dimensions")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Training loop
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)
    print()

    best_avg_corr = -1.0
    patience_counter = 0
    history = []
    model_path = None

    for epoch in range(args.n_epochs):
        train_loss, train_ortho, train_corrs = train_epoch(
            model, train_loader, optimizer, args.ortho_weight, args.device
        )

        test_metrics = evaluate(model, test_loader, args.device)

        test_corrs = {axis: test_metrics[f'{axis}_corr'] for axis in AXES}
        test_accs = {axis: test_metrics[f'{axis}_acc'] for axis in AXES}
        test_mses = {axis: test_metrics[f'{axis}_mse'] for axis in AXES}

        avg_train_corr = np.mean(list(train_corrs.values()))
        avg_test_corr = np.mean(list(test_corrs.values()))
        avg_test_acc = np.mean(list(test_accs.values()))

        ortho_metrics = compute_probe_orthogonality(model)

        print(f"Epoch {epoch + 1:3d}/{args.n_epochs}")
        print(f"  Train loss: {train_loss:.4f} (ortho: {train_ortho:.4f})")
        print(f"  Train corr: {avg_train_corr:.3f} (V:{train_corrs['valence']:.3f} "
              f"A:{train_corrs['arousal']:.3f} D:{train_corrs['dominance']:.3f} "
              f"AA:{train_corrs['trust']:.3f})")
        print(f"  Test corr:  {avg_test_corr:.3f} (V:{test_corrs['valence']:.3f} "
              f"A:{test_corrs['arousal']:.3f} D:{test_corrs['dominance']:.3f} "
              f"AA:{test_corrs['trust']:.3f})")
        print(f"  Test acc:   {avg_test_acc:.3f} (V:{test_accs['valence']:.3f} "
              f"A:{test_accs['arousal']:.3f} D:{test_accs['dominance']:.3f} "
              f"AA:{test_accs['trust']:.3f})")
        print(f"  Ortho:      mean={ortho_metrics['mean']:.4f} "
              f"min={ortho_metrics['min']:.4f} max={ortho_metrics['max']:.4f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_ortho_loss': train_ortho,
            'train_correlations': train_corrs,
            'test_correlations': test_corrs,
            'test_accuracies': test_accs,
            'test_mses': test_mses,
            'avg_train_corr': avg_train_corr,
            'avg_test_corr': avg_test_corr,
            'avg_test_acc': avg_test_acc,
            'orthogonality': ortho_metrics,
        })

        if avg_test_corr > best_avg_corr:
            best_avg_corr = avg_test_corr
            patience_counter = 0

            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = args.output_dir / f"probe_layer{args.layer}_ortho{args.ortho_weight}_seed{args.seed}.pt"

            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'args': vars(args),
                'best_epoch': epoch + 1,
                'best_avg_corr': best_avg_corr,
                'test_metrics': test_metrics,
                'steering_vectors': model.get_steering_vectors(),
            }, model_path)

            print(f"  → Best model saved (avg_corr={best_avg_corr:.3f})")
        else:
            patience_counter += 1

        print()

        if patience_counter >= args.patience:
            print(f"✓ Early stopping at epoch {epoch + 1}")
            break

    # Save history
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    history_serializable = convert_to_json_serializable(history)
    history_path = args.output_dir / f"history_layer{args.layer}_ortho{args.ortho_weight}_seed{args.seed}.json"
    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best test correlation: {best_avg_corr:.3f}")
    if model_path:
        print(f"Model saved to: {model_path}")
    print(f"History saved to: {history_path}")
    print()


if __name__ == "__main__":
    main()
