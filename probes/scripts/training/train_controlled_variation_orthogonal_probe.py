#!/usr/bin/env python3
"""Train orthogonal regularized probes on controlled variation data.

This script trains source-specific emotion probes (user or assistant) with
orthogonality penalties to avoid neutral and shared emotion directions.

Usage:
    # Train user probe at layer 30 with λ=100
    python train_controlled_variation_orthogonal_probe.py \
        --isolation-type user \
        --layer 30 \
        --lambda-reg 100 \
        --k-neutral 20 \
        --k-shared 10 \
        --output outputs/probes/controlled_variation/orthogonal/

    # Sweep multiple lambda values
    for lambda in 10 100 1000; do
        python train_controlled_variation_orthogonal_probe.py \
            --isolation-type user \
            --layer 30 \
            --lambda-reg $lambda \
            --output outputs/probes/controlled_variation/orthogonal/
    done
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm


class ControlledVariationDataset(Dataset):
    """Dataset for controlled variation activations."""

    def __init__(self, h5_path: Path, layer: int, condition: str = "emotional"):
        """
        Args:
            h5_path: Path to isolation h5 file (user_isolation.h5 or assistant_isolation.h5)
            layer: Layer index to extract
            condition: 'emotional' or 'neutral'
        """
        self.h5_path = h5_path
        self.layer = layer
        self.condition = condition

        # Load metadata to get labels
        with h5py.File(h5_path, 'r') as f:
            # Read metadata as JSON string
            metadata_json = f['metadata'][()]
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')
            metadata = json.loads(metadata_json)

            # Extract emotion labels from metadata list
            self.pair_keys = []
            self.labels = []
            for entry in metadata:
                self.pair_keys.append(entry['id'])
                self.labels.append(entry['emotion'])

        # Create emotion to index mapping
        self.emotion_to_idx = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'sadness': 4,
            'surprise': 5,
        }

        # Filter out any unknown emotions
        valid_indices = [i for i, label in enumerate(self.labels)
                        if label in self.emotion_to_idx]
        self.pair_keys = [self.pair_keys[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

    def __len__(self):
        return len(self.pair_keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            pair = f['activations'][self.pair_keys[idx]]
            activation = torch.from_numpy(pair[self.condition][self.layer]).float()

        label = self.emotion_to_idx[self.labels[idx]]

        return activation, label


class SourceSpecificEmotionProbe(nn.Module):
    """Linear probe with orthogonality regularization."""

    def __init__(self, hidden_dim: int, num_emotions: int, orthogonal_pcs: np.ndarray = None):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_emotions)

        # Register orthogonal PCs as buffer (not trained)
        if orthogonal_pcs is not None:
            self.register_buffer(
                'orthogonal_pcs',
                torch.from_numpy(orthogonal_pcs).float()
            )
        else:
            self.orthogonal_pcs = None

    def forward(self, x):
        return self.linear(x)

    def compute_orthogonality_penalty(self):
        """Compute penalty for projection onto orthogonal PCs."""
        if self.orthogonal_pcs is None:
            return torch.tensor(0.0)

        w = self.linear.weight  # [num_emotions, hidden_dim]
        U = self.orthogonal_pcs  # [hidden_dim, k]

        # Project weight vectors onto orthogonal directions
        projections = w @ U  # [num_emotions, k]

        # Penalty is sum of squared projections
        penalty = torch.sum(projections ** 2)

        return penalty


def load_orthogonal_pcs(
    base_dir: Path,
    layer: int,
    k_neutral: int = 20,
    k_shared: int = 10,
    use_neutral: bool = True,
    use_shared: bool = True,
) -> np.ndarray:
    """Load and concatenate neutral and shared PCs."""
    pcs = []

    if use_neutral:
        neutral_path = base_dir / "neutral_pcs" / f"layer_{layer}_neutral_pcs_k{k_neutral}.npy"
        neutral_pcs = np.load(neutral_path)
        pcs.append(neutral_pcs)

    if use_shared:
        shared_path = base_dir / "shared_pcs" / f"layer_{layer}_shared_pcs_k{k_shared}.npy"
        shared_pcs = np.load(shared_path)
        pcs.append(shared_pcs)

    if len(pcs) == 0:
        return None

    return np.concatenate(pcs, axis=1)  # [hidden_dim, k_total]


def train_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lambda_reg: float,
    num_epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 3,
    weight_decay: float = 1.0,
    device: str = 'cuda',
) -> Dict:
    """Train probe with orthogonality regularization."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'orthogonal_penalty': [],
    }

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        train_penalties = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_x)
            loss_emotion = criterion(logits, batch_y)

            # Orthogonality penalty
            loss_orthogonal = model.compute_orthogonality_penalty()

            # Combined loss
            loss_total = loss_emotion + lambda_reg * loss_orthogonal

            # Backward pass
            loss_total.backward()
            optimizer.step()

            # Track metrics
            train_losses.append(loss_total.item())
            train_penalties.append(loss_orthogonal.item())
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(batch_y.cpu().numpy())

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                loss_emotion = criterion(logits, batch_y)
                loss_orthogonal = model.compute_orthogonality_penalty()
                loss_total = loss_emotion + lambda_reg * loss_orthogonal

                val_losses.append(loss_total.item())
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        # Compute metrics
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(np.mean(train_losses))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(np.mean(val_losses))
        history['val_acc'].append(val_acc)
        history['orthogonal_penalty'].append(np.mean(train_penalties))

        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={history['train_loss'][-1]:.4f}, "
                  f"Train Acc={train_acc:.4f}, "
                  f"Val Acc={val_acc:.4f}, "
                  f"Penalty={history['orthogonal_penalty'][-1]:.4f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, history, best_val_acc


def evaluate_probe(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> Dict:
    """Evaluate probe on test set."""

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_logits = np.vstack(all_logits)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # Classification report
    emotion_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    report = classification_report(all_labels, all_preds, target_names=emotion_names, output_dict=True)

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': all_preds,
        'labels': all_labels,
        'logits': all_logits,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train orthogonal regularized probe on controlled variation"
    )
    parser.add_argument(
        "--isolation-type",
        type=str,
        required=True,
        choices=["user", "assistant"],
        help="Which isolation type to train on",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to train probe on",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        required=True,
        help="Regularization strength (0 = no regularization)",
    )
    parser.add_argument(
        "--k-neutral",
        type=int,
        default=20,
        help="Number of neutral PCs (default: 20)",
    )
    parser.add_argument(
        "--k-shared",
        type=int,
        default=10,
        help="Number of shared PCs (default: 10)",
    )
    parser.add_argument(
        "--use-neutral",
        type=bool,
        default=True,
        help="Use neutral PCs in regularization (default: True)",
    )
    parser.add_argument(
        "--use-shared",
        type=bool,
        default=True,
        help="Use shared PCs in regularization (default: True)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0,
        help="L2 regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Train split ratio (default: 0.7)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/probes/controlled_variation/orthogonal",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("TRAINING ORTHOGONAL REGULARIZED PROBE")
    print("=" * 80)
    print(f"Isolation type: {args.isolation_type}")
    print(f"Layer: {args.layer}")
    print(f"Lambda: {args.lambda_reg}")
    print(f"k_neutral: {args.k_neutral}, k_shared: {args.k_shared}")
    print(f"Use neutral: {args.use_neutral}, Use shared: {args.use_shared}")
    print(f"Device: {device}")
    print()

    # Load data
    data_path = Path(f"outputs/activations/controlled_variation/{args.isolation_type}_isolation.h5")
    print(f"Loading data from: {data_path}")

    dataset = ControlledVariationDataset(data_path, args.layer, condition="emotional")
    print(f"Total samples: {len(dataset)}")

    # Split data
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load orthogonal PCs
    if args.lambda_reg > 0:
        print("\nLoading orthogonal PCs...")
        orthogonal_pcs_dir = Path("outputs/orthogonal_pcs/controlled_variation")
        orthogonal_pcs = load_orthogonal_pcs(
            orthogonal_pcs_dir,
            args.layer,
            k_neutral=args.k_neutral,
            k_shared=args.k_shared,
            use_neutral=args.use_neutral,
            use_shared=args.use_shared,
        )
        print(f"Orthogonal PCs shape: {orthogonal_pcs.shape}")
    else:
        print("\nNo regularization (lambda=0)")
        orthogonal_pcs = None

    # Create model
    hidden_dim = 5376  # Gemma-3-27B hidden dimension
    num_emotions = 6

    model = SourceSpecificEmotionProbe(hidden_dim, num_emotions, orthogonal_pcs)
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    # Train
    print("\nTraining...")
    model, history, best_val_acc = train_probe(
        model,
        train_loader,
        val_loader,
        lambda_reg=args.lambda_reg,
        num_epochs=args.num_epochs,
        lr=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay,
        device=device,
    )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Evaluate on in-distribution test set
    print("\n" + "=" * 80)
    print(f"EVALUATING ON {args.isolation_type.upper()} TEST SET (IN-DISTRIBUTION)")
    print("=" * 80)

    test_results = evaluate_probe(model, test_loader, device)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")

    # Evaluate on cross-isolation (other isolation type)
    other_isolation = "assistant" if args.isolation_type == "user" else "user"
    other_data_path = Path(f"outputs/activations/controlled_variation/{other_isolation}_isolation.h5")

    print("\n" + "=" * 80)
    print(f"EVALUATING ON {other_isolation.upper()} TEST SET (CROSS-ISOLATION)")
    print("=" * 80)

    if other_data_path.exists():
        other_dataset = ControlledVariationDataset(other_data_path, args.layer, condition="emotional")
        other_loader = DataLoader(other_dataset, batch_size=args.batch_size, shuffle=False)

        cross_results = evaluate_probe(model, other_loader, device)
        print(f"Cross-Isolation Accuracy: {cross_results['accuracy']:.4f}")
        print(f"(Random baseline: {1/6:.4f} = 16.7%)")
    else:
        cross_results = None
        print("Cross-isolation data not found")

    # Save results
    output_dir = Path(args.output) / f"{args.isolation_type}_layer{args.layer}_lambda{args.lambda_reg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # Save results
    results = {
        'args': vars(args),
        'best_val_acc': float(best_val_acc),
        'test_results': {
            'accuracy': test_results['accuracy'],
            'classification_report': test_results['classification_report'],
            'confusion_matrix': test_results['confusion_matrix'],
        },
        'cross_isolation_results': {
            'accuracy': cross_results['accuracy'] if cross_results else None,
            'classification_report': cross_results['classification_report'] if cross_results else None,
        } if cross_results else None,
        'history': history,
    }

    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Isolation type: {args.isolation_type}")
    print(f"Layer: {args.layer}")
    print(f"Lambda: {args.lambda_reg}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy (in-distribution): {test_results['accuracy']:.4f}")
    if cross_results:
        print(f"Test accuracy (cross-isolation): {cross_results['accuracy']:.4f}")
        specificity = 1 - cross_results['accuracy']
        print(f"Source specificity: {specificity:.4f} (higher is better)")
    print("=" * 80)


if __name__ == "__main__":
    main()
