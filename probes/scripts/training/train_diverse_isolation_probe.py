#!/usr/bin/env python3
"""
Train probes on diverse isolation data with opposite-source emotion regularization.

Key difference from controlled variation:
- Data: Global activations where both user AND assistant have varied emotions
- Regularization: Penalize OPPOSITE-source emotion PCs (not shared emotion PCs)
  - For user probe: penalize assistant emotion PCs
  - For assistant probe: penalize user emotion PCs
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split


class DiverseIsolationDataset(Dataset):
    """Dataset for diverse isolation activations."""

    def __init__(self, data_path: str, layer: int):
        with h5py.File(data_path, 'r') as f:
            # Load activations for specific layer
            self.activations = f[f'layer_{layer}'][:]

            # Load metadata
            metadata_json = f['metadata'][()]
            if isinstance(metadata_json, bytes):
                metadata_json = metadata_json.decode('utf-8')
            metadata = json.loads(metadata_json)

            # Get emotions
            emotions = [entry['emotion'] for entry in metadata]

            # Create emotion to index mapping
            unique_emotions = sorted(set(emotions))
            self.emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}
            self.idx_to_emotion = {i: e for e, i in self.emotion_to_idx.items()}

            # Convert to indices
            self.labels = [self.emotion_to_idx[e] for e in emotions]

        self.activations = torch.FloatTensor(self.activations)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


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
            return 0.0

        # Get weight vector (hidden_dim,)
        w = self.linear.weight  # (num_emotions, hidden_dim)

        # Project onto orthogonal PCs: w @ U
        # U shape: (hidden_dim, k)
        # w shape: (num_emotions, hidden_dim)
        # Result: (num_emotions, k)
        projections = w @ self.orthogonal_pcs

        # Compute L2 norm
        penalty = (projections ** 2).sum()

        return penalty


def train_probe(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lambda_reg: float,
    num_epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 3,
    weight_decay: float = 10.0,
    device: str = 'cuda',
):
    """Train probe with orthogonality regularization."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            # Add orthogonality penalty
            if lambda_reg > 0:
                penalty = model.compute_orthogonality_penalty()
                loss = loss + lambda_reg * penalty

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_acc = val_correct / val_total

        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    return best_val_acc


def evaluate_probe(
    model: nn.Module,
    test_loader: DataLoader,
    emotion_names: list,
    device: str = 'cuda',
):
    """Evaluate probe on test set."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=emotion_names, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train diverse isolation probe")
    parser.add_argument("--isolation-type", type=str, required=True, choices=["user", "assistant"])
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--lambda-reg", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output", type=str, default="outputs/probes/diverse_isolation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("TRAINING DIVERSE ISOLATION PROBE")
    print("=" * 80)
    print(f"Isolation type: {args.isolation_type}")
    print(f"Layer: {args.layer}")
    print(f"Lambda: {args.lambda_reg}")
    print(f"Device: {device}")
    print()

    # Load data
    data_path = f"outputs/activations/diverse_isolation/{args.isolation_type}_isolation.h5"
    print(f"Loading data from: {data_path}")

    dataset = DiverseIsolationDataset(data_path, args.layer)

    print(f"Total samples: {len(dataset)}")
    print(f"Emotions: {list(dataset.emotion_to_idx.keys())}")

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load orthogonal PCs
    orthogonal_pcs = None
    if args.lambda_reg > 0:
        pc_path = f"outputs/orthogonal_pcs/diverse_isolation/{args.isolation_type}/layer_{args.layer}.npz"
        print(f"Loading orthogonal PCs from: {pc_path}")
        pc_data = np.load(pc_path)
        orthogonal_pcs = pc_data['orthogonal_pcs']
        print(f"Orthogonal PCs shape: {orthogonal_pcs.shape}")
        print()

    # Create model
    hidden_dim = dataset.activations.shape[1]
    num_emotions = len(dataset.emotion_to_idx)

    model = SourceSpecificEmotionProbe(hidden_dim, num_emotions, orthogonal_pcs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params} parameters")
    print()

    # Train
    print("Training...")
    best_val_acc = train_probe(
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

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    test_results = evaluate_probe(model, test_loader, list(dataset.emotion_to_idx.keys()), device)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print()

    # Save model and results
    output_dir = Path(args.output) / f"{args.isolation_type}_layer{args.layer}_lambda{args.lambda_reg}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    results = {
        'args': vars(args),
        'best_val_acc': float(best_val_acc),
        'test_results': test_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Isolation type: {args.isolation_type}")
    print(f"Layer: {args.layer}")
    print(f"Lambda: {args.lambda_reg}")
    print(f"Validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
