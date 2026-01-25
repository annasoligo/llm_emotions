#!/usr/bin/env python3
"""Orthogonality-regularized probe training.

This module implements linear probes with orthogonality regularization to neutral PCs.
The key idea is to penalize probe weights that align with neutral-dominated principal
components, forcing the probe to learn emotion representations that are orthogonal
to neutral content structure.

Loss function:
    L = L_emotion + λ * ||U.T @ w||²

Where:
    - L_emotion: Standard cross-entropy loss for emotion classification
    - U: Matrix of neutral PCs [hidden_dim, k]
    - w: Probe weights [n_classes, hidden_dim]
    - λ: Orthogonality penalty weight
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class OrthogonalRegularizedProbe(nn.Module):
    """Linear probe with orthogonality regularization to neutral PCs.

    This probe is trained to predict emotions while being penalized for
    aligning with neutral-dominated principal components.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        neutral_pcs: Optional[np.ndarray] = None,
    ):
        """Initialize orthogonal regularized probe.

        Args:
            input_dim: Dimensionality of input activations
            n_classes: Number of emotion classes to predict
            neutral_pcs: [input_dim, k] matrix of neutral PC directions.
                        If None, no orthogonality regularization is applied.
        """
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Linear classifier
        self.linear = nn.Linear(input_dim, n_classes)

        # Register neutral PCs as buffer (not a parameter, but moved to device)
        if neutral_pcs is not None:
            self.register_buffer(
                'neutral_pcs',
                torch.FloatTensor(neutral_pcs)
            )
            self.has_neutral_pcs = True
        else:
            self.register_buffer('neutral_pcs', None)
            self.has_neutral_pcs = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through probe.

        Args:
            x: [batch, input_dim] activation vectors

        Returns:
            logits: [batch, n_classes] class logits
        """
        return self.linear(x)

    def compute_orthogonality_penalty(self) -> torch.Tensor:
        """Compute ||U.T @ w||² orthogonality penalty.

        The penalty measures how much the probe weights align with the
        neutral PC subspace. Higher values mean more alignment (bad).

        Returns:
            penalty: Scalar tensor, 0 if no neutral PCs provided
        """
        if not self.has_neutral_pcs:
            return torch.tensor(0.0, device=self.linear.weight.device)

        # Get probe weights: [n_classes, input_dim]
        w = self.linear.weight

        # Get neutral PCs: [input_dim, k]
        U = self.neutral_pcs

        # Project weights onto neutral subspace: w @ U = [n_classes, k]
        projections = w @ U

        # Penalty is squared Frobenius norm of projections
        penalty = torch.sum(projections ** 2)

        return penalty

    def get_orthogonality_metrics(self) -> Dict[str, float]:
        """Compute detailed orthogonality metrics for analysis.

        Returns:
            metrics: Dictionary with:
                - ortho_frobenius: Frobenius norm of projections
                - ortho_normalized: Normalized overlap (0=orthogonal, 1=aligned)
                - neutral_fraction: Fraction of probe weight in neutral subspace
        """
        if not self.has_neutral_pcs:
            return {
                'ortho_frobenius': 0.0,
                'ortho_normalized': 0.0,
                'neutral_fraction': 0.0,
            }

        with torch.no_grad():
            w = self.linear.weight.cpu().numpy()  # [n_classes, input_dim]
            U = self.neutral_pcs.cpu().numpy()    # [input_dim, k]

            # Projection onto neutral subspace
            projections = w @ U  # [n_classes, k]

            # Frobenius norm
            ortho_frobenius = np.linalg.norm(projections, 'fro')

            # Normalized overlap (0 = fully orthogonal, 1 = fully aligned)
            w_norm = np.linalg.norm(w)
            ortho_normalized = ortho_frobenius / (w_norm + 1e-8)

            # Fraction of weight in neutral subspace
            neutral_component = U @ (U.T @ w.T)  # Project w onto neutral subspace
            neutral_fraction = np.linalg.norm(neutral_component) / (w_norm + 1e-8)

        return {
            'ortho_frobenius': float(ortho_frobenius),
            'ortho_normalized': float(ortho_normalized),
            'neutral_fraction': float(neutral_fraction),
        }


def train_orthogonal_regularized_probe(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    test_activations: np.ndarray,
    test_labels: np.ndarray,
    neutral_pcs: Optional[np.ndarray] = None,
    lambda_ortho: float = 1.0,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 10,
    weight_decay: float = 1.0,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict:
    """Train orthogonality-regularized probe with early stopping.

    Args:
        train_activations: [n_train, input_dim] training activations
        train_labels: [n_train] integer labels (0 to n_classes-1)
        test_activations: [n_test, input_dim] test activations
        test_labels: [n_test] integer labels
        neutral_pcs: [input_dim, k] neutral PC directions. If None, trains standard probe.
        lambda_ortho: Weight for orthogonality penalty (default: 1.0)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        batch_size: Batch size for training (default: 32)
        max_epochs: Maximum number of epochs (default: 500)
        patience: Early stopping patience (default: 10)
        weight_decay: L2 regularization strength (default: 1.0)
        device: Device to use ('cuda' or 'cpu')
        verbose: Print training progress

    Returns:
        results: Dictionary with:
            - model: Trained OrthogonalRegularizedProbe
            - train_accuracy: Final train accuracy
            - test_accuracy: Final test accuracy
            - best_test_accuracy: Best test accuracy during training
            - best_epoch: Epoch with best test accuracy
            - total_epochs: Total epochs trained
            - history: Training history with losses and metrics
            - config: Training configuration
            - ortho_metrics: Final orthogonality metrics
    """
    # Validate inputs
    if train_activations.shape[0] != len(train_labels):
        raise ValueError(
            f"Train activations ({train_activations.shape[0]}) and labels "
            f"({len(train_labels)}) must have same length"
        )
    if test_activations.shape[0] != len(test_labels):
        raise ValueError(
            f"Test activations ({test_activations.shape[0]}) and labels "
            f"({len(test_labels)}) must have same length"
        )
    if train_activations.shape[1] != test_activations.shape[1]:
        raise ValueError(
            f"Train and test activations must have same input_dim: "
            f"{train_activations.shape[1]} vs {test_activations.shape[1]}"
        )

    n_classes = len(np.unique(train_labels))
    input_dim = train_activations.shape[1]

    if verbose:
        print(f"Training orthogonal regularized probe:")
        print(f"  Train samples: {len(train_labels)}")
        print(f"  Test samples: {len(test_labels)}")
        print(f"  Input dim: {input_dim}")
        print(f"  Classes: {n_classes}")
        if neutral_pcs is not None:
            print(f"  Neutral PCs: {neutral_pcs.shape[1]} directions")
            print(f"  Lambda orthogonality: {lambda_ortho}")
        else:
            print(f"  No orthogonality regularization (standard probe)")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Patience: {patience}")
        print(f"  Weight decay (L2): {weight_decay}")

    # Convert to tensors
    train_acts_tensor = torch.from_numpy(train_activations).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    test_acts_tensor = torch.from_numpy(test_activations).float()
    test_labels_tensor = torch.from_numpy(test_labels).long()

    # Create datasets and loaders
    train_dataset = TensorDataset(train_acts_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_acts_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = OrthogonalRegularizedProbe(input_dim, n_classes, neutral_pcs).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "emotion_loss": [],
        "ortho_penalty": [],
        "ortho_normalized": [],
        "epoch": [],
    }

    # Early stopping state
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    # Training loop
    if verbose:
        print("\nTraining...")
        epoch_iterator = tqdm(range(max_epochs), desc="Epochs")
    else:
        epoch_iterator = range(max_epochs)

    for epoch in epoch_iterator:
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_emotion_loss = 0.0
        epoch_ortho_penalty = 0.0

        for acts, labels in train_loader:
            acts = acts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(acts)
            emotion_loss = criterion(logits, labels)

            # Orthogonality penalty
            ortho_penalty = model.compute_orthogonality_penalty()

            # Total loss
            total_loss = emotion_loss + lambda_ortho * ortho_penalty

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Track metrics
            train_loss += total_loss.item() * len(labels)
            epoch_emotion_loss += emotion_loss.item() * len(labels)
            epoch_ortho_penalty += ortho_penalty.item() * len(labels)

            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        # Evaluate on test
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for acts, labels in test_loader:
                acts = acts.to(device)
                labels = labels.to(device)

                logits = model(acts)
                loss = criterion(logits, labels)

                test_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += len(labels)

        # Compute metrics
        train_loss /= train_total
        train_acc = train_correct / train_total
        test_loss /= test_total
        test_acc = test_correct / test_total
        epoch_emotion_loss /= train_total
        epoch_ortho_penalty /= train_total

        # Get orthogonality metrics
        ortho_metrics = model.get_orthogonality_metrics()

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["emotion_loss"].append(epoch_emotion_loss)
        history["ortho_penalty"].append(epoch_ortho_penalty)
        history["ortho_normalized"].append(ortho_metrics['ortho_normalized'])
        history["epoch"].append(epoch)

        # Early stopping check
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress every 10 epochs or on improvement
        if verbose and ((epoch + 1) % 10 == 0 or epochs_without_improvement == 0):
            star = "🌟" if epochs_without_improvement == 0 else ""
            print(
                f"Epoch {epoch+1:3d}/{max_epochs}: "
                f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
                f"Emotion Loss={epoch_emotion_loss:.4f}, "
                f"Ortho Penalty={epoch_ortho_penalty:.6f}, "
                f"Ortho Norm={ortho_metrics['ortho_normalized']:.4f} {star}"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final evaluation with best model
    model.eval()
    with torch.no_grad():
        # Train predictions
        train_preds = []
        train_labels_ordered = []
        for acts, labels in train_loader:
            acts = acts.to(device)
            logits = model(acts)
            preds = logits.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_ordered.extend(labels.cpu().numpy())
        train_preds = np.array(train_preds)
        train_labels_ordered = np.array(train_labels_ordered)

        # Test predictions
        test_preds = []
        test_labels_ordered = []
        for acts, labels in test_loader:
            acts = acts.to(device)
            logits = model(acts)
            preds = logits.argmax(dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels_ordered.extend(labels.cpu().numpy())
        test_preds = np.array(test_preds)
        test_labels_ordered = np.array(test_labels_ordered)

    final_train_acc = accuracy_score(train_labels_ordered, train_preds)
    final_test_acc = accuracy_score(test_labels_ordered, test_preds)
    final_ortho_metrics = model.get_orthogonality_metrics()

    results = {
        "model": model,
        "train_accuracy": final_train_acc,
        "test_accuracy": final_test_acc,
        "best_test_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "history": history,
        "config": {
            "n_classes": n_classes,
            "input_dim": input_dim,
            "neutral_pcs_shape": neutral_pcs.shape if neutral_pcs is not None else None,
            "lambda_ortho": lambda_ortho,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "weight_decay": weight_decay,
        },
        "train_predictions": train_preds,
        "test_predictions": test_preds,
        "ortho_metrics": final_ortho_metrics,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Final Results (best model from epoch {best_epoch+1}):")
        print(f"  Train accuracy: {final_train_acc:.4f}")
        print(f"  Test accuracy: {final_test_acc:.4f}")
        print(f"  Best test accuracy: {best_test_acc:.4f}")
        if neutral_pcs is not None:
            print(f"\nOrthogonality Metrics:")
            print(f"  Frobenius norm: {final_ortho_metrics['ortho_frobenius']:.4f}")
            print(f"  Normalized overlap: {final_ortho_metrics['ortho_normalized']:.4f}")
            print(f"  Neutral fraction: {final_ortho_metrics['neutral_fraction']:.4f}")
        print(f"{'='*60}")

    return results
