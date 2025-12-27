"""Probe training - orthogonal and linear probes."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class EmotionProbes(nn.Module):
    """Learnable probe directions with orthogonality constraints."""

    def __init__(self, hidden_dim: int, n_emotions: int):
        """Initialize emotion probes.

        Args:
            hidden_dim: Dimensionality of hidden states
            n_emotions: Number of emotion classes
        """
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_emotions <= 0:
            raise ValueError(f"n_emotions must be positive, got {n_emotions}")

        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_emotions = n_emotions

        # User and assistant probe directions
        self.user_probes = nn.Parameter(torch.randn(n_emotions, hidden_dim))
        self.assistant_probes = nn.Parameter(torch.randn(n_emotions, hidden_dim))

        # Initialize with small values for stability
        nn.init.xavier_normal_(self.user_probes, gain=0.1)
        nn.init.xavier_normal_(self.assistant_probes, gain=0.1)

    def forward(self, activations: torch.Tensor, probe_type: str) -> torch.Tensor:
        """Project activations onto probe directions.

        Args:
            activations: [batch, hidden_dim] activations
            probe_type: 'user' or 'assistant'

        Returns:
            Logits [batch, n_emotions]

        Raises:
            ValueError: If probe_type invalid or shape mismatch
        """
        if probe_type not in ("user", "assistant"):
            raise ValueError(f"probe_type must be 'user' or 'assistant', got {probe_type}")

        if activations.shape[1] != self.hidden_dim:
            raise ValueError(
                f"Expected activations dim {self.hidden_dim}, got {activations.shape[1]}"
            )

        probes = self.user_probes if probe_type == "user" else self.assistant_probes

        # Normalize probes
        probes_norm = probes / probes.norm(dim=1, keepdim=True)

        return activations @ probes_norm.T  # [batch, n_emotions]

    def orthogonality_loss(self) -> torch.Tensor:
        """Penalize non-orthogonality between user and assistant subspaces.

        Returns:
            Orthogonality loss (0 = perfect orthogonality)
        """
        # Normalize
        user_norm = self.user_probes / self.user_probes.norm(dim=1, keepdim=True)
        asst_norm = self.assistant_probes / self.assistant_probes.norm(dim=1, keepdim=True)

        # Cross-group dot products
        cross_dots = user_norm @ asst_norm.T  # [n_emotions, n_emotions]

        # Penalize squared dot products
        return (cross_dots**2).mean()

    def get_probe_vectors(self, probe_type: str) -> np.ndarray:
        """Get normalized probe vectors.

        Args:
            probe_type: 'user' or 'assistant'

        Returns:
            Normalized probe vectors [n_emotions, hidden_dim]
        """
        if probe_type not in ("user", "assistant"):
            raise ValueError(f"probe_type must be 'user' or 'assistant', got {probe_type}")

        probes = self.user_probes if probe_type == "user" else self.assistant_probes
        probes_norm = probes / probes.norm(dim=1, keepdim=True)

        return probes_norm.detach().cpu().numpy()


class EmotionDataset(Dataset):
    """Dataset of emotion-labeled activations."""

    def __init__(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        speaker_types: list[str],
    ):
        """Initialize dataset.

        Args:
            activations: [n_samples, hidden_dim] activations
            labels: [n_samples] emotion labels (integers)
            speaker_types: [n_samples] 'user' or 'assistant'

        Raises:
            ValueError: If shapes don't match or invalid data
        """
        if activations.shape[0] != len(labels):
            raise ValueError(
                f"Activations ({activations.shape[0]}) and labels ({len(labels)}) "
                "must have same length"
            )
        if len(labels) != len(speaker_types):
            raise ValueError(
                f"Labels ({len(labels)}) and speaker_types ({len(speaker_types)}) "
                "must have same length"
            )

        for i, st in enumerate(speaker_types):
            if st not in ("user", "assistant"):
                raise ValueError(
                    f"speaker_types[{i}] must be 'user' or 'assistant', got {st}"
                )

        self.activations = torch.from_numpy(activations).float()
        self.labels = torch.from_numpy(labels).long()
        self.speaker_types = speaker_types

    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: int) -> dict:
        return {
            "activation": self.activations[idx],
            "label": self.labels[idx],
            "speaker_type": self.speaker_types[idx],
        }


def train_orthogonal_probes(
    train_data: dict,
    test_data: dict,
    n_emotions: int,
    hidden_dim: int,
    orthogonality_weight: float = 0.1,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    n_epochs: int = 100,
    device: str = "cuda",
) -> dict:
    """Train orthogonal probes with soft orthogonality constraint.

    Args:
        train_data: Dict with 'activations', 'labels', 'speaker_types'
        test_data: Dict with 'activations', 'labels', 'speaker_types'
        n_emotions: Number of emotion classes
        hidden_dim: Activation dimensionality
        orthogonality_weight: Weight for orthogonality loss
        batch_size: Batch size for training
        learning_rate: Learning rate
        n_epochs: Number of training epochs
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Results dict with probes, metrics, history

    Raises:
        ValueError: If data format invalid
    """
    # Validate inputs
    required_keys = ["activations", "labels", "speaker_types"]
    for key in required_keys:
        if key not in train_data:
            raise ValueError(f"train_data missing key: {key}")
        if key not in test_data:
            raise ValueError(f"test_data missing key: {key}")

    # Create datasets
    train_dataset = EmotionDataset(
        train_data["activations"],
        train_data["labels"],
        train_data["speaker_types"],
    )
    test_dataset = EmotionDataset(
        test_data["activations"],
        test_data["labels"],
        test_data["speaker_types"],
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = EmotionProbes(hidden_dim, n_emotions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        "train_loss": [],
        "train_user_acc": [],
        "train_asst_acc": [],
        "test_user_acc": [],
        "test_asst_acc": [],
        "ortho_loss": [],
    }

    # Training loop
    print(f"Training for {n_epochs} epochs...")
    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            activations = batch["activation"].to(device)
            labels = batch["label"].to(device)
            speaker_types = batch["speaker_type"]

            # Separate user and assistant
            user_mask = [st == "user" for st in speaker_types]
            asst_mask = [st == "assistant" for st in speaker_types]

            optimizer.zero_grad()

            # Classification loss
            loss = 0.0
            if any(user_mask):
                user_acts = activations[user_mask]
                user_labels = labels[user_mask]
                user_logits = model(user_acts, "user")
                loss = loss + criterion(user_logits, user_labels)

            if any(asst_mask):
                asst_acts = activations[asst_mask]
                asst_labels = labels[asst_mask]
                asst_logits = model(asst_acts, "assistant")
                loss = loss + criterion(asst_logits, asst_labels)

            # Orthogonality loss
            ortho_loss = model.orthogonality_loss()
            loss = loss + orthogonality_weight * ortho_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            # Train accuracy
            train_user_correct = 0
            train_user_total = 0
            train_asst_correct = 0
            train_asst_total = 0

            for batch in train_loader:
                activations = batch["activation"].to(device)
                labels = batch["label"].to(device)
                speaker_types = batch["speaker_type"]

                user_mask = [st == "user" for st in speaker_types]
                asst_mask = [st == "assistant" for st in speaker_types]

                if any(user_mask):
                    user_acts = activations[user_mask]
                    user_labels = labels[user_mask]
                    user_preds = model(user_acts, "user").argmax(dim=1)
                    train_user_correct += (user_preds == user_labels).sum().item()
                    train_user_total += len(user_labels)

                if any(asst_mask):
                    asst_acts = activations[asst_mask]
                    asst_labels = labels[asst_mask]
                    asst_preds = model(asst_acts, "assistant").argmax(dim=1)
                    train_asst_correct += (asst_preds == asst_labels).sum().item()
                    train_asst_total += len(asst_labels)

            # Test accuracy
            test_user_correct = 0
            test_user_total = 0
            test_asst_correct = 0
            test_asst_total = 0

            for batch in test_loader:
                activations = batch["activation"].to(device)
                labels = batch["label"].to(device)
                speaker_types = batch["speaker_type"]

                user_mask = [st == "user" for st in speaker_types]
                asst_mask = [st == "assistant" for st in speaker_types]

                if any(user_mask):
                    user_acts = activations[user_mask]
                    user_labels = labels[user_mask]
                    user_preds = model(user_acts, "user").argmax(dim=1)
                    test_user_correct += (user_preds == user_labels).sum().item()
                    test_user_total += len(user_labels)

                if any(asst_mask):
                    asst_acts = activations[asst_mask]
                    asst_labels = labels[asst_labels]
                    asst_preds = model(asst_acts, "assistant").argmax(dim=1)
                    test_asst_correct += (asst_preds == asst_labels).sum().item()
                    test_asst_total += len(asst_labels)

        # Record history
        history["train_loss"].append(epoch_loss / n_batches)
        history["train_user_acc"].append(train_user_correct / max(train_user_total, 1))
        history["train_asst_acc"].append(train_asst_correct / max(train_asst_total, 1))
        history["test_user_acc"].append(test_user_correct / max(test_user_total, 1))
        history["test_asst_acc"].append(test_asst_correct / max(test_asst_total, 1))
        history["ortho_loss"].append(ortho_loss.item())

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}: "
                f"Loss={history['train_loss'][-1]:.4f}, "
                f"Test User Acc={history['test_user_acc'][-1]:.4f}, "
                f"Test Asst Acc={history['test_asst_acc'][-1]:.4f}, "
                f"Ortho={history['ortho_loss'][-1]:.4f}"
            )

    # Return results
    results = {
        "model": model,
        "history": history,
        "user_probes": model.get_probe_vectors("user"),
        "assistant_probes": model.get_probe_vectors("assistant"),
        "config": {
            "n_emotions": n_emotions,
            "hidden_dim": hidden_dim,
            "orthogonality_weight": orthogonality_weight,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
        },
    }

    return results


def train_linear_probes(
    train_data: dict,
    test_data: dict,
    n_emotions: int,
    penalty: str = "l2",
    C: float = 1.0,
) -> dict:
    """Train linear probes using scikit-learn.

    Args:
        train_data: Dict with 'activations', 'labels', 'speaker_types'
        test_data: Dict with 'activations', 'labels', 'speaker_types'
        n_emotions: Number of emotion classes
        penalty: Regularization type ('l1' or 'l2')
        C: Inverse regularization strength

    Returns:
        Results dict with probes and metrics

    Raises:
        ValueError: If data format invalid
    """
    # Validate inputs
    required_keys = ["activations", "labels", "speaker_types"]
    for key in required_keys:
        if key not in train_data:
            raise ValueError(f"train_data missing key: {key}")
        if key not in test_data:
            raise ValueError(f"test_data missing key: {key}")

    # Separate user and assistant data
    train_user_mask = np.array([st == "user" for st in train_data["speaker_types"]])
    train_asst_mask = np.array([st == "assistant" for st in train_data["speaker_types"]])

    test_user_mask = np.array([st == "user" for st in test_data["speaker_types"]])
    test_asst_mask = np.array([st == "assistant" for st in test_data["speaker_types"]])

    # Train user probe
    user_probe = LogisticRegression(penalty=penalty, C=C, max_iter=1000, random_state=42)
    user_probe.fit(
        train_data["activations"][train_user_mask],
        train_data["labels"][train_user_mask],
    )

    # Train assistant probe
    asst_probe = LogisticRegression(penalty=penalty, C=C, max_iter=1000, random_state=42)
    asst_probe.fit(
        train_data["activations"][train_asst_mask],
        train_data["labels"][train_asst_mask],
    )

    # Evaluate
    user_test_acc = accuracy_score(
        test_data["labels"][test_user_mask],
        user_probe.predict(test_data["activations"][test_user_mask]),
    )

    asst_test_acc = accuracy_score(
        test_data["labels"][test_asst_mask],
        asst_probe.predict(test_data["activations"][test_asst_mask]),
    )

    # Disentanglement: cross-accuracy (should be low)
    user_on_asst = accuracy_score(
        test_data["labels"][test_asst_mask],
        user_probe.predict(test_data["activations"][test_asst_mask]),
    )

    asst_on_user = accuracy_score(
        test_data["labels"][test_user_mask],
        asst_probe.predict(test_data["activations"][test_user_mask]),
    )

    results = {
        "user_probe": user_probe,
        "assistant_probe": asst_probe,
        "user_test_acc": user_test_acc,
        "asst_test_acc": asst_test_acc,
        "user_on_asst_cross_acc": user_on_asst,
        "asst_on_user_cross_acc": asst_on_user,
        "disentanglement_score": (
            (user_test_acc + asst_test_acc) / 2 - (user_on_asst + asst_on_user) / 2
        ),
        "config": {
            "n_emotions": n_emotions,
            "penalty": penalty,
            "C": C,
        },
    }

    print(f"\nResults:")
    print(f"  User test acc: {user_test_acc:.4f}")
    print(f"  Asst test acc: {asst_test_acc:.4f}")
    print(f"  User on asst (cross): {user_on_asst:.4f}")
    print(f"  Asst on user (cross): {asst_on_user:.4f}")
    print(f"  Disentanglement score: {results['disentanglement_score']:.4f}")

    return results


def train_multiclass_probe(
    train_activations: np.ndarray,
    train_labels: np.ndarray,
    test_activations: np.ndarray,
    test_labels: np.ndarray,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 500,
    patience: int = 10,
    weight_decay: float = 1.0,
    l1_lambda: float = 0.0,
    device: str = "cuda",
) -> dict:
    """Train a single multiclass probe with early stopping.

    Args:
        train_activations: [n_train, hidden_dim] training activations
        train_labels: [n_train] integer labels (0 to n_classes-1)
        test_activations: [n_test, hidden_dim] test activations
        test_labels: [n_test] integer labels
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        patience: Early stopping patience (epochs without improvement)
        weight_decay: L2 regularization strength (default: 1.0)
        l1_lambda: L1 regularization strength (default: 0.0, disabled)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Results dict with probe, metrics, and training history

    Raises:
        ValueError: If data format invalid or shapes mismatch
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
            f"Train and test activations must have same hidden_dim: "
            f"{train_activations.shape[1]} vs {test_activations.shape[1]}"
        )

    n_classes = len(np.unique(train_labels))
    hidden_dim = train_activations.shape[1]

    print(f"Training multiclass probe:")
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Test samples: {len(test_labels)}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Classes: {n_classes}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Patience: {patience}")
    print(f"  Weight decay (L2): {weight_decay}")
    print(f"  L1 lambda: {l1_lambda}")

    # Convert to tensors
    train_acts_tensor = torch.from_numpy(train_activations).float()
    train_labels_tensor = torch.from_numpy(train_labels).long()
    test_acts_tensor = torch.from_numpy(test_activations).float()
    test_labels_tensor = torch.from_numpy(test_labels).long()

    # Create datasets and loaders
    train_dataset = torch.utils.data.TensorDataset(train_acts_tensor, train_labels_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_acts_tensor, test_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model (simple linear classifier)
    model = nn.Linear(hidden_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epoch": [],
    }

    # Early stopping state
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    # Training loop
    print("\nTraining...")
    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for acts, labels in train_loader:
            acts = acts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(acts)
            loss = criterion(logits, labels)

            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_reg = torch.tensor(0., requires_grad=True).to(device)
                for param in model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1)
                loss = loss + l1_lambda * l1_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
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

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
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
        if (epoch + 1) % 10 == 0 or epochs_without_improvement == 0:
            print(
                f"Epoch {epoch+1:3d}/{max_epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f} "
                f"{'🌟' if epochs_without_improvement == 0 else ''}"
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Final evaluation on both sets with best model
    model.eval()
    with torch.no_grad():
        # Train set - need to collect labels too since train_loader shuffles
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

        # Test set - need to collect labels too since test_loader may shuffle
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
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "weight_decay": weight_decay,
        },
        "train_predictions": train_preds,
        "test_predictions": test_preds,
    }

    print(f"\n{'='*60}")
    print(f"Final Results (best model from epoch {best_epoch+1}):")
    print(f"  Train accuracy: {final_train_acc:.4f}")
    print(f"  Test accuracy: {final_test_acc:.4f}")
    print(f"  Best test accuracy: {best_test_acc:.4f}")
    print(f"{'='*60}")

    return results
