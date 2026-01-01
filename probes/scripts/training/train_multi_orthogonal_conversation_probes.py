#!/usr/bin/env python3
"""Train K orthogonal user/assistant emotion probe pairs on conversation data.

Each of K sets contains:
- User emotion probes [6 emotions, hidden_dim]
- Assistant emotion probes [6 emotions, hidden_dim]

Orthogonality constraints:
1. Within each set k: user_k ⊥ assistant_k (user/assistant separation)
2. Between all sets: All 12K probes mutually orthogonal

This discovers multiple independent ways to represent the user/assistant emotion distinction.

Usage:
    # Soft orthogonality (loss-based)
    python train_multi_orthogonal_conversation_probes.py --layer 30 --n-sets 10 --ortho-weight 100000.0
    
    # Hard orthogonality (Gram-Schmidt)
    python train_multi_orthogonal_conversation_probes.py --layer 30 --n-sets 10 --ortho-weight 100000.0 --gram-schmidt
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


class MultiOrthogonalConversationProbes(nn.Module):
    """K orthogonal user/assistant emotion probe pairs."""

    def __init__(self, hidden_dim: int, n_emotions: int = 6, n_sets: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_emotions = n_emotions
        self.n_sets = n_sets

        # All probe pairs: [K, 2, n_emotions, hidden_dim]
        # dim 1: 0=user, 1=assistant
        self.probe_sets = nn.Parameter(torch.randn(n_sets, 2, n_emotions, hidden_dim))

        # Initialize
        for i in range(n_sets):
            nn.init.xavier_normal_(self.probe_sets[i, 0], gain=0.1)  # user
            nn.init.xavier_normal_(self.probe_sets[i, 1], gain=0.1)  # assistant

        # Normalize to unit vectors
        with torch.no_grad():
            self.probe_sets.data = F.normalize(self.probe_sets.data, dim=3)

    def forward(self, activations: torch.Tensor, set_idx: int, probe_type: str) -> torch.Tensor:
        """Project activations onto probe directions.
        
        Args:
            activations: [batch, hidden_dim]
            set_idx: Which probe set (0 to K-1)
            probe_type: 'user' or 'assistant'
            
        Returns:
            [batch, n_emotions] logits
        """
        type_idx = 0 if probe_type == 'user' else 1
        probes = self.probe_sets[set_idx, type_idx]  # [n_emotions, hidden_dim]
        probes_norm = F.normalize(probes, dim=1)
        return activations @ probes_norm.T

    def orthogonality_loss(self):
        """All-pairs orthogonality loss.
        
        Enforces:
        1. Within each set: user ⊥ assistant
        2. Between sets: all probes mutually orthogonal
        """
        # Flatten to [K*2*n_emotions, hidden_dim]
        n_total = self.n_sets * 2 * self.n_emotions
        probes_flat = self.probe_sets.reshape(n_total, self.hidden_dim)
        probes_norm = F.normalize(probes_flat, dim=1)

        # Compute all pairwise dot products
        dots = probes_norm @ probes_norm.T  # [K*2*n_emotions, K*2*n_emotions]

        # Mask diagonal (self-dots should be 1)
        mask = ~torch.eye(n_total, dtype=torch.bool, device=dots.device)
        off_diag_dots = dots[mask]

        # Penalize all off-diagonal dot products
        return (off_diag_dots ** 2).mean()

    def gram_schmidt_orthogonalize(self):
        """Apply Gram-Schmidt orthogonalization to all probes."""
        with torch.no_grad():
            n_total = self.n_sets * 2 * self.n_emotions
            probes_flat = self.probe_sets.data.reshape(n_total, self.hidden_dim)

            orthogonal_probes = torch.zeros_like(probes_flat)
            for i in range(n_total):
                vec = probes_flat[i].clone()
                # Subtract projections onto all previous vectors
                for j in range(i):
                    projection = torch.dot(vec, orthogonal_probes[j]) * orthogonal_probes[j]
                    vec = vec - projection

                # Normalize
                vec_norm = vec.norm()
                if vec_norm > 1e-8:
                    orthogonal_probes[i] = vec / vec_norm
                else:
                    # Random restart if zero
                    orthogonal_probes[i] = torch.randn_like(vec)
                    for j in range(i):
                        projection = torch.dot(orthogonal_probes[i], orthogonal_probes[j]) * orthogonal_probes[j]
                        orthogonal_probes[i] = orthogonal_probes[i] - projection
                    orthogonal_probes[i] = F.normalize(orthogonal_probes[i], dim=0)

            # Reshape back
            self.probe_sets.data = orthogonal_probes.reshape(self.n_sets, 2, self.n_emotions, self.hidden_dim)


class EmotionDataset(Dataset):
    def __init__(self, activations, user_labels, asst_labels):
        self.activations = activations
        self.user_labels = user_labels
        self.asst_labels = asst_labels

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return {
            'activation': self.activations[idx],
            'user_label': self.user_labels[idx],
            'asst_label': self.asst_labels[idx],
        }


def load_conversation_data(h5_path: Path, layer: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load conversation activations and dual labels."""
    print(f"Loading conversation data from {h5_path}")
    print(f"Layer: {layer}")

    emotion_to_idx = {e: i for i, e in enumerate(EMOTIONS)}

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

            if user_emotion not in EMOTIONS or asst_emotion not in EMOTIONS:
                continue

            conv_group = acts_group[conv_id]

            # Use global mean emotional activations
            # emotional is a 2D dataset: [n_layers, hidden_dim]
            activation = conv_group["emotional"][layer]

            all_activations.append(activation)
            all_user_labels.append(emotion_to_idx[user_emotion])
            all_asst_labels.append(emotion_to_idx[asst_emotion])

    activations = np.array(all_activations)
    user_labels = np.array(all_user_labels)
    asst_labels = np.array(all_asst_labels)

    print(f"Loaded {len(activations)} samples")
    print(f"Activation shape: {activations.shape}")

    # Balance dataset
    unique_user, counts = np.unique(user_labels, return_counts=True)
    min_count = counts.min()
    print(f"\nAuto-balancing to {min_count} samples per class")

    balanced_indices = []
    for emotion_idx in range(len(EMOTIONS)):
        indices = np.where(user_labels == emotion_idx)[0]
        selected = np.random.choice(indices, min_count, replace=False)
        balanced_indices.extend(selected)

    activations = activations[balanced_indices]
    user_labels = user_labels[balanced_indices]
    asst_labels = asst_labels[balanced_indices]

    print(f"Balanced dataset: {len(activations)} samples\n")

    return activations, user_labels, asst_labels


def train_epoch(
    model: MultiOrthogonalConversationProbes,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ortho_weight: float,
    device: str,
    use_gram_schmidt: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_ortho_loss = 0
    
    # Track accuracy for each set
    user_correct = [0] * model.n_sets
    asst_correct = [0] * model.n_sets
    total_samples = 0

    for batch in dataloader:
        activations = batch['activation'].to(device)
        user_labels = batch['user_label'].to(device)
        asst_labels = batch['asst_label'].to(device)

        optimizer.zero_grad()

        # Task loss: Average over all K sets
        task_loss = 0
        for k in range(model.n_sets):
            user_logits = model(activations, k, 'user')
            asst_logits = model(activations, k, 'assistant')
            
            task_loss += F.cross_entropy(user_logits, user_labels)
            task_loss += F.cross_entropy(asst_logits, asst_labels)
            
            # Track accuracy
            user_pred = user_logits.argmax(dim=1)
            asst_pred = asst_logits.argmax(dim=1)
            user_correct[k] += (user_pred == user_labels).sum().item()
            asst_correct[k] += (asst_pred == asst_labels).sum().item()

        task_loss = task_loss / (2 * model.n_sets)

        # Orthogonality loss
        ortho_loss = model.orthogonality_loss()

        # Combined loss
        loss = task_loss + ortho_weight * ortho_loss

        loss.backward()
        optimizer.step()

        # Normalize to unit sphere
        with torch.no_grad():
            model.probe_sets.data = F.normalize(model.probe_sets.data, dim=3)

        # Apply Gram-Schmidt if requested
        if use_gram_schmidt:
            model.gram_schmidt_orthogonalize()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_ortho_loss += ortho_loss.item()
        total_samples += len(activations)

    # Compute mean accuracies
    user_accs = [c / total_samples for c in user_correct]
    asst_accs = [c / total_samples for c in asst_correct]
    mean_user_acc = np.mean(user_accs)
    mean_asst_acc = np.mean(asst_accs)

    return {
        'loss': total_loss / len(dataloader),
        'task_loss': total_task_loss / len(dataloader),
        'ortho_loss': total_ortho_loss / len(dataloader),
        'mean_user_acc': mean_user_acc,
        'mean_asst_acc': mean_asst_acc,
        'user_accs': user_accs,
        'asst_accs': asst_accs,
    }


def evaluate(
    model: MultiOrthogonalConversationProbes,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    user_correct = [0] * model.n_sets
    asst_correct = [0] * model.n_sets
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            activations = batch['activation'].to(device)
            user_labels = batch['user_label'].to(device)
            asst_labels = batch['asst_label'].to(device)

            for k in range(model.n_sets):
                user_logits = model(activations, k, 'user')
                asst_logits = model(activations, k, 'assistant')

                user_pred = user_logits.argmax(dim=1)
                asst_pred = asst_logits.argmax(dim=1)
                
                user_correct[k] += (user_pred == user_labels).sum().item()
                asst_correct[k] += (asst_pred == asst_labels).sum().item()

            total_samples += len(activations)

    user_accs = [c / total_samples for c in user_correct]
    asst_accs = [c / total_samples for c in asst_correct]

    return {
        'mean_user_acc': np.mean(user_accs),
        'mean_asst_acc': np.mean(asst_accs),
        'user_accs': user_accs,
        'asst_accs': asst_accs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--n-sets', type=int, default=10, help='Number of orthogonal pairs (K)')
    parser.add_argument('--ortho-weight', type=float, default=100000.0)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gram-schmidt', action='store_true', help='Use Gram-Schmidt re-orthogonalization')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load data
    h5_path = Path('/workspace-vast/annas/git/research-tools/outputs/data/activations/conversations2_combined.h5')
    activations, user_labels, asst_labels = load_conversation_data(h5_path, args.layer)

    # Split
    indices = np.arange(len(activations))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=args.seed, stratify=user_labels)

    train_acts = torch.FloatTensor(activations[train_idx])
    test_acts = torch.FloatTensor(activations[test_idx])
    train_user_labels = torch.LongTensor(user_labels[train_idx])
    test_user_labels = torch.LongTensor(user_labels[test_idx])
    train_asst_labels = torch.LongTensor(asst_labels[train_idx])
    test_asst_labels = torch.LongTensor(asst_labels[test_idx])

    train_dataset = EmotionDataset(train_acts, train_user_labels, train_asst_labels)
    test_dataset = EmotionDataset(test_acts, test_user_labels, test_asst_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    print("=" * 80)
    print(f"Training with K={args.n_sets} probe pairs, ortho_weight={args.ortho_weight}")
    if args.gram_schmidt:
        print("Using Gram-Schmidt re-orthogonalization")
    print("=" * 80)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    hidden_dim = activations.shape[1]
    model = MultiOrthogonalConversationProbes(hidden_dim, n_emotions=6, n_sets=args.n_sets).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_acc = 0
    patience_counter = 0
    history = []

    for epoch in range(args.max_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args.ortho_weight, args.device, args.gram_schmidt)
        test_metrics = evaluate(model, test_loader, args.device)

        mean_acc = (test_metrics['mean_user_acc'] + test_metrics['mean_asst_acc']) / 2

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'test': test_metrics,
        })

        print(f"Epoch {epoch:3d} | Loss: {train_metrics['loss']:.4f} "
              f"(task={train_metrics['task_loss']:.4f}, ortho={train_metrics['ortho_loss']:.4f}) | "
              f"Test Acc: user={test_metrics['mean_user_acc']:.4f}, asst={test_metrics['mean_asst_acc']:.4f}")

        # Early stopping
        if mean_acc > best_acc:
            best_acc = mean_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    test_metrics = evaluate(model, test_loader, args.device)
    print(f"Mean user accuracy: {test_metrics['mean_user_acc']:.4f}")
    print(f"Mean assistant accuracy: {test_metrics['mean_asst_acc']:.4f}")
    print(f"User accuracies: {[f'{a:.4f}' for a in test_metrics['user_accs']]}")
    print(f"Assistant accuracies: {[f'{a:.4f}' for a in test_metrics['asst_accs']]}")

    # Check orthogonality
    with torch.no_grad():
        ortho_loss = model.orthogonality_loss().item()
        print(f"\nOrthogonality (mean squared dot): {ortho_loss:.6f}")

    # Save
    output_dir = Path('/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/multi_orthogonal')
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = '_gramschmidt' if args.gram_schmidt else ''
    output_path = output_dir / f'probe_k{args.n_sets}_layer{args.layer}_ortho{args.ortho_weight}{suffix}.pkl'

    result = {
        'model_state': model.state_dict(),
        'probe_sets': model.probe_sets.detach().cpu().numpy(),  # [K, 2, 6, hidden_dim]
        'label_names': EMOTIONS,
        'test_metrics': test_metrics,
        'history': history,
        'args': vars(args),
    }

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"\nSaved results to {output_path}")


if __name__ == '__main__':
    main()
