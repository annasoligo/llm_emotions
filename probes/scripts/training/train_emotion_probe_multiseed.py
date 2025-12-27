#!/usr/bin/env python3
"""Train emotion probes with multiple seeds for statistical robustness.

Usage:
    # Raw activations
    python train_emotion_probe_multiseed.py --layer 20 --n-components 0 --seed 42

    # With cPCA reduction
    python train_emotion_probe_multiseed.py --layer 20 --n-components 10 --seed 42
"""

import argparse
import pickle
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from probes.core import load_cpca_results


def load_emotion_data(
    h5_path: Path,
    layer: int,
    cpca_components: Optional[np.ndarray] = None,
    n_components: Optional[int] = None,
    seed: int = 42,
):
    """Load emotion data from texts_combined.h5."""
    print(f"Loading data from {h5_path} at layer {layer}")

    with h5py.File(h5_path, "r") as f:
        activations_group = f["activations"]

        # Emotion mapping (matching your existing setup)
        emotion_names = ["angry", "bored", "calm", "excited", "happy", "sad", "baseline"]
        emotion_to_idx = {e: i for i, e in enumerate(emotion_names)}

        all_acts = []
        all_labels = []

        # Collect activations from all conversations
        for conv_key in activations_group.keys():
            conv_group = activations_group[conv_key]

            if "user_emotion" not in conv_group.attrs or "asst_emotion" not in conv_group.attrs:
                continue

            user_emotion = conv_group.attrs["user_emotion"]
            asst_emotion = conv_group.attrs["asst_emotion"]

            # Get layer activations
            if f"layer_{layer}" not in conv_group:
                continue

            layer_acts = conv_group[f"layer_{layer}"][:]  # [n_tokens, hidden_dim]

            # Average across tokens
            mean_act = layer_acts.mean(axis=0)  # [hidden_dim]

            # Add user emotion sample
            if user_emotion in emotion_to_idx:
                all_acts.append(mean_act)
                all_labels.append(emotion_to_idx[user_emotion])

            # Add assistant emotion sample (if different conversation, could be from turns)
            # For now, we're using the same activation for both - you may want to use turn-specific acts
            if asst_emotion in emotion_to_idx:
                all_acts.append(mean_act)
                all_labels.append(emotion_to_idx[asst_emotion])

        X = np.stack(all_acts)  # [n_samples, hidden_dim]
        y = np.array(all_labels)  # [n_samples]

    print(f"Loaded {len(X)} samples across {len(emotion_names)} emotions")

    # Apply cPCA projection if provided
    if cpca_components is not None:
        if n_components is not None and n_components > 0:
            # Use only top n_components
            cpca_components = cpca_components[:n_components]  # [n_components, hidden_dim]

        print(f"Projecting onto {cpca_components.shape[0]} cPCA components")
        X = X @ cpca_components.T  # [n_samples, n_components]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test, emotion_names


def train_probe(X_train, y_train, X_test, y_test, seed=42):
    """Train a simple linear probe with PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Create model
    model = nn.Linear(n_features, n_classes)

    # Prepare data
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = 100
    batch_size = 32
    best_test_acc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    for epoch in range(n_epochs):
        # Training
        model.train()
        perm = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_correct = 0

        for i in range(0, len(X_train_t), batch_size):
            indices = perm[i:i+batch_size]
            batch_X = X_train_t[indices]
            batch_y = y_train_t[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_X)
            n_correct += (outputs.argmax(dim=1) == batch_y).sum().item()

        train_loss = epoch_loss / len(X_train_t)
        train_acc = n_correct / len(X_train_t)

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            test_acc = (test_outputs.argmax(dim=1) == y_test_t).sum().item() / len(y_test_t)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Early stopping
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

    print(f"Best test accuracy: {best_test_acc:.3f} at epoch {best_epoch}")

    return model, history, best_test_acc, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/activations/texts_combined.h5")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--n-components", type=int, default=0,
                       help="Number of PCs (0 = raw, 5/10/20 for cPCA)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cpca-results", type=str,
                       default="probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz")
    parser.add_argument("--output-dir", type=str, default="results/emotion_probes_multiseed")
    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cPCA components if needed
    cpca_components = None
    if args.n_components > 0:
        cpca_path = Path(args.cpca_results)
        print(f"Loading cPCA components from {cpca_path}")
        cpca_data = np.load(cpca_path)
        all_components = cpca_data['components']  # [n_layers, 50, hidden_dim]

        if args.layer < all_components.shape[0]:
            cpca_components = all_components[args.layer]  # [50, hidden_dim]
        else:
            print(f"Error: Layer {args.layer} not found in cPCA results")
            return

    # Load data
    X_train, X_test, y_train, y_test, label_names = load_emotion_data(
        data_path,
        args.layer,
        cpca_components=cpca_components,
        n_components=args.n_components if args.n_components > 0 else None,
        seed=args.seed
    )

    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    # Train probe
    model, history, best_test_acc, best_epoch = train_probe(
        X_train, y_train, X_test, y_test, seed=args.seed
    )

    # Get final predictions
    model.eval()
    with torch.no_grad():
        train_preds = model(torch.from_numpy(X_train).float()).argmax(dim=1).numpy()
        test_preds = model(torch.from_numpy(X_test).float()).argmax(dim=1).numpy()

    train_acc = (train_preds == y_train).mean()
    test_acc = (test_preds == y_test).mean()

    # Save probe
    probe_name = f"probe_layer{args.layer}_nc{args.n_components}_seed{args.seed}.pkl"
    probe_path = output_dir / probe_name

    probe_data = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'best_test_accuracy': best_test_acc,
        'best_epoch': best_epoch,
        'total_epochs': len(history['train_acc']),
        'history': history,
        'config': {
            'layer': args.layer,
            'n_components': args.n_components,
            'seed': args.seed,
            'data_path': str(data_path),
        },
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'label_names': label_names,
        'use_cpca': args.n_components > 0,
        'cpca_results_path': args.cpca_results if args.n_components > 0 else None,
    }

    with open(probe_path, 'wb') as f:
        pickle.dump(probe_data, f)

    print(f"\nProbe saved to {probe_path}")
    print(f"Final train accuracy: {train_acc:.3f}")
    print(f"Final test accuracy: {test_acc:.3f}")
    print(f"Best test accuracy: {best_test_acc:.3f}")


if __name__ == "__main__":
    main()
