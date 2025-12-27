#!/usr/bin/env python3
"""Train all configurations (all dims × all seeds) for a single layer.

This is much more efficient than one job per probe because:
- Model loaded once and reused
- Data loaded once per dim
- Reduces SLURM overhead from 2480 jobs → 62 jobs

Usage:
    python train_layer_all_configs.py --layer 20
"""

import argparse
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def load_texts_data(h5_path, layer, cpca_components=None, n_components=None, seed=42):
    """Load emotion data from texts.h5format."""
    with h5py.File(h5_path, 'r') as f:
        acts_group = f['activations']

        # Emotion mapping (6 emotions + neutral)
        emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
        emotion_map = {e: i for i, e in enumerate(emotions)}

        all_acts = []
        all_labels = []

        # First collect emotional activations
        for key in acts_group.keys():
            parts = key.split('_')
            if len(parts) < 4:
                continue

            emotion = parts[-1]
            if emotion not in emotions[:-1]:  # Exclude neutral
                continue

            # Load emotional activations
            act = acts_group[key]['emotional'][layer]
            all_acts.append(act)
            all_labels.append(emotion_map[emotion])

        # Now add neutral activations
        for key in acts_group.keys():
            parts = key.split('_')
            if len(parts) < 4:
                continue

            emotion = parts[-1]
            if emotion not in emotions[:-1]:  # Only emotions we care about
                continue

            # Load neutral activations
            act = acts_group[key]['neutral'][layer]
            all_acts.append(act)
            all_labels.append(emotion_map["neutral"])

        X = np.stack(all_acts)
        y = np.array(all_labels)

    # Apply cPCA if provided
    if cpca_components is not None:
        if n_components and n_components > 0:
            cpca_components = cpca_components[:n_components]
        X = X @ cpca_components.T

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test, emotions


def train_probe(X_train, y_train, X_test, y_test, seed=42):
    """Train linear probe."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    model = nn.Linear(n_features, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    n_epochs = 100
    batch_size = 32
    best_test_acc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(n_epochs):
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

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            test_acc = (test_outputs.argmax(dim=1) == y_test_t).sum().item() / len(y_test_t)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return model, history, best_test_acc, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--data", type=str, default="data/activations/texts_combined.h5")
    parser.add_argument("--cpca-results", type=str,
                       default="probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz")
    parser.add_argument("--output-dir", type=str, default="results/emotion_probes_multiseed")
    parser.add_argument("--n-components-list", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)

    print("=" * 80)
    print(f"TRAINING ALL CONFIGS FOR LAYER {args.layer}")
    print("=" * 80)
    print(f"Dimensionalities: {args.n_components_list}")
    print(f"Seeds: {args.seeds}")
    print(f"Total probes to train: {len(args.n_components_list) * len(args.seeds)}")
    print()

    # Load cPCA components once
    cpca_components_by_layer = None
    if any(nc > 0 for nc in args.n_components_list):
        cpca_path = Path(args.cpca_results)
        print(f"Loading cPCA components from {cpca_path}")
        cpca_data = np.load(cpca_path)
        all_components = cpca_data['components']  # [n_layers, 50, hidden_dim]
        if args.layer < all_components.shape[0]:
            cpca_components_by_layer = all_components[args.layer]  # [50, hidden_dim]
        else:
            print(f"Error: Layer {args.layer} not in cPCA results")
            return

    # Train all configs
    for n_comp in args.n_components_list:
        print("\n" + "=" * 80)
        print(f"DIMENSIONALITY: {'Raw' if n_comp == 0 else f'{n_comp} PCs'}")
        print("=" * 80)

        for seed in args.seeds:
            probe_name = f"probe_layer{args.layer}_nc{n_comp}_seed{seed}.pkl"
            probe_path = output_dir / probe_name

            # Skip if already exists
            if probe_path.exists():
                print(f"  Seed {seed}: Already exists, skipping")
                continue

            print(f"  Seed {seed}: Training...")

            try:
                # Load data
                cpca_comp = cpca_components_by_layer if n_comp > 0 else None
                n_comp_to_use = n_comp if n_comp > 0 else None

                X_train, X_test, y_train, y_test, label_names = load_texts_data(
                    data_path,
                    args.layer,
                    cpca_components=cpca_comp,
                    n_components=n_comp_to_use,
                    seed=seed
                )

                # Train probe
                model, history, best_test_acc, best_epoch = train_probe(
                    X_train, y_train, X_test, y_test, seed=seed
                )

                # Get final predictions
                import torch
                model.eval()
                with torch.no_grad():
                    train_preds = model(torch.from_numpy(X_train).float()).argmax(dim=1).numpy()
                    test_preds = model(torch.from_numpy(X_test).float()).argmax(dim=1).numpy()

                train_acc = (train_preds == y_train).mean()
                test_acc = (test_preds == y_test).mean()

                # Save probe
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
                        'n_components': n_comp,
                        'seed': seed,
                        'data_path': str(data_path),
                    },
                    'train_predictions': train_preds,
                    'test_predictions': test_preds,
                    'label_names': label_names,
                    'use_cpca': n_comp > 0,
                    'cpca_results_path': args.cpca_results if n_comp > 0 else None,
                }

                with open(probe_path, 'wb') as f:
                    pickle.dump(probe_data, f)

                print(f"    ✓ Saved: test_acc={test_acc:.3f}, best={best_test_acc:.3f}")

            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue

    print("\n" + "=" * 80)
    print(f"LAYER {args.layer} COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
