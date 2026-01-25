#!/usr/bin/env python3
"""
Evaluate cross-source accuracy and cosine similarity for diverse isolation probes.

Cross-source accuracy: Train on user emotion, test on assistant emotion (and vice versa)
Cosine similarity: Measure alignment between user and assistant probe weight vectors
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset


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


def load_trained_probe(model_path: str, device: str = 'cuda') -> nn.Module:
    """Load a trained probe from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get model config from saved state dict
    hidden_dim = checkpoint['linear.weight'].shape[1]
    num_classes = checkpoint['linear.weight'].shape[0]

    # Get orthogonal PCs if saved
    orthogonal_pcs = checkpoint.get('orthogonal_pcs', None)
    if orthogonal_pcs is not None:
        orthogonal_pcs = orthogonal_pcs.cpu().numpy()

    model = SourceSpecificEmotionProbe(hidden_dim, num_classes, orthogonal_pcs)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    return model


def evaluate_cross_source(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
) -> float:
    """Evaluate probe on opposite-source test set."""
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
    return accuracy


def compute_cosine_similarity(model1: nn.Module, model2: nn.Module) -> float:
    """Compute cosine similarity between two probe weight vectors."""
    # Get weight matrices (num_classes, hidden_dim)
    w1 = model1.linear.weight.detach().cpu().numpy()
    w2 = model2.linear.weight.detach().cpu().numpy()

    # Flatten to vectors
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()

    # Compute cosine similarity
    cos_sim = np.dot(w1_flat, w2_flat) / (np.linalg.norm(w1_flat) * np.linalg.norm(w2_flat))

    return float(cos_sim)


def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-source accuracy and cosine similarity")
    parser.add_argument("--layers", type=int, nargs='+', default=[0, 10, 20, 30, 40, 50, 60])
    parser.add_argument("--lambdas", type=float, nargs='+', default=[0, 10, 100])
    parser.add_argument("--output", type=str, default="outputs/probes/diverse_isolation")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("CROSS-SOURCE EVALUATION AND COSINE SIMILARITY")
    print("=" * 80)
    print()

    results = []

    for layer in args.layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer}")
        print(f"{'='*80}")

        # Load test datasets
        user_data_path = "outputs/activations/diverse_isolation/user_isolation.h5"
        asst_data_path = "outputs/activations/diverse_isolation/assistant_isolation.h5"

        user_dataset = DiverseIsolationDataset(user_data_path, layer)
        asst_dataset = DiverseIsolationDataset(asst_data_path, layer)

        user_loader = DataLoader(user_dataset, batch_size=args.batch_size, shuffle=False)
        asst_loader = DataLoader(asst_dataset, batch_size=args.batch_size, shuffle=False)

        for lambda_reg in args.lambdas:
            print(f"\n--- Lambda={lambda_reg} ---")

            # Load user and assistant probes (use float format for directory names)
            user_model_path = Path(args.output) / f"user_layer{layer}_lambda{float(lambda_reg)}" / "model.pt"
            asst_model_path = Path(args.output) / f"assistant_layer{layer}_lambda{float(lambda_reg)}" / "model.pt"

            if not user_model_path.exists() or not asst_model_path.exists():
                print(f"  ⚠ Models not found, skipping")
                continue

            user_model = load_trained_probe(str(user_model_path), device)
            asst_model = load_trained_probe(str(asst_model_path), device)

            # Evaluate cross-source accuracy
            # User probe on assistant data
            user_on_asst = evaluate_cross_source(user_model, asst_loader, device)

            # Assistant probe on user data
            asst_on_user = evaluate_cross_source(asst_model, user_loader, device)

            # Compute cosine similarity
            cos_sim = compute_cosine_similarity(user_model, asst_model)

            print(f"  User probe on assistant data: {user_on_asst:.4f}")
            print(f"  Assistant probe on user data: {asst_on_user:.4f}")
            print(f"  Cosine similarity: {cos_sim:.4f}")

            results.append({
                'layer': layer,
                'lambda': lambda_reg,
                'user_on_asst': user_on_asst,
                'asst_on_user': asst_on_user,
                'cos_sim': cos_sim,
            })

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'Layer':<6} | {'Lambda':<8} | {'U→A':<8} | {'A→U':<8} | {'CosSim':<8}")
    print(f"{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for r in results:
        print(f"{r['layer']:<6} | {r['lambda']:<8.1f} | {r['user_on_asst']:<8.4f} | {r['asst_on_user']:<8.4f} | {r['cos_sim']:<8.4f}")

    # Save results
    output_file = Path(args.output) / "cross_source_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Find configurations with lowest cross-source accuracy
    sorted_by_cross = sorted(results, key=lambda x: (x['user_on_asst'] + x['asst_on_user']) / 2)

    print("\nMost source-specific (lowest cross-source accuracy):")
    for r in sorted_by_cross[:5]:
        avg_cross = (r['user_on_asst'] + r['asst_on_user']) / 2
        print(f"  Layer {r['layer']}, λ={r['lambda']}: avg={avg_cross:.4f}, cos_sim={r['cos_sim']:.4f}")

    # Find configurations with lowest cosine similarity
    sorted_by_cos = sorted(results, key=lambda x: x['cos_sim'])

    print("\nMost orthogonal weights (lowest cosine similarity):")
    for r in sorted_by_cos[:5]:
        avg_cross = (r['user_on_asst'] + r['asst_on_user']) / 2
        print(f"  Layer {r['layer']}, λ={r['lambda']}: cos_sim={r['cos_sim']:.4f}, cross_acc={avg_cross:.4f}")


if __name__ == "__main__":
    main()
