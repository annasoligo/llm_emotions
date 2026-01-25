#!/usr/bin/env python3
"""
Evaluate orthogonal regularized probes trained on controlled variation data
on the conversation dataset.

This tests how well the source-specific probes generalize to real conversation data.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


def load_conversation_data(
    data_path: str,
    layer: int,
    emotion_type: str = "user",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load conversation data for a specific layer.

    Args:
        data_path: Path to conversations2.h5
        layer: Layer index to extract
        emotion_type: "user" or "asst" - which emotion to predict

    Returns:
        activations: (n_samples, hidden_size)
        labels: (n_samples,) integer labels
        label_names: List of emotion names
    """
    with h5py.File(data_path, 'r') as f:
        # Load metadata
        metadata_json = f['metadata'][()]
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode('utf-8')
        metadata = json.loads(metadata_json)

        # Get emotion field
        emotion_field = f"{emotion_type}_emotion"

        # Extract activations and labels
        activations_list = []
        emotions = []

        for entry in metadata:
            sample_id = entry['id']
            emotion = entry[emotion_field]

            # Load activation for this sample at specified layer
            activation = f[f'activations/{sample_id}/global'][layer]  # shape: (5376,)

            activations_list.append(activation)
            emotions.append(emotion)

        # Convert to numpy arrays
        activations = np.stack(activations_list, axis=0)  # (n_samples, 5376)

        # Create label mapping
        unique_emotions = sorted(set(emotions))
        emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}
        labels = np.array([emotion_to_idx[e] for e in emotions])

        print(f"Loaded {len(activations)} samples from conversation data")
        print(f"Predicting {emotion_type} emotion")
        print(f"Emotions: {unique_emotions}")
        print(f"Activations shape: {activations.shape}")

        return activations, labels, unique_emotions


def load_trained_probe(model_path: str, device: str = 'cuda') -> nn.Module:
    """Load a trained probe model."""
    checkpoint = torch.load(model_path, map_location=device)

    # Get model config from saved state dict
    hidden_dim = checkpoint['linear.weight'].shape[1]
    num_classes = checkpoint['linear.weight'].shape[0]

    # Get orthogonal PCs if saved
    orthogonal_pcs = checkpoint.get('orthogonal_pcs', None)
    if orthogonal_pcs is not None:
        orthogonal_pcs = orthogonal_pcs.cpu().numpy()

    # Create model
    model = SourceSpecificEmotionProbe(
        hidden_dim=hidden_dim,
        num_emotions=num_classes,
        orthogonal_pcs=orthogonal_pcs,
    )

    # Load weights
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    print(f"Loaded probe: hidden_dim={hidden_dim}, num_classes={num_classes}")
    if orthogonal_pcs is not None:
        print(f"Orthogonal PCs shape: {orthogonal_pcs.shape}")

    return model


def evaluate_probe(
    model: nn.Module,
    activations: np.ndarray,
    labels: np.ndarray,
    label_names: List[str],
    device: str = 'cuda',
) -> Dict:
    """Evaluate probe on data."""
    # Convert to tensors
    X = torch.FloatTensor(activations).to(device)
    y = torch.LongTensor(labels).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(X)
        predictions = torch.argmax(logits, dim=1)

    # Convert back to numpy
    predictions = predictions.cpu().numpy()
    labels_np = labels

    # Calculate metrics
    accuracy = accuracy_score(labels_np, predictions)

    # Classification report
    report = classification_report(
        labels_np,
        predictions,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(labels_np, predictions)

    results = {
        'accuracy': float(accuracy),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'label_names': label_names,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate controlled variation probes on conversation data"
    )
    parser.add_argument(
        "--probe-path",
        type=str,
        required=True,
        help="Path to trained probe (model.pt file)",
    )
    parser.add_argument(
        "--conversation-data",
        type=str,
        default="outputs/data/activations/conversations2.h5",
        help="Path to conversation activation data",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=30,
        help="Layer to evaluate",
    )
    parser.add_argument(
        "--emotion-type",
        type=str,
        choices=["user", "asst"],
        default="user",
        help="Which emotion to predict (user or assistant)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("EVALUATING CONTROLLED VARIATION PROBE ON CONVERSATION DATA")
    print("=" * 80)
    print(f"Probe path: {args.probe_path}")
    print(f"Conversation data: {args.conversation_data}")
    print(f"Layer: {args.layer}")
    print(f"Emotion type: {args.emotion_type}")
    print(f"Device: {args.device}")
    print()

    # Load conversation data
    print("Loading conversation data...")
    activations, labels, label_names = load_conversation_data(
        args.conversation_data,
        args.layer,
        args.emotion_type,
    )
    print()

    # Load trained probe
    print("Loading trained probe...")
    model = load_trained_probe(args.probe_path, args.device)
    print()

    # Evaluate
    print("Evaluating...")
    results = evaluate_probe(model, activations, labels, label_names, args.device)
    print()

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print()
    print("Classification Report:")
    print(results['classification_report'])
    print()

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])

    # Header
    print(f"{'':12}", end='')
    for name in label_names:
        print(f"{name[:10]:>10}", end='')
    print()

    # Rows
    for i, name in enumerate(label_names):
        print(f"{name[:12]:12}", end='')
        for j in range(len(label_names)):
            print(f"{cm[i, j]:10d}", end='')
        print()

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
