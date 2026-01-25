"""
Compute per-layer probe baselines from WildChat data.
Saves baselines to JSON for use in layerwise plotting.

Directly loads h5 activations and applies probes without using the complex infrastructure.
"""
import json
import pickle
import numpy as np
import h5py
from pathlib import Path

# Config
BASELINE_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/wildchat/google_gemma_3_27b_it')
PROBE_BASE = Path('/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed')
CPCA_PATH = Path('/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz')
OUTPUT_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/probe_baselines')
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
ALL_LAYERS = list(range(0, 62))


def load_probe(probe_path: Path):
    """Load a probe pickle file and extract the Linear model."""
    import torch
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)

    # The probe is stored as a PyTorch Linear layer in 'model' key
    model = probe_data['model']
    sd = model.state_dict()

    return {
        'weight': sd['weight'].numpy(),  # [n_classes, hidden_dim]
        'bias': sd['bias'].numpy()        # [n_classes]
    }


def apply_probe(activations: np.ndarray, probe: dict, apply_softmax: bool = True) -> np.ndarray:
    """
    Apply a linear probe to activations.

    Args:
        activations: [n_samples, hidden_dim]
        probe: dict with 'weight' [n_classes, hidden_dim] and 'bias' [n_classes]
        apply_softmax: If True, apply softmax to get probabilities (matches preprocessing)

    Returns:
        scores: [n_samples, n_classes] - probabilities if apply_softmax=True, else raw logits
    """
    weight = probe['weight']  # [n_classes, hidden_dim]
    bias = probe['bias']      # [n_classes]

    # Linear: logits = activations @ weight.T + bias
    logits = activations @ weight.T + bias

    if apply_softmax:
        # Apply softmax to match preprocessing (which stores probabilities)
        # softmax(x) = exp(x) / sum(exp(x))
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))  # Numerical stability
        scores = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    else:
        scores = logits

    return scores


def compute_baselines_for_probe(probe_pattern: str, n_components: int = 0):
    """Compute per-layer baselines for a probe type."""

    # Load cPCA components if needed
    # cPCA components are per-layer: shape (num_layers, n_components, hidden_dim)
    cpca_components_all = None
    if n_components > 0:
        if CPCA_PATH.exists():
            cpca_data = np.load(CPCA_PATH)
            # Shape: (62, 50, 5376) = (num_layers, max_components, hidden_dim)
            cpca_components_all = cpca_data['components'][:, :n_components, :]
            print(f"  Loaded cPCA components: {cpca_components_all.shape}")
        else:
            print(f"  WARNING: cPCA path not found: {CPCA_PATH}")
            return {}

    baselines_per_layer = {}

    for layer in ALL_LAYERS:
        try:
            # Load baseline activations for this layer
            h5_path = BASELINE_DIR / f'layer{layer}_activations.h5'
            if not h5_path.exists():
                print(f"  Layer {layer}: SKIP (no h5 file)")
                continue

            with h5py.File(h5_path, 'r') as f:
                # Use assistant_turn activations
                activations = f['assistant_turn'][:]  # [n_samples, hidden_dim]

            # Apply cPCA if needed - use THIS layer's components
            if cpca_components_all is not None:
                # cpca_components_all[layer] shape: (n_components, hidden_dim)
                layer_cpca = cpca_components_all[layer]  # [n_components, hidden_dim]
                activations = activations @ layer_cpca.T  # [n_samples, n_components]

            # Load probe for this layer
            probe_file = probe_pattern.format(layer=layer)
            probe_path = PROBE_BASE / probe_file

            if not probe_path.exists():
                print(f"  Layer {layer}: SKIP (no probe file)")
                continue

            probe = load_probe(probe_path)

            # Apply probe - use raw logits (not softmax) for z-score normalization
            scores = apply_probe(activations, probe, apply_softmax=False)  # [n_samples, 7]
            # Exclude neutral (index 6) - only keep 6 emotions
            scores = scores[:, :6]  # [n_samples, 6]

            # Compute mean and std
            baselines_per_layer[layer] = {
                'mean': scores.mean(axis=0).tolist(),
                'std': scores.std(axis=0).tolist()
            }

            if layer % 10 == 0:
                print(f"  Layer {layer}: mean={baselines_per_layer[layer]['mean'][:2]}... (n={len(activations)})")

        except Exception as e:
            print(f"  Layer {layer}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    return baselines_per_layer


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute baselines for text_raw (nc=0, seed=0)
    print("\n" + "="*60)
    print("Computing baselines for text_raw (nc=0)")
    print("="*60)

    text_raw_baselines = compute_baselines_for_probe(
        probe_pattern='probe_layer{layer}_nc0_seed0.pkl',
        n_components=0
    )

    output_file = OUTPUT_DIR / 'text_raw_baselines.json'
    with open(output_file, 'w') as f:
        json.dump({
            'probe_type': 'text_raw',
            'n_components': 0,
            'seed': 0,
            'emotions': EMOTIONS,
            'baselines': {str(k): v for k, v in text_raw_baselines.items()}
        }, f, indent=2)
    print(f"\nSaved to {output_file}")

    # Compute baselines for text_cpca (nc=10, seed=0)
    print("\n" + "="*60)
    print("Computing baselines for text_cpca (nc=10)")
    print("="*60)

    text_cpca_baselines = compute_baselines_for_probe(
        probe_pattern='probe_layer{layer}_nc10_seed0.pkl',
        n_components=10
    )

    output_file = OUTPUT_DIR / 'text_cpca_baselines.json'
    with open(output_file, 'w') as f:
        json.dump({
            'probe_type': 'text_cpca',
            'n_components': 10,
            'seed': 0,
            'emotions': EMOTIONS,
            'baselines': {str(k): v for k, v in text_cpca_baselines.items()}
        }, f, indent=2)
    print(f"\nSaved to {output_file}")

    print("\n" + "="*60)
    print("DONE! Baselines saved to:")
    print(f"  - {OUTPUT_DIR / 'text_raw_baselines.json'}")
    print(f"  - {OUTPUT_DIR / 'text_cpca_baselines.json'}")
    print("="*60)


if __name__ == '__main__':
    main()
