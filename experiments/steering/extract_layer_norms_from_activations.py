"""
Extract layer norms from existing activation files.

This avoids recomputing activations - we use the neutral text activations
that were already collected for probe training.

Saves results to experiments/steering/layer_norms.json
"""

import json
import h5py
import numpy as np
from pathlib import Path


def extract_gemma_norms():
    """Extract layer norms from Gemma activation file."""
    filepath = Path("probes/ua_emotion_disentangle/data/gemma3_27b_ua_emotions_v4_alllayers.h5")

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    layer_norms = {}
    with h5py.File(filepath, 'r') as f:
        layers = f.attrs['layers']

        for layer_idx in layers:
            layer_key = f'layer_{layer_idx}'
            if layer_key in f:
                layer_data = f[layer_key]
                if 'final_token_activations' in layer_data:
                    acts = layer_data['final_token_activations'][:]
                    norms = np.linalg.norm(acts, axis=-1)
                    layer_norms[int(layer_idx)] = {
                        "mean": float(np.mean(norms)),
                        "std": float(np.std(norms)),
                        "n": len(norms),
                    }

    return {
        "model_id": "google/gemma-3-27b-it",
        "n_layers": 62,
        "hidden_dim": 5376,
        "source": str(filepath),
        "layers": {str(k): v for k, v in layer_norms.items()},
    }


def extract_qwen32b_norms():
    """Extract layer norms from Qwen 32B activation file."""
    filepath = Path("probes/ua_emotion_disentangle/data/qwen3_32b_texts_combined.h5")

    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    layer_norms = {}
    with h5py.File(filepath, 'r') as f:
        layers = f.attrs['layers']
        acts = f['activations']
        all_keys = list(acts.keys())

        # Initialize collectors
        norm_collectors = {int(l): [] for l in layers}

        # Collect norms from neutral activations
        for key in all_keys:
            if 'neutral' in acts[key]:
                neutral = acts[key]['neutral'][:]  # (n_layers, hidden_dim)
                for i, layer in enumerate(layers):
                    norm = np.linalg.norm(neutral[i])
                    norm_collectors[int(layer)].append(norm)

        # Compute stats
        for layer, norms in norm_collectors.items():
            if norms:
                layer_norms[layer] = {
                    "mean": float(np.mean(norms)),
                    "std": float(np.std(norms)),
                    "n": len(norms),
                }

    return {
        "model_id": "Qwen/Qwen3-32B",
        "n_layers": 64,
        "hidden_dim": 5120,
        "source": str(filepath),
        "layers": {str(k): v for k, v in layer_norms.items()},
    }


def extract_qwen235b_norms():
    """Extract layer norms from Qwen 235B activation files."""
    files = [
        "probes/ua_emotion_disentangle/data/qwen3_235b_a22b_ua_layers_0_19_20260114_180547.h5",
        "probes/ua_emotion_disentangle/data/qwen3_235b_a22b_ua_layers_20_39_20260114_180547.h5",
        "probes/ua_emotion_disentangle/data/qwen3_235b_a22b_ua_layers_40_59_20260114_180547.h5",
        "probes/ua_emotion_disentangle/data/qwen3_235b_a22b_ua_layers_60_79_20260114_180547.h5",
        "probes/ua_emotion_disentangle/data/qwen3_235b_a22b_ua_layers_80_93_20260114_180547.h5",
    ]

    layer_norms = {}
    sources = []

    for fpath in files:
        filepath = Path(fpath)
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        sources.append(str(filepath))

        with h5py.File(filepath, 'r') as f:
            layers = f.attrs['layers']

            for layer_idx in layers:
                layer_key = f'layer_{layer_idx}'
                if layer_key in f:
                    layer_data = f[layer_key]
                    if 'final_token_activations' in layer_data:
                        acts = layer_data['final_token_activations'][:]
                        norms = np.linalg.norm(acts, axis=-1)
                        layer_norms[int(layer_idx)] = {
                            "mean": float(np.mean(norms)),
                            "std": float(np.std(norms)),
                            "n": len(norms),
                        }

    return {
        "model_id": "Qwen/Qwen3-235B-A22B",
        "n_layers": 94,
        "hidden_dim": 4096,
        "source": sources,
        "layers": {str(k): v for k, v in layer_norms.items()},
    }


def main():
    output_path = Path("experiments/steering/layer_norms.json")

    results = {}

    # Extract Gemma norms
    print("Extracting Gemma 27B layer norms...")
    gemma = extract_gemma_norms()
    if gemma:
        results["gemma"] = gemma
        print(f"  Extracted {len(gemma['layers'])} layers")
        print(f"  Layer 30 norm: {gemma['layers']['30']['mean']:.2f}")

    # Extract Qwen 32B norms
    print("\nExtracting Qwen 32B layer norms...")
    qwen32b = extract_qwen32b_norms()
    if qwen32b:
        results["qwen32b"] = qwen32b
        print(f"  Extracted {len(qwen32b['layers'])} layers")
        print(f"  Layer 30 norm: {qwen32b['layers']['30']['mean']:.2f}")

    # Extract Qwen 235B norms
    print("\nExtracting Qwen 235B layer norms...")
    qwen235b = extract_qwen235b_norms()
    if qwen235b:
        results["qwen235b"] = qwen235b
        print(f"  Extracted {len(qwen235b['layers'])} layers")
        print(f"  Layer 50 norm: {qwen235b['layers']['50']['mean']:.2f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Print comparison with hardcoded values
    print("\n" + "="*60)
    print("COMPARISON WITH HARDCODED VALUES")
    print("="*60)
    print(f"{'Model':<15} {'Layer':<8} {'Extracted':<12} {'Hardcoded':<12} {'Diff %':<10}")
    print("-"*60)

    comparisons = [
        ("gemma", 30, 42151.76),
        ("qwen32b", 30, 142.0),
        ("qwen235b", 50, 23.28),
    ]

    for model, layer, hardcoded in comparisons:
        if model in results and str(layer) in results[model]['layers']:
            extracted = results[model]['layers'][str(layer)]['mean']
            diff_pct = (extracted - hardcoded) / hardcoded * 100
            print(f"{model:<15} {layer:<8} {extracted:<12.2f} {hardcoded:<12.2f} {diff_pct:+.1f}%")


if __name__ == "__main__":
    main()
