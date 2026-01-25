#!/usr/bin/env python3
"""
Extract layer norms from text pair activations.

Computes the mean L2 norm of the residual stream at each layer,
using only the neutral text activations as the baseline.

Output format (JSON):
{
    "model": "gemma3_27b",
    "source": "steering_tests/activations/test_gemma3_27b_text_pairs",
    "hidden_dim": 5376,
    "n_samples": 1500,
    "layers": {
        "0": {"mean": 891.42, "std": 14.37, "n": 1500},
        "1": {"mean": 977.98, "std": 7.97, "n": 1500},
        ...
    }
}

Usage:
    python -m steering_tests.steering_utils.extract_layer_norms \
        --activations steering_tests/activations/test_gemma3_27b_text_pairs \
        --output steering_tests/steering_utils/layer_norms/gemma3_27b.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


def extract_neutral_norms(
    activations_dir: str,
    output_file: str,
    model_name: Optional[str] = None,
) -> Dict:
    """
    Extract layer norms from neutral activations only.

    Args:
        activations_dir: Directory containing layer_XX.pkl files
        output_file: Output JSON file path
        model_name: Optional model name for metadata

    Returns:
        Dict with layer norm statistics
    """
    act_path = Path(activations_dir)

    # Find all layer pkl files
    layer_files = sorted(act_path.glob("layer_*.pkl"))
    if not layer_files:
        raise ValueError(f"No layer_*.pkl files found in {activations_dir}")

    print(f"Found {len(layer_files)} layer files")

    # Infer model name from directory if not provided
    if model_name is None:
        dir_name = act_path.name
        # e.g., test_gemma3_27b_text_pairs -> gemma3_27b
        model_name = dir_name.replace("test_", "").replace("_text_pairs", "")

    results = {
        "model": model_name,
        "source": str(activations_dir),
        "hidden_dim": None,
        "n_samples": None,
        "layers": {}
    }

    for layer_file in tqdm(layer_files, desc="Processing layers"):
        # Extract layer number from filename (layer_00.pkl -> 0)
        layer_num = int(layer_file.stem.split("_")[1])

        with open(layer_file, 'rb') as f:
            data = pickle.load(f)

        # Collect norms from neutral activations only
        norms = []
        for key, activation in data.items():
            # Only use neutral activations
            if "_neutral" in key:
                # activation is shape (hidden_dim,)
                norm = np.linalg.norm(activation)
                norms.append(norm)

                # Set hidden_dim from first activation
                if results["hidden_dim"] is None:
                    results["hidden_dim"] = activation.shape[0]

        if norms:
            norms = np.array(norms)
            results["layers"][str(layer_num)] = {
                "mean": float(np.mean(norms)),
                "std": float(np.std(norms)),
                "n": len(norms)
            }

            # Set n_samples from first layer
            if results["n_samples"] is None:
                results["n_samples"] = len(norms)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nExtracted norms for {len(results['layers'])} layers")
    print(f"Hidden dim: {results['hidden_dim']}")
    print(f"Samples per layer: {results['n_samples']}")
    print(f"Saved to: {output_path}")

    # Print sample of layer norms
    print("\nSample layer norms:")
    for layer in sorted(results['layers'].keys(), key=int)[:5]:
        info = results['layers'][layer]
        print(f"  Layer {layer}: {info['mean']:.2f} ± {info['std']:.2f}")
    print("  ...")
    for layer in sorted(results['layers'].keys(), key=int)[-3:]:
        info = results['layers'][layer]
        print(f"  Layer {layer}: {info['mean']:.2f} ± {info['std']:.2f}")

    return results


def extract_all_models():
    """Extract layer norms for all available text pair activations."""
    base_dir = Path(__file__).parent.parent / "activations"
    output_dir = Path(__file__).parent / "layer_norms"

    # Find all text_pairs activation directories
    text_pair_dirs = list(base_dir.glob("*_text_pairs"))

    print(f"Found {len(text_pair_dirs)} text pair activation directories:")
    for d in text_pair_dirs:
        print(f"  - {d.name}")
    print()

    for act_dir in text_pair_dirs:
        # Infer model name
        model_name = act_dir.name.replace("test_", "").replace("_text_pairs", "")
        output_file = output_dir / f"{model_name}.json"

        print(f"\n{'='*60}")
        print(f"Processing: {act_dir.name}")
        print(f"{'='*60}")

        try:
            extract_neutral_norms(
                str(act_dir),
                str(output_file),
                model_name
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Extract layer norms from text pair activations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--activations", "-a",
        type=str,
        default=None,
        help="Directory containing layer_XX.pkl files (if not provided, processes all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--model-name", "-m",
        type=str,
        default=None,
        help="Model name for metadata (inferred from directory if not provided)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all text_pairs activation directories"
    )

    args = parser.parse_args()

    if args.all:
        extract_all_models()
    elif args.activations:
        if args.output is None:
            # Auto-generate output path
            act_path = Path(args.activations)
            model_name = act_path.name.replace("test_", "").replace("_text_pairs", "")
            args.output = f"steering_tests/steering_utils/layer_norms/{model_name}.json"

        extract_neutral_norms(args.activations, args.output, args.model_name)
    else:
        parser.print_help()
        print("\nExample:")
        print("  python -m steering_tests.steering_utils.extract_layer_norms --all")
        print("  python -m steering_tests.steering_utils.extract_layer_norms -a steering_tests/activations/test_gemma3_27b_text_pairs")


if __name__ == "__main__":
    main()
