#!/bin/bash
#SBATCH --job-name=autointerp_sv_L50_L60
#SBATCH --output=gemmascope/slurm_jobs/autointerp_steering_vectors_L50_L60_%j.out
#SBATCH --error=gemmascope/slurm_jobs/autointerp_steering_vectors_L50_L60_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Auto-interpret top/bottom 20 features from steering vector analysis for layers 50 and 60

set -e

cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/.secrets/load_secrets.sh
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

export HF_HOME="/workspace-vast/annas/.cache/huggingface"
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
echo "ANTHROPIC_API_KEY set: $([ -n "$ANTHROPIC_API_KEY" ] && echo 'yes' || echo 'no')"

python << 'PYTHON_SCRIPT'
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import anthropic

# Add gemmascope to path
sys.path.insert(0, 'gemmascope')
from autointerp_features import (
    load_sae_example_data,
    get_feature_examples,
    interpret_feature,
)

# Create Anthropic client
client = anthropic.Anthropic()

def process_layer(layer: int, n_features: int = 20):
    """Process steering vector results for a layer."""

    summary_path = f"gemmascope/outputs/steering_vector_analysis/textmeandiff_layer{layer}_262k.summary.json"
    output_path = f"gemmascope/outputs/steering_vector_analysis/autointerp_layer{layer}_262k.json"

    print(f"\n{'='*60}")
    print(f" Processing Layer {layer}")
    print(f"{'='*60}")

    # Load steering vector analysis results
    print(f"Loading results from {summary_path}")
    with open(summary_path) as f:
        sv_data = json.load(f)

    # Load SAE example data
    print(f"Loading SAE example data for layer {layer}...")
    example_data = load_sae_example_data(
        model_size="27b",
        layer=layer,
        width="262k",
        l0="small",
        use_all_layers=True,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    # Collect all unique features to interpret
    features_to_interp = {}  # feature_idx -> {emotion, direction, cosine_sim}

    for emotion, results in sv_data['results'].items():
        # Top N (max cosine - aligned with emotion)
        for feat in results['max_cosine'][:n_features]:
            idx = feat['feature_idx']
            if idx not in features_to_interp:
                features_to_interp[idx] = {
                    'emotions': [],
                    'cosine_sim': feat['cosine_sim'],
                }
            features_to_interp[idx]['emotions'].append(f"{emotion}_max")

        # Bottom N (min cosine - opposite to emotion)
        for feat in results['min_cosine'][:n_features]:
            idx = feat['feature_idx']
            if idx not in features_to_interp:
                features_to_interp[idx] = {
                    'emotions': [],
                    'cosine_sim': feat['cosine_sim'],
                }
            features_to_interp[idx]['emotions'].append(f"{emotion}_min")

    print(f"Will interpret {len(features_to_interp)} unique features")

    # Gather examples for all features
    print("Gathering examples...")
    feature_examples = {}
    for idx in features_to_interp:
        examples = get_feature_examples(
            example_data, idx, tokenizer,
            n_top=10, n_bottom=10, n_middle=10, context_size=15
        )
        feature_examples[idx] = examples

    # Interpret features concurrently
    print("Interpreting features with 20 concurrent API calls...")
    all_results = []

    def interp_one(idx):
        examples = feature_examples[idx]
        interpretation = interpret_feature(
            client,
            examples,
            model="claude-sonnet-4-20250514",
        )
        return {
            'feature_idx': idx,
            'emotions': features_to_interp[idx]['emotions'],
            'cosine_sim': features_to_interp[idx]['cosine_sim'],
            'interpretation': interpretation,
            'top_tokens': [t for _, t, _ in examples.get('top', [])[:5]],
            'bottom_tokens': [t for _, t, _ in examples.get('bottom', [])[:5]],
        }

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(interp_one, idx): idx for idx in features_to_interp}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                print(f"Feature {idx}: {result['interpretation'][:60]}...")
            except Exception as e:
                print(f"Error interpreting feature {idx}: {e}")

    # Save results
    with open(output_path, 'w') as f:
        json.dump({
            'layer': layer,
            'n_features_per_emotion': n_features,
            'total_features': len(all_results),
            'results': all_results,
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    return all_results


# Process layers 50 and 60
for layer in [50, 60]:
    try:
        process_layer(layer, n_features=20)
    except Exception as e:
        print(f"Error processing layer {layer}: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")
PYTHON_SCRIPT

echo "All layers processed!"
