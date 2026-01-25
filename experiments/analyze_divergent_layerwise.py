#!/usr/bin/env python3
"""
Re-analyze divergent probe/logit cases with per-layer breakdown.

Takes the divergent cases and computes scores at individual layers
to see if divergence is consistent across layers or layer-specific.

Usage:
    python experiments/analyze_divergent_layerwise.py \
        --divergent-csv experiments/distress_analysis/divergent_scores.csv \
        --wildchat-csv experiments/distress_analysis/wildchat_distress.csv \
        --output experiments/distress_analysis/divergent_layerwise.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from probes.scripts.probe_pipeline import ProbeActivationExtractor, ProbeInference
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from emotion_logit_lens.core import project_to_logits, compute_emotion_scores_from_logits
from emotion_logit_lens.emotion_tokens import EmotionTokenManager
from emotion_logit_lens.baseline_loader import LogitBaselineLoader

# Constants
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
DISTRESS_EMOTIONS = ['anger', 'fear', 'sadness']
# Gemma-3-27B has 62 layers (0-61) - we have probes for all!
ALL_LAYERS = list(range(0, 62))  # All layers for both probe and logit lens
LAYER_RANGES = {
    'very_early': list(range(0, 12)),    # 0-11
    'early': list(range(12, 24)),        # 12-23
    'mid': list(range(24, 40)),          # 24-39
    'late': list(range(40, 52)),         # 40-51
    'very_late': list(range(52, 62)),    # 52-61
}


def extract_response_activations(
    model,
    tokenizer,
    user_message: str,
    assistant_response: str,
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Extract mean activations for the assistant response tokens."""
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Find assistant response token range
    pre_response_messages = [{"role": "user", "content": user_message}, {"role": "assistant", "content": ""}]
    pre_response_prompt = tokenizer.apply_chat_template(
        pre_response_messages, tokenize=False, add_generation_prompt=True
    )
    pre_response_ids = tokenizer(pre_response_prompt, return_tensors="pt")["input_ids"]
    response_start_idx = pre_response_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    layer_activations = {}
    for layer in layers:
        hidden_states = outputs.hidden_states[layer + 1]
        response_hidden = hidden_states[0, response_start_idx:, :]
        if response_hidden.shape[0] == 0:
            response_hidden = hidden_states[0, -1:, :]
        mean_activation = response_hidden.mean(dim=0).float().cpu().numpy()
        layer_activations[layer] = mean_activation

    return layer_activations


def compute_probe_scores_per_layer(
    activations: Dict[int, np.ndarray],
    probe_inference: ProbeInference,
    baseline_stats_per_layer: Dict[int, Dict[str, np.ndarray]],
    layers: List[int]
) -> Dict[int, Dict[str, float]]:
    """Compute probe scores at each layer individually with per-layer normalization."""
    layer_scores = {}

    for layer in layers:
        activation = activations[layer]
        scores = probe_inference.predict(
            activations=activation.reshape(1, -1),
            layer=layer,
            n_components=0,
            seed=0
        )[0]

        # Z-score normalize using THIS layer's baseline
        layer_baseline = baseline_stats_per_layer[layer]
        normalized = (scores - layer_baseline['mean']) / (layer_baseline['std'] + 1e-8)
        layer_scores[layer] = {
            emotion: float(normalized[i]) for i, emotion in enumerate(EMOTIONS)
        }
        layer_scores[layer]['distress'] = sum(
            layer_scores[layer][e] for e in DISTRESS_EMOTIONS
        )

    return layer_scores


def compute_logit_scores_per_layer(
    model,
    activations: Dict[int, np.ndarray],
    emotion_token_manager: EmotionTokenManager,
    logit_baseline_loader: LogitBaselineLoader,
    layers: List[int]
) -> Dict[int, Dict[str, float]]:
    """Compute logit lens scores at each layer individually."""
    emotion_token_ids = emotion_token_manager.load_emotion_token_ids()
    layer_scores = {}

    for layer in layers:
        activation = activations[layer]
        logits = project_to_logits(model, activation)

        try:
            layer_stats = logit_baseline_loader.load_layer_stats(layer)
            baseline_stats = {
                'layers_data': {str(layer): {'statistics': layer_stats}}
            }
        except FileNotFoundError:
            baseline_stats = None

        scores = compute_emotion_scores_from_logits(
            logits=logits,
            emotion_token_ids=emotion_token_ids,
            baseline_stats=baseline_stats,
            layer=layer,
            aggregation='mean'
        )

        layer_scores[layer] = {emotion: scores.get(emotion, 0.0) for emotion in EMOTIONS}
        layer_scores[layer]['distress'] = sum(
            layer_scores[layer][e] for e in DISTRESS_EMOTIONS
        )

    return layer_scores


def format_conversation(row: pd.Series, full_wildchat_df: pd.DataFrame = None) -> str:
    """Format the conversation for display."""
    # Try to get full text from wildchat df if available
    user_msg = row['user_message']
    assistant_resp = row['assistant_response']

    output = []
    output.append("=" * 80)
    output.append(f"Conversation ID: {row['conversation_id']}, Turn: {row['turn_number']}")
    output.append(f"Divergence Type: {row['divergence_type']}")
    output.append("=" * 80)
    output.append("")
    output.append("USER PROMPT:")
    output.append("-" * 40)
    output.append(user_msg)
    output.append("")
    output.append("ASSISTANT RESPONSE:")
    output.append("-" * 40)
    output.append(assistant_resp)
    output.append("")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(description="Re-analyze divergent cases with per-layer breakdown")
    parser.add_argument(
        "--divergent-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments/distress_analysis/divergent_scores.csv",
        help="CSV with divergent cases"
    )
    parser.add_argument(
        "--wildchat-csv",
        type=Path,
        default=PROJECT_ROOT / "experiments/distress_analysis/wildchat_distress.csv",
        help="Full WildChat results CSV (for full prompts)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments/distress_analysis/divergent_layerwise.csv",
        help="Output CSV with per-layer scores"
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=PROJECT_ROOT / "experiments/distress_analysis/divergent_analysis.txt",
        help="Output text file with formatted analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-27b-it",
        help="Model to load"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("LAYERWISE ANALYSIS OF DIVERGENT CASES")
    print("=" * 80)

    # Load divergent cases
    df_div = pd.read_csv(args.divergent_csv)
    print(f"Loaded {len(df_div)} divergent cases")

    # Load full WildChat data for complete prompts
    df_full = pd.read_csv(args.wildchat_csv)

    # Re-fetch full prompts from wildchat csv (they may be truncated in divergent)
    for i, row in df_div.iterrows():
        conv_id = row['conversation_id']
        turn = row['turn_number']
        match = df_full[(df_full['conversation_id'] == conv_id) & (df_full['turn_number'] == turn)]
        if len(match) > 0:
            df_div.at[i, 'user_message'] = match.iloc[0]['user_message']
            df_div.at[i, 'assistant_response'] = match.iloc[0]['assistant_response']

    # Load model
    print("\nLoading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        print("  Using flash_attention_2")
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"
        )
        print("  Using sdpa")
    model.eval()

    # Initialize probes
    print("\nInitializing probes...")
    probe_dir = PROJECT_ROOT / "outputs/probes/emotion_probes/text_based/multiseed"
    probe_inference = ProbeInference(
        probe_dir=probe_dir,
        cpca_path=None,
        device='cuda',
        probe_pattern='probe_layer{layer}_nc0_seed0.pkl'
    )

    baseline_dir = PROJECT_ROOT / "data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"
    baseline_loader = WildChatBaselineLoader(
        aggregation_type="all_tokens",
        baseline_dir=baseline_dir
    )

    # Pre-compute PER-LAYER baseline statistics for probe scores
    print("  Computing per-layer probe baselines...")
    import h5py
    probe_baseline_per_layer = {}
    for layer in tqdm(ALL_LAYERS, desc="  Loading baselines"):
        h5_file = baseline_dir / f"layer{layer}_activations.h5"
        with h5py.File(h5_file, 'r') as f:
            baseline_acts = f["all_tokens"][:]  # [n_samples, hidden_dim]

        # Apply probe to all baseline samples and compute mean/std of scores
        scores_list = []
        for i in range(baseline_acts.shape[0]):
            scores = probe_inference.predict(
                activations=baseline_acts[i:i+1],
                layer=layer,
                n_components=0,
                seed=0
            )[0]
            scores_list.append(scores)

        scores_array = np.stack(scores_list, axis=0)  # [n_samples, n_emotions]
        probe_baseline_per_layer[layer] = {
            'mean': np.mean(scores_array, axis=0),
            'std': np.std(scores_array, axis=0)
        }
    print(f"  Computed baselines for {len(probe_baseline_per_layer)} layers")

    # Initialize logit lens
    print("\nInitializing logit lens...")
    emotion_token_manager = EmotionTokenManager(model_name="google_gemma_3_27b_it")
    logit_baseline_dir = PROJECT_ROOT / "data/baselines/logit_emotion_alpaca"
    logit_baseline_loader = LogitBaselineLoader(
        baseline_dir=logit_baseline_dir,
        model_name="google_gemma_3_27b_it"
    )

    # Process each divergent case
    print("\nProcessing divergent cases...")
    results = []
    txt_output = []

    for idx, row in tqdm(df_div.iterrows(), total=len(df_div)):
        user_msg = row['user_message']
        assistant_resp = row['assistant_response']

        try:
            # Extract activations for ALL layers (0-45)
            activations = extract_response_activations(
                model=model,
                tokenizer=tokenizer,
                user_message=user_msg,
                assistant_response=assistant_resp,
                layers=ALL_LAYERS
            )

            # Get per-layer scores (with per-layer normalization)
            probe_layer_scores = compute_probe_scores_per_layer(
                activations, probe_inference, probe_baseline_per_layer, ALL_LAYERS
            )
            # Logit lens for all layers
            logit_layer_scores = compute_logit_scores_per_layer(
                model, activations, emotion_token_manager, logit_baseline_loader, ALL_LAYERS
            )

            # Aggregate by layer range
            result = {
                'conversation_id': row['conversation_id'],
                'turn_number': row['turn_number'],
                'divergence_type': row['divergence_type'],
                'user_message': user_msg,
                'assistant_response': assistant_resp,
                'probe_distress_all': row['probe_distress'],
                'logit_distress_all': row['logit_distress'],
            }

            # Add per-range aggregates
            for range_name, layers in LAYER_RANGES.items():
                probe_distress_range = np.mean([probe_layer_scores[l]['distress'] for l in layers])
                logit_distress_range = np.mean([logit_layer_scores[l]['distress'] for l in layers])
                result[f'probe_distress_{range_name}'] = probe_distress_range
                result[f'logit_distress_{range_name}'] = logit_distress_range

            # Add individual layer scores for ALL layers
            for layer in ALL_LAYERS:
                result[f'probe_L{layer}'] = probe_layer_scores[layer]['distress']
                result[f'logit_L{layer}'] = logit_layer_scores[layer]['distress']

            results.append(result)

            # Format for text output
            txt_output.append(format_conversation(row))
            txt_output.append("\nSCORES BY LAYER RANGE:")
            txt_output.append("-" * 60)
            txt_output.append(f"{'Range':<15} {'Probe Distress':>15} {'Logit Distress':>15} {'Difference':>12}")
            txt_output.append("-" * 60)
            for range_name in ['very_early', 'early', 'mid', 'late', 'very_late']:
                p = result[f'probe_distress_{range_name}']
                l = result[f'logit_distress_{range_name}']
                txt_output.append(f"{range_name:<15} {p:>15.2f}σ {l:>15.2f}σ {l-p:>12.2f}σ")
            txt_output.append("")
            txt_output.append("SCORES BY INDIVIDUAL LAYER:")
            txt_output.append("-" * 50)
            txt_output.append(f"{'Layer':<8} {'Probe':>12} {'Logit':>12} {'Diff':>12}")
            for layer in ALL_LAYERS:
                p = result[f'probe_L{layer}']
                l = result[f'logit_L{layer}']
                txt_output.append(f"L{layer:<7} {p:>12.2f} {l:>12.2f} {l-p:>12.2f}")
            txt_output.append("\n")

        except Exception as e:
            print(f"\nError processing conv {row['conversation_id']}: {e}")
            continue

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\nSaved layerwise CSV to: {args.output}")

    with open(args.output_txt, 'w') as f:
        f.write("\n".join(txt_output))
    print(f"Saved formatted analysis to: {args.output_txt}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: LAYER RANGE DIFFERENCES")
    print("=" * 80)

    for div_type in df_results['divergence_type'].unique():
        subset = df_results[df_results['divergence_type'] == div_type]
        print(f"\n{div_type.upper()} (n={len(subset)}):")
        print(f"  {'Range':<15} {'Mean Probe':>12} {'Mean Logit':>12} {'Mean Diff':>12}")
        print("  " + "-" * 55)
        for range_name in ['very_early', 'early', 'mid', 'late', 'very_late']:
            mean_p = subset[f'probe_distress_{range_name}'].mean()
            mean_l = subset[f'logit_distress_{range_name}'].mean()
            print(f"  {range_name:<15} {mean_p:>12.2f}σ {mean_l:>12.2f}σ {mean_l-mean_p:>12.2f}σ")


if __name__ == "__main__":
    main()
