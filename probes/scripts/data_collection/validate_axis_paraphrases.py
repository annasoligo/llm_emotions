#!/usr/bin/env python3
"""Validate axis paraphrases using axis_lens.

Checks if generated paraphrases actually match the intended axis levels.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not/emotion_evals")

from emo_lens.logit_lens_emotion_direct import (
    load_axis_token_groups,
    compute_axis_scores_from_activations,
)
from emo_lens.token_trajectories import (
    extract_token_level_activations,
)
from emo_lens.model_utils import load_base_model


AXIS_NAMES = ["valence", "arousal", "dominance", "trust"]
LEVEL_TO_VALUE = {"low": -1, "neutral": 0, "high": 1}


def parse_combo_key(combo_key: str) -> list:
    """Parse combo key like '[high, low, high, high]' into list of values."""
    # Remove brackets and split
    combo_str = combo_key.strip("[]")
    levels = [level.strip() for level in combo_str.split(",")]
    return [LEVEL_TO_VALUE[level] for level in levels]


def validate_paraphrases(
    paraphrases_file: Path,
    model_name: str = "unsloth/gemma-3-27b-it",
    layers: list = None,
    device: str = "cuda",
):
    """Validate paraphrases using axis_lens."""

    if layers is None:
        layers = list(range(40, 51))  # Default: layers 40-50

    print("=" * 80)
    print("VALIDATING AXIS PARAPHRASES")
    print("=" * 80)
    print(f"Input file: {paraphrases_file}")
    print(f"Model: {model_name}")
    print(f"Layers: {layers[0]}-{layers[-1]}")
    print(f"Device: {device}")
    print()

    # Load data
    with open(paraphrases_file) as f:
        data = json.load(f)

    neutral_text = data["neutral_text"]
    paraphrases = data["paraphrases"]

    print(f"Neutral text: {neutral_text}")
    print(f"Number of paraphrases: {len(paraphrases)}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_base_model(
        model_name,
        device_map=device if device == "auto" else device,
        torch_dtype=torch.bfloat16,
    )
    print("✓ Model loaded")
    print()

    # Load axis token groups
    print("Loading axis token groups...")
    axis_token_groups = load_axis_token_groups(model_name)
    print("✓ Axis token groups loaded")
    print()

    # Load baseline stats
    baseline_path = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/instruct_baselines") / model_name.replace("/", "_") / "generated_tokens_avg" / "baseline_stats.json"

    if not baseline_path.exists():
        print(f"WARNING: Baseline stats not found at {baseline_path}")
        print("Will compute scores without normalization")
        baseline_stats = None
    else:
        with open(baseline_path) as f:
            baseline_stats = json.load(f)
        print(f"✓ Baseline stats loaded from {baseline_path}")
        print()

    # Validate each paraphrase
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()

    results = []

    for combo_key, paraphrase_text in paraphrases.items():
        # Parse intended levels
        intended_levels = parse_combo_key(combo_key)

        # Extract just the paraphrase text (remove Claude's analysis)
        if "Analysis" in paraphrase_text or "analysis" in paraphrase_text.lower():
            lines = paraphrase_text.split("\n")
            clean_lines = []
            for line in lines:
                if "analysis" in line.lower() or "**valence" in line.lower() or "**arousal" in line.lower():
                    break
                if line.strip() and not line.startswith("#"):
                    clean_lines.append(line)
            paraphrase_text = "\n".join(clean_lines).strip()

        # Remove any remaining prefixes like "Here's a rewrite..."
        if paraphrase_text.startswith("Here"):
            lines = paraphrase_text.split("\n")
            paraphrase_text = "\n".join(lines[1:]).strip()

        # Extract activations
        activations_by_token, token_ids = extract_token_level_activations(
            model=model,
            tokenizer=tokenizer,
            prompt=paraphrase_text,
            layers=layers,
        )

        # Average activations across tokens and layers
        all_activations = []
        for token_acts in activations_by_token.values():
            for layer in layers:
                if layer in token_acts:
                    act = token_acts[layer]
                    # Convert numpy to torch if needed
                    if isinstance(act, np.ndarray):
                        act = torch.from_numpy(act)
                    all_activations.append(act)

        avg_activation = torch.stack(all_activations).mean(dim=0)  # [hidden_dim]

        # Convert to numpy for compute_axis_scores_from_activations
        if isinstance(avg_activation, torch.Tensor):
            avg_activation = avg_activation.cpu().numpy()

        # Compute axis scores (function expects list of activations)
        # If we don't have baseline stats, we can't compute axis scores properly
        # For now, just project to vocab and compute raw scores
        if baseline_stats is None:
            print(f"  Warning: No baseline stats, computing raw axis scores without normalization")
            # Project to vocab
            with torch.no_grad():
                avg_activation_tensor = torch.from_numpy(avg_activation).to(model.device).unsqueeze(0)
                if hasattr(model.lm_head, 'weight'):
                    avg_activation_tensor = avg_activation_tensor.to(model.lm_head.weight.dtype)
                logits = model.project_on_vocab(avg_activation_tensor)[0].cpu().float().numpy()  # [vocab_size]

            # Compute raw axis scores (no normalization)
            axis_scores_dict = {}
            for axis_name, token_groups in axis_token_groups.items():
                high_tokens = token_groups.get('high', [])
                low_tokens = token_groups.get('low', [])

                high_logits = [logits[tid] for tid in high_tokens] if high_tokens else [0.0]
                low_logits = [logits[tid] for tid in low_tokens] if low_tokens else [0.0]

                high_score = np.mean(high_logits)
                low_score = np.mean(low_logits)

                axis_scores_dict[axis_name] = high_score - low_score
        else:
            axis_scores_list = compute_axis_scores_from_activations(
                model=model,
                activations=[avg_activation],  # Wrap in list
                axis_token_groups=axis_token_groups,
                layer=layers[0],  # Use first layer (doesn't matter since we pre-averaged)
                ref_stats=baseline_stats,
                subtract_mean=True,
                aggregation="mean",
            )

            # Extract the single result (it's a list of dicts)
            axis_scores_dict = axis_scores_list[0] if axis_scores_list else {}

        # Compare with intended levels
        result = {
            "combo_key": combo_key,
            "intended": {axis: level for axis, level in zip(AXIS_NAMES, intended_levels)},
            "measured": axis_scores_dict,  # Already a dict with axis names as keys
            "text_preview": paraphrase_text[:100] + "...",
        }

        # Compute agreement (sign match)
        agreements = []
        for axis, intended_val in zip(AXIS_NAMES, intended_levels):
            measured_val = result["measured"][axis]
            if intended_val == 0:
                # For neutral, check if measured is close to 0
                agreement = abs(measured_val) < 0.5
            else:
                # For high/low, check if sign matches
                agreement = np.sign(measured_val) == np.sign(intended_val)
            agreements.append(agreement)

        result["agreement"] = agreements
        result["agreement_rate"] = sum(agreements) / len(agreements)

        results.append(result)

        # Print result
        print(f"Combination: {combo_key}")
        print(f"Text: {result['text_preview']}")
        print(f"Intended: {result['intended']}")
        print(f"Measured: {result['measured']}")
        print(f"Agreement: {result['agreement']} ({result['agreement_rate']:.1%})")
        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    overall_agreement = np.mean([r["agreement_rate"] for r in results])
    print(f"Overall agreement rate: {overall_agreement:.1%}")
    print()

    # Per-axis agreement
    print("Per-axis agreement:")
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        axis_agreements = [r["agreement"][axis_idx] for r in results]
        axis_agreement_rate = sum(axis_agreements) / len(axis_agreements)
        print(f"  {axis_name:20s}: {axis_agreement_rate:.1%}")

    print()

    # Save results
    output_file = paraphrases_file.parent / f"{paraphrases_file.stem}_validation.json"
    with open(output_file, "w") as f:
        json.dump({
            "validation_results": results,
            "summary": {
                "overall_agreement": float(overall_agreement),
                "per_axis_agreement": {
                    axis: float(sum([r["agreement"][i] for r in results]) / len(results))
                    for i, axis in enumerate(AXIS_NAMES)
                },
            },
        }, f, indent=2)

    print(f"Validation results saved to: {output_file}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate axis paraphrases with axis_lens")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/workspace-vast/annas/git/research-tools/probes/data/axis_paraphrases/axis_paraphrases_test.jsonl"),
        help="Input paraphrases file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-27b-it",
        help="Model name",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="40-50",
        help="Layer range (e.g., '40-50')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    # Parse layer range
    if "-" in args.layers:
        start, end = map(int, args.layers.split("-"))
        layers = list(range(start, end + 1))
    else:
        layers = [int(args.layers)]

    validate_paraphrases(
        paraphrases_file=args.input,
        model_name=args.model,
        layers=layers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
