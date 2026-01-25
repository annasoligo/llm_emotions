#!/usr/bin/env python3
"""
Auto-interpret SAE features using Claude Sonnet.

Takes features aligned with a LoRA and generates interpretations
using top, bottom, and middle activating examples.
"""

import torch
import json
import argparse
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import anthropic
from dataclasses import dataclass
from typing import Optional
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor


SYSTEM_EXPLAINER = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. You are analysing a neuron in a language model. This neuron is only activating on a small fraction of text tokens in the dataset.
Guidelines:

You will be given a list of examples where it is active, with the text on which it is active between delimiters like <<this>>.
- Try to produce a concise final description of when the neuron is active. Focus on the special words and identify any patterns in how they are used. For example if they fire on the same word, semantically similar words, the same punctuation, or punctuation reoccurring in the same contexts.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not include the delimiters (<< >>) in your explanation.
- Do not make lists of possible explanations or activations. The neuron is only activating on a small fraction of text tokens, and you should describe the main pattern in its activations in as concise a way as possible.
- Make your explanation less than 20 words. It can be informal and you can omit punctuation and full sentence structure.
- The last line of your response must be the formatted explanation, using EXPLANATION:

For example:
e.g.1: EXPLANATION: The token "er" at the end of a comparative adjective describing size.
e.g.2: EXPLANATION: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
e.g.3: EXPLANATION: Common idioms in text conveying positive sentiment.
"""


def load_tokenizer(model_name: str = "google/gemma-3-27b-it"):
    """Load tokenizer for decoding token IDs."""
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {model_name}...")
    return AutoTokenizer.from_pretrained(model_name)


def load_sae_example_data(
    model_size: str = "27b",
    category: str = "resid_post",
    layer: int = 20,
    width: str = "16k",
    l0: str = "small",
    instruction_tuned: bool = True,
    use_all_layers: bool = False,
) -> dict:
    """Load example activation data for feature interpretation."""
    variant = "it" if instruction_tuned else "pt"
    repo_id = f"google/gemma-scope-2-{model_size}-{variant}"

    actual_category = f"{category}_all" if use_all_layers else category
    filename = f"{actual_category}/layer_{layer}_width_{width}_l0_{l0}/examples.safetensors"

    print(f"Loading example data from {repo_id}/{filename}")
    path_to_data = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_file(path_to_data)


def get_feature_examples(
    example_data: dict,
    feature_idx: int,
    tokenizer,
    n_top: int = 10,
    n_bottom: int = 10,
    n_middle: int = 10,
    context_size: int = 15,
) -> dict:
    """
    Get top, bottom, and middle activating examples for a feature with context.

    Returns dict with:
    - top_examples: list of (context_str, token_str, activation) for highest activations
    - bottom_examples: list of (context_str, token_str, activation) for lowest activations
    - middle_examples: list of (context_str, token_str, activation) for median activations
    """
    result = {"feature_idx": feature_idx}

    # Helper to get context around a position
    def get_context_str(seq_idx, pos, token_seqs, ctx_size=15):
        """Get context string around a position in a sequence."""
        if seq_idx >= len(token_seqs):
            return None
        seq = token_seqs[seq_idx]
        start = max(0, pos - ctx_size)
        end = min(len(seq), pos + ctx_size + 1)

        # Decode tokens
        context_ids = seq[start:end].tolist()
        tokens = tokenizer.convert_ids_to_tokens(context_ids)

        # Mark the target token
        target_idx = pos - start
        if 0 <= target_idx < len(tokens):
            tokens[target_idx] = f"<<{tokens[target_idx]}>>"

        # Join tokens (handle sentencepiece style)
        text = "".join(t.replace("▁", " ") for t in tokens).strip()
        return text

    # Get token sequences for context lookup
    token_seqs = example_data.get("tokens")
    seq_ids = example_data.get("seq_ids")
    positions = example_data.get("positions")
    activations = example_data.get("activations")

    has_context = token_seqs is not None and seq_ids is not None and positions is not None

    # Top activating examples
    if "top_tokens" in example_data:
        top_toks = example_data["top_tokens"]
        if feature_idx < len(top_toks):
            token_ids = top_toks[feature_idx][:n_top].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            examples = []
            for i, (tok_id, tok_str) in enumerate(zip(token_ids, tokens)):
                act = None
                context = None

                # Get activation
                if activations is not None and feature_idx < len(activations):
                    if i < activations.shape[1]:
                        act = float(activations[feature_idx, i])

                # Get context
                if has_context and feature_idx < len(seq_ids):
                    if i < seq_ids.shape[1]:
                        sid = int(seq_ids[feature_idx, i])
                        pos = int(positions[feature_idx, i])
                        context = get_context_str(sid, pos, token_seqs, context_size)

                examples.append((context, tok_str, act))

            result["top"] = examples
            result["top_tokens"] = tokens  # Keep for backward compat

    # Bottom activating tokens
    if "bottom_tokens" in example_data:
        bottom_toks = example_data["bottom_tokens"]
        if feature_idx < len(bottom_toks):
            token_ids = bottom_toks[feature_idx][:n_bottom].tolist()
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Bottom tokens don't have context in standard format
            result["bottom"] = [(None, t, None) for t in tokens]
            result["bottom_tokens"] = tokens

    # Middle activating examples (50th percentile)
    # Get from middle of the activations array (which is sorted by activation)
    if activations is not None and feature_idx < len(activations):
        n_acts = activations.shape[1]
        mid_start = n_acts // 2 - n_middle // 2
        mid_end = mid_start + n_middle

        if has_context and feature_idx < len(seq_ids):
            middle_examples = []
            for i in range(mid_start, min(mid_end, n_acts)):
                sid = int(seq_ids[feature_idx, i])
                pos = int(positions[feature_idx, i])
                act = float(activations[feature_idx, i])
                context = get_context_str(sid, pos, token_seqs, context_size)

                # Get the token at this position
                if sid < len(token_seqs) and pos < len(token_seqs[sid]):
                    tok_id = int(token_seqs[sid][pos])
                    tok_str = tokenizer.convert_ids_to_tokens([tok_id])[0]
                else:
                    tok_str = "?"

                middle_examples.append((context, tok_str, act))

            result["middle"] = middle_examples

    # Get frequency if available
    if "feature_frequencies" in example_data:
        freq = example_data["feature_frequencies"]
        if feature_idx < len(freq):
            result["frequency"] = float(freq[feature_idx])

    return result


def format_examples_for_prompt(examples: dict) -> str:
    """Format feature examples into a prompt for the explainer."""
    lines = []

    lines.append(f"Feature {examples['feature_idx']}")
    if "frequency" in examples:
        lines.append(f"Activation frequency: {examples['frequency']:.2e}")

    lines.append("\n=== TOP ACTIVATING EXAMPLES (highest activation) ===")
    if "top" in examples:
        for i, item in enumerate(examples["top"], 1):
            if len(item) == 3:
                context, token, act = item
            else:
                context, token, act = None, item[0], item[1] if len(item) > 1 else None

            act_str = f" [act={act:.3f}]" if act is not None else ""

            if context:
                # Context already has <<token>> markers
                lines.append(f"{i}.{act_str} {context}")
            else:
                lines.append(f"{i}.{act_str} <<{token}>>")

    if "bottom" in examples:
        lines.append("\n=== BOTTOM ACTIVATING EXAMPLES (lowest/negative activation) ===")
        for i, item in enumerate(examples["bottom"], 1):
            if len(item) == 3:
                context, token, act = item
            else:
                context, token, act = None, item[0], item[1] if len(item) > 1 else None

            act_str = f" [act={act:.3f}]" if act is not None else ""

            if context:
                lines.append(f"{i}.{act_str} {context}")
            else:
                lines.append(f"{i}.{act_str} <<{token}>>")

    if "middle" in examples:
        lines.append("\n=== MIDDLE ACTIVATING EXAMPLES (50th percentile) ===")
        for i, item in enumerate(examples["middle"], 1):
            if len(item) == 3:
                context, token, act = item
            else:
                context, token, act = None, item[0], item[1] if len(item) > 1 else None

            act_str = f" [act={act:.3f}]" if act is not None else ""

            if context:
                lines.append(f"{i}.{act_str} {context}")
            else:
                lines.append(f"{i}.{act_str} <<{token}>>")

    return "\n".join(lines)


def interpret_feature(
    client: anthropic.Anthropic,
    examples: dict,
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """Use Claude to interpret a feature based on its examples."""

    prompt = format_examples_for_prompt(examples)

    response = client.messages.create(
        model=model,
        max_tokens=200,
        system=SYSTEM_EXPLAINER,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.content[0].text

    # Extract the EXPLANATION line
    for line in text.split("\n"):
        if line.startswith("EXPLANATION:"):
            return line.replace("EXPLANATION:", "").strip()

    # If no EXPLANATION line found, return the full response
    return text.strip()


def interpret_features_concurrent(
    client: anthropic.Anthropic,
    features_with_examples: list,
    feature_groups: dict,
    feature_cosines: dict = None,
    model: str = "claude-sonnet-4-20250514",
    max_concurrent: int = 20,
) -> list:
    """
    Interpret multiple features concurrently.

    Args:
        features_with_examples: list of (feat_idx, examples) tuples
        feature_groups: dict mapping group names to feature lists
        feature_cosines: dict mapping feature IDs to cosine similarities
        max_concurrent: max concurrent API calls
    """
    if feature_cosines is None:
        feature_cosines = {}

    results = [None] * len(features_with_examples)

    def process_one(idx_and_data):
        i, (feat_idx, examples) = idx_and_data
        try:
            interpretation = interpret_feature(client, examples, model)
            groups = [g for g, feats in feature_groups.items() if feat_idx in feats]
            cosine_sim = feature_cosines.get(f"B_{feat_idx}")

            # Extract tokens and contexts
            top_tokens = examples.get("top_tokens", [])
            bottom_tokens = examples.get("bottom_tokens", [])
            top_contexts = []
            if "top" in examples:
                for item in examples["top"]:
                    if len(item) == 3 and item[0]:
                        top_contexts.append(item[0])

            return i, {
                "feature_idx": feat_idx,
                "groups": groups,
                "cosine_sim": cosine_sim,
                "top_tokens": top_tokens,
                "bottom_tokens": bottom_tokens,
                "top_contexts": top_contexts[:5] if top_contexts else None,
                "frequency": examples.get("frequency"),
                "interpretation": interpretation,
            }
        except Exception as e:
            print(f"  Error interpreting feature {feat_idx}: {e}")
            return i, {
                "feature_idx": feat_idx,
                "groups": [],
                "interpretation": f"ERROR: {e}",
            }

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = list(executor.map(process_one, enumerate(features_with_examples)))

    for i, result in futures:
        results[i] = result
        print(f"[{i+1}/{len(features_with_examples)}] Feature {result['feature_idx']}: {result.get('interpretation', 'N/A')[:60]}...")

    return results


def load_lora_analysis_results(summary_path: str) -> dict:
    """Load the LoRA analysis summary to get top features and cosine sims."""
    with open(summary_path) as f:
        data = json.load(f)

    # Also try to load the full cosine similarities if available
    # Only for .summary.json files (B matrix), not A_matrix.json files
    if summary_path.endswith(".summary.json"):
        sims_path = summary_path.replace(".summary.json", ".all_sims.npy")
        if Path(sims_path).exists():
            data["all_sims"] = np.load(sims_path, allow_pickle=True)

    return data


def load_features_from_files(b_path: str, a_path: str = None, n_features: int = 50) -> dict:
    """
    Load features from B and optionally A analysis files.

    Returns dict with:
    - b_max: top features by max cosine with B
    - b_min: top features by min cosine with B
    - a_max: top features by max cosine with W_down @ A^T (if a_path provided)
    - a_min: top features by min cosine with W_down @ A^T (if a_path provided)
    """
    result = {}

    # Load B matrix results
    with open(b_path) as f:
        b_data = json.load(f)
    result["b_max"] = b_data.get("top_promoted", [])[:n_features]
    result["b_min"] = b_data.get("top_suppressed", [])[:n_features]

    # Load A matrix results if provided
    if a_path:
        with open(a_path) as f:
            a_data = json.load(f)
        result["a_max"] = a_data.get("top_max_cos", [])[:n_features]
        result["a_min"] = a_data.get("top_min_cos", [])[:n_features]

    return result


def main():
    parser = argparse.ArgumentParser(description="Auto-interpret SAE features")
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Path to LoRA analysis summary JSON (B matrix)",
    )
    parser.add_argument(
        "--a-matrix-path",
        type=str,
        default=None,
        help="Path to A matrix analysis JSON",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--sae-width",
        type=str,
        default="16k",
    )
    parser.add_argument(
        "--sae-l0",
        type=str,
        default="small",
    )
    parser.add_argument(
        "--use-all-layers",
        action="store_true",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="27b",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of top features to interpret",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path",
    )
    parser.add_argument(
        "--interp-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for interpretation",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Number of concurrent API calls (default 1, max recommended 20)",
    )

    args = parser.parse_args()

    if not args.summary_path and not args.a_matrix_path:
        raise ValueError("Must provide at least one of --summary-path or --a-matrix-path")

    # Load features from both files
    feature_groups = {}

    # Store cosine similarities for each feature
    feature_cosines = {}

    if args.summary_path:
        print(f"Loading B matrix analysis from {args.summary_path}")
        b_data = load_lora_analysis_results(args.summary_path)
        feature_groups["B_max_cos"] = b_data.get("top_promoted", [])[:args.n_features]
        feature_groups["B_min_cos"] = b_data.get("top_suppressed", [])[:args.n_features]

        # Load cosine sims if available
        if "all_sims" in b_data:
            b_sims = b_data["all_sims"]
            for feat in feature_groups["B_max_cos"] + feature_groups["B_min_cos"]:
                if feat < len(b_sims):
                    feature_cosines[f"B_{feat}"] = float(b_sims[feat])

    if args.a_matrix_path:
        print(f"Loading A matrix analysis from {args.a_matrix_path}")
        a_data = load_lora_analysis_results(args.a_matrix_path)
        feature_groups["A_max_cos"] = a_data.get("top_max_cos", [])[:args.n_features]
        feature_groups["A_min_cos"] = a_data.get("top_min_cos", [])[:args.n_features]

        # Note: A matrix sims would need separate loading

    # Collect all unique features
    all_features = []
    for group_name, features in feature_groups.items():
        all_features.extend(features)

    # Remove duplicates while preserving order
    seen = set()
    features_to_interpret = [f for f in all_features if not (f in seen or seen.add(f))]

    print(f"Will interpret {len(features_to_interpret)} unique features across {len(feature_groups)} groups")

    # Load example data
    example_data = load_sae_example_data(
        model_size=args.model_size,
        layer=args.sae_layer,
        width=args.sae_width,
        l0=args.sae_l0,
        use_all_layers=args.use_all_layers,
    )

    # Print available keys in example data
    print(f"Example data keys: {list(example_data.keys())}")
    for k, v in example_data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Load tokenizer
    tokenizer = load_tokenizer(f"google/gemma-3-{args.model_size}-it")

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Get examples for all features first
    print(f"\nGathering examples for {len(features_to_interpret)} features...")
    features_with_examples = []
    for feat_idx in features_to_interpret:
        examples = get_feature_examples(example_data, feat_idx, tokenizer)
        features_with_examples.append((feat_idx, examples))

    # Interpret features (concurrent or sequential)
    if args.concurrent > 1:
        print(f"\nInterpreting features with {args.concurrent} concurrent API calls...")
        results = interpret_features_concurrent(
            client, features_with_examples, feature_groups,
            feature_cosines=feature_cosines,
            model=args.interp_model, max_concurrent=args.concurrent
        )
    else:
        print(f"\nInterpreting features sequentially...")
        results = []
        for i, (feat_idx, examples) in enumerate(features_with_examples):
            print(f"\n[{i+1}/{len(features_to_interpret)}] Interpreting feature {feat_idx}...")
            print(f"  Top tokens: {[t for _, t, _ in examples.get('top', [])]}")
            print(f"  Bottom tokens: {[t for _, t, _ in examples.get('bottom', [])]}")

            interpretation = interpret_feature(client, examples, model=args.interp_model)
            print(f"  Interpretation: {interpretation}")

            groups = [g for g, feats in feature_groups.items() if feat_idx in feats]

            # Get cosine similarity if available
            cosine_sim = feature_cosines.get(f"B_{feat_idx}")

            # Extract tokens from examples (handle both old and new format)
            top_tokens = examples.get("top_tokens", [])
            bottom_tokens = examples.get("bottom_tokens", [])

            # Extract contexts if available
            top_contexts = []
            if "top" in examples:
                for item in examples["top"]:
                    if len(item) == 3 and item[0]:  # (context, token, act)
                        top_contexts.append(item[0])

            results.append({
                "feature_idx": feat_idx,
                "groups": groups,
                "cosine_sim": cosine_sim,
                "top_tokens": top_tokens,
                "bottom_tokens": bottom_tokens,
                "top_contexts": top_contexts[:5] if top_contexts else None,
                "frequency": examples.get("frequency"),
                "interpretation": interpretation,
            })

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {output_path}")

    # Print summary by group
    print("\n" + "=" * 80)
    print("FEATURE INTERPRETATIONS BY GROUP")
    print("=" * 80)

    for group_name in feature_groups.keys():
        print(f"\n{'='*40}")
        print(f"GROUP: {group_name}")
        print(f"{'='*40}")
        group_results = [r for r in results if group_name in r.get("groups", [])]
        for r in group_results[:25]:  # Show top 25 per group
            cos_str = f" (cos={r['cosine_sim']:.4f})" if r.get('cosine_sim') else ""
            print(f"\nFeature {r['feature_idx']}{cos_str}:")
            print(f"  Top tokens: {r['top_tokens'][:5]}")
            print(f"  Bottom tokens: {r['bottom_tokens'][:5]}")
            print(f"  Interpretation: {r['interpretation']}")

    return results


if __name__ == "__main__":
    main()
