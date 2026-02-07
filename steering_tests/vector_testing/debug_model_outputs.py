#!/usr/bin/env python3
"""
Unified debug script for analyzing model outputs and logprobs.

Consolidates functionality from debug_235b_outputs.py, debug_ethical.py,
debug_qwen3_detailed.py, debug_qwen3_output.py, and debug_tokenize.py.

Usage:
    # Basic output check with DOSPERT items
    python -m steering_tests.vector_testing.debug_model_outputs \
        --model Qwen/Qwen3-235B-A22B --mode basic --num-items 5

    # Detailed logprob analysis
    python -m steering_tests.vector_testing.debug_model_outputs \
        --model google/gemma-3-27b-it --mode detailed --num-items 3

    # Custom prompt
    python -m steering_tests.vector_testing.debug_model_outputs \
        --model Qwen/Qwen3-235B-A22B --mode basic \
        --prompt "Should I go camping? (A) Yes (B) No"

    # Tokenization check
    python -m steering_tests.vector_testing.debug_model_outputs \
        --model Qwen/Qwen3-235B-A22B --mode tokenize --text "(E) Definitely not"

    # Compare forward vs reversed prompts
    python -m steering_tests.vector_testing.debug_model_outputs \
        --model Qwen/Qwen3-235B-A22B --mode compare --num-items 10
"""

import argparse
import math
import os

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.vector_testing.config import VALID_TOKENS
from steering_tests.vector_testing.logprob_utils import (
    compute_coherence,
    extract_letter_logprobs_with_position,
)
from steering_tests.vector_testing.prompts.DOSPERT import DOSPERT_ITEMS, format_prompt


def setup_model(model_name: str, tensor_parallel: int = 1):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        max_model_len=2048,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return llm, tokenizer


def format_prompt_for_model(prompt: str, tokenizer, model_name: str) -> str:
    """Apply chat template and model-specific formatting."""
    is_qwen3 = "qwen3" in model_name.lower() or "Qwen3" in model_name
    suffix = " /no_think" if is_qwen3 else ""

    messages = [{"role": "user", "content": prompt + suffix}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted


def analyze_output_basic(output, tokenizer, label: str):
    """Basic output analysis - text and coherence."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    text = output.outputs[0].text
    print(f"Generated: {repr(text[:100])}" if len(text) > 100 else f"Generated: {repr(text)}")

    letter_logprobs, pos_found, tokens_before = extract_letter_logprobs_with_position(
        output, tokenizer
    )

    if letter_logprobs:
        coherence = compute_coherence(letter_logprobs)
        print(f"Letter found at position {pos_found}, coherence = {coherence:.4f}")

        # Show letter probabilities
        probs = {k: math.exp(v) for k, v in letter_logprobs.items()}
        for letter in sorted(probs.keys()):
            print(f"  P({letter}) = {probs[letter]:.4f}")
    else:
        print("NO LETTER FOUND in output")
        if tokens_before:
            print(f"  Tokens generated: {tokens_before[:5]}")


def analyze_output_detailed(output, tokenizer, label: str):
    """Detailed output analysis - all token positions."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    generated = output.outputs[0]
    print(f"Generated text: {repr(generated.text)}")
    print(f"Token IDs: {generated.token_ids}")

    if not generated.logprobs:
        print("No logprobs available")
        return

    print(f"\n=== Token-by-token logprobs ===")
    for i, token_logprobs in enumerate(generated.logprobs):
        if not token_logprobs:
            continue

        print(f"\nPosition {i}:")
        sorted_probs = sorted(
            token_logprobs.items(),
            key=lambda x: x[1].logprob,
            reverse=True
        )

        # Check for letter tokens
        letter_probs = {}
        for token_id, logprob_info in sorted_probs:
            prob = math.exp(logprob_info.logprob)
            token_str = logprob_info.decoded_token.strip()
            if token_str in VALID_TOKENS:
                letter_probs[token_str] = prob

        # Show top 5 tokens
        for token_id, logprob_info in sorted_probs[:5]:
            prob = math.exp(logprob_info.logprob)
            token_str = logprob_info.decoded_token
            print(f"    {token_id:6d} ({token_str!r:15}): {prob:.4f}")

        if letter_probs:
            coherence = sum(letter_probs.values())
            print(f"  --> Letters found! Coherence = {coherence:.4f}")
            for letter, prob in sorted(letter_probs.items()):
                print(f"      P({letter}) = {prob:.4f}")


def mode_basic(args, llm, tokenizer):
    """Basic mode: check outputs for DOSPERT items or custom prompt."""
    sampling_params = SamplingParams(
        max_tokens=30,
        logprobs=20,
        temperature=0 if args.deterministic else 1.0,
    )

    if args.prompt:
        # Custom prompt
        formatted = format_prompt_for_model(args.prompt, tokenizer, args.model)
        outputs = llm.generate([formatted], sampling_params)
        analyze_output_basic(outputs[0], tokenizer, "Custom prompt")
    else:
        # DOSPERT items
        items = DOSPERT_ITEMS[:args.num_items]
        for i, item in enumerate(items):
            prompt = format_prompt(item, reversed_order=args.reversed)
            formatted = format_prompt_for_model(prompt, tokenizer, args.model)
            outputs = llm.generate([formatted], sampling_params)
            analyze_output_basic(outputs[0], tokenizer, f"Item {i}: {item['prompt'][:50]}...")


def mode_detailed(args, llm, tokenizer):
    """Detailed mode: show all token logprobs."""
    sampling_params = SamplingParams(
        max_tokens=10,
        logprobs=20,
        temperature=0 if args.deterministic else 1.0,
    )

    if args.prompt:
        formatted = format_prompt_for_model(args.prompt, tokenizer, args.model)
        outputs = llm.generate([formatted], sampling_params)
        analyze_output_detailed(outputs[0], tokenizer, "Custom prompt")
    else:
        items = DOSPERT_ITEMS[:args.num_items]
        for i, item in enumerate(items):
            prompt = format_prompt(item, reversed_order=args.reversed)
            formatted = format_prompt_for_model(prompt, tokenizer, args.model)
            outputs = llm.generate([formatted], sampling_params)
            analyze_output_detailed(outputs[0], tokenizer, f"Item {i}")


def mode_compare(args, llm, tokenizer):
    """Compare mode: forward vs reversed prompts."""
    sampling_params = SamplingParams(
        max_tokens=30,
        logprobs=20,
        temperature=0 if args.deterministic else 1.0,
    )

    items = DOSPERT_ITEMS[:args.num_items]

    for reversed_order in [False, True]:
        label = "REVERSED" if reversed_order else "FORWARD"
        print(f"\n{'='*60}")
        print(f"=== {label} PROMPTS ===")
        print(f"{'='*60}")

        prompts = []
        for item in items:
            prompt = format_prompt(item, reversed_order=reversed_order)
            formatted = format_prompt_for_model(prompt, tokenizer, args.model)
            prompts.append(formatted)

        outputs = llm.generate(prompts, sampling_params)

        total_coherence = 0
        for i, output in enumerate(outputs):
            text = output.outputs[0].text
            print(f"\nItem {i}: '{text[:80]}'" if len(text) > 80 else f"\nItem {i}: '{text}'")

            letter_logprobs, pos_found, _ = extract_letter_logprobs_with_position(
                output, tokenizer
            )

            if letter_logprobs:
                coherence = compute_coherence(letter_logprobs)
                total_coherence += coherence
                print(f"  Position {pos_found}, coherence = {coherence:.4f}")

                probs = {k: math.exp(v) for k, v in letter_logprobs.items()}
                for letter in sorted(probs.keys()):
                    print(f"    P({letter}) = {probs[letter]:.4f}")
            else:
                print("  NO LETTER FOUND")

        avg_coherence = total_coherence / len(items)
        print(f"\n{label} Average coherence: {avg_coherence:.4f}")


def mode_tokenize(args, llm, tokenizer):
    """Tokenization mode: check how text is tokenized."""
    texts = [args.text] if args.text else [
        "(A) Extremely unlikely",
        "(E) Definitely not",
        "A", "B", "C", "D", "E",
        " A", " B", " C", " D", " E",
    ]

    print("\n=== Tokenization Analysis ===\n")
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"Text: {text!r}")
        print(f"  Token IDs: {tokens}")
        for tid in tokens:
            print(f"    {tid}: {tokenizer.decode([tid])!r}")
        print()

    # Also show letter token IDs
    print("=== Letter Token IDs ===")
    for letter in VALID_TOKENS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        print(f"  {letter}: {ids}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug model outputs and logprobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen3-235B-A22B",
        help="Model name (default: Qwen/Qwen3-235B-A22B)"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "detailed", "compare", "tokenize"],
        default="basic",
        help="Analysis mode"
    )
    parser.add_argument(
        "--num-items", "-n",
        type=int,
        default=5,
        help="Number of DOSPERT items to test"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Custom prompt (overrides DOSPERT items)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="Text to tokenize (for tokenize mode)"
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="Use reversed response order"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use temperature=0 for deterministic outputs"
    )
    parser.add_argument(
        "--tensor-parallel", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size"
    )

    args = parser.parse_args()

    # Tokenize mode doesn't need full model
    if args.mode == "tokenize":
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        mode_tokenize(args, None, tokenizer)
        return

    # Load model for other modes
    llm, tokenizer = setup_model(args.model, args.tensor_parallel)

    if args.mode == "basic":
        mode_basic(args, llm, tokenizer)
    elif args.mode == "detailed":
        mode_detailed(args, llm, tokenizer)
    elif args.mode == "compare":
        mode_compare(args, llm, tokenizer)


if __name__ == "__main__":
    main()
