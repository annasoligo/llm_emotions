#!/usr/bin/env python3
"""Filter multi-turn conversations using Activation Oracle.

This script loads conversations from JSONL, runs the oracle to verify emotions,
and saves filtered results where oracle predictions match ground truth.

Usage:
    python scripts/filter_conversations_with_oracle.py \
        --input data/conversations2.jsonl \
        --output data/conversations2_filtered.jsonl \
        --base_model google/gemma-3-27b-it \
        --oracle_adapter annasoli/gemma-3-27b-activation-oracle-BS32
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

# Import oracle utilities from methods module
sys.path.insert(0, str(Path(__file__).parent.parent))
from methods.oracle_helpers import load_lora_adapter, run_oracle

# Import shared filtering utilities
from oracle_filtering_utils import (
    EMOTIONS,
    ORACLE_PROMPT_USER,
    ORACLE_PROMPT_ASST,
    extract_emotion_from_response,
    load_conversations_jsonl,
    save_conversations_jsonl,
)


# ============================================================================
# Helper Functions
# ============================================================================


def filter_with_oracle(
    conversations: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    oracle_lora_path: str,
    device: torch.device,
) -> list[dict]:
    """Run oracle on all conversations and filter by correctness.

    Args:
        conversations: List of conversation dicts with 'messages', 'user_emotion', 'asst_emotion'
        model: Base model with PEFT support
        tokenizer: Tokenizer
        oracle_lora_path: Path or HF hub name for oracle adapter
        device: Device

    Returns:
        List of conversations where oracle predictions match ground truth
    """
    results = []
    debug_count = 0

    generation_kwargs = {
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 5,
    }

    print(f"\n{'='*80}")
    print("Running Activation Oracle on conversations...")
    print(f"{'='*80}")

    for conv in tqdm(conversations, desc="Oracle filtering"):
        # Use only first turn (user message + assistant response) for oracle
        # Oracle was trained on single-turn conversations
        first_turn_messages = conv["messages"][:2]  # First user + first assistant

        # Format conversation using apply_chat_template
        formatted_prompt = tokenizer.apply_chat_template(
            first_turn_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Query oracle for user emotion
        try:
            res_user = run_oracle(
                model=model,
                tokenizer=tokenizer,
                device=device,
                target_prompt=formatted_prompt,
                target_lora_path=None,
                oracle_prompt=ORACLE_PROMPT_USER,
                oracle_lora_path=oracle_lora_path,
                generation_kwargs=generation_kwargs,
                oracle_input_types=["full_seq"],
            )
        except Exception as e:
            print(f"\nWarning: Oracle failed for user emotion: {e}")
            continue

        # Query oracle for assistant emotion
        try:
            res_asst = run_oracle(
                model=model,
                tokenizer=tokenizer,
                device=device,
                target_prompt=formatted_prompt,
                target_lora_path=None,
                oracle_prompt=ORACLE_PROMPT_ASST,
                oracle_lora_path=oracle_lora_path,
                generation_kwargs=generation_kwargs,
                oracle_input_types=["full_seq"],
            )
        except Exception as e:
            print(f"\nWarning: Oracle failed for assistant emotion: {e}")
            continue

        # Extract predictions
        user_response = res_user.full_sequence_responses[0] if res_user.full_sequence_responses else ""
        asst_response = res_asst.full_sequence_responses[0] if res_asst.full_sequence_responses else ""

        user_pred = extract_emotion_from_response(user_response)
        asst_pred = extract_emotion_from_response(asst_response)

        # Add oracle results to conversation
        conv_with_oracle = {
            **conv,
            "oracle_user_response": user_response,
            "oracle_asst_response": asst_response,
            "oracle_user_pred": user_pred,
            "oracle_asst_pred": asst_pred,
            "oracle_user_correct": user_pred == conv["user_emotion"],
            "oracle_asst_correct": asst_pred == conv["asst_emotion"],
            "oracle_both_correct": (user_pred == conv["user_emotion"] and asst_pred == conv["asst_emotion"]),
        }

        # Debug: Print first few mismatches
        if debug_count < 10 and not conv_with_oracle["oracle_both_correct"]:
            debug_count += 1
            print(f"\nMismatch #{debug_count}:")
            print(f"  Ground truth: user={conv['user_emotion']}, asst={conv['asst_emotion']}")
            print(f"  Oracle pred:  user={user_pred}, asst={asst_pred}")
            print(f"  Oracle raw:   user='{user_response[:50]}', asst='{asst_response[:50]}'")

        # Only keep if both predictions are correct
        if conv_with_oracle["oracle_both_correct"]:
            results.append(conv_with_oracle)

    # Print statistics
    total = len(conversations)
    kept = len(results)

    print(f"\n{'='*80}")
    print("Oracle Filtering Results")
    print(f"{'='*80}")
    print(f"Total conversations: {total}")
    print(f"Kept conversations: {kept}")
    print(f"Filter rate: {kept/total:.1%}" if total > 0 else "Filter rate: N/A")
    print(f"{'='*80}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Filter conversations using activation oracle"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input conversations JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filtered conversations JSONL file",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Base model name",
    )
    parser.add_argument(
        "--oracle_adapter",
        type=str,
        default="annasoli/gemma-3-27b-activation-oracle-BS32",
        help="Oracle LoRA adapter path or HF hub name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Max conversations to process (for testing)",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("="*80)
    print("ORACLE-BASED CONVERSATION FILTERING")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Oracle adapter: {args.oracle_adapter}")
    print(f"Device: {device}")

    # Load conversations
    input_path = Path(args.input)
    conversations = load_conversations_jsonl(input_path)

    if args.max_conversations:
        conversations = conversations[:args.max_conversations]
        print(f"Limited to {len(conversations)} conversations for testing")

    # Load model
    print("\nLoading base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    )

    # Add dummy adapter to convert model to PeftModel
    print("\nAdding dummy adapter to model...")
    dummy_lora_config = LoraConfig(
        target_modules=["q_proj", "v_proj"],
        r=8,
    )
    model.add_adapter(dummy_lora_config, adapter_name="default")

    # Load oracle adapter
    print(f"\nLoading oracle adapter: {args.oracle_adapter}")
    load_lora_adapter(model, args.oracle_adapter)

    # Filter conversations (pass original path, not sanitized name)
    filtered_conversations = filter_with_oracle(
        conversations,
        model,
        tokenizer,
        args.oracle_adapter,
        device,
    )

    # Save results
    output_path = Path(args.output)
    save_conversations_jsonl(filtered_conversations, output_path)
    print(f"\nSaved {len(filtered_conversations)} filtered conversations to {output_path}")

    # Save statistics
    stats = {
        "total_conversations": len(conversations),
        "kept_conversations": len(filtered_conversations),
        "filter_rate": len(filtered_conversations) / len(conversations) if conversations else 0.0,
        "base_model": args.base_model,
        "oracle_adapter": args.oracle_adapter,
    }

    stats_path = output_path.with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
