#!/usr/bin/env python3
"""
Base vs Instruct Model Frustration Continuation Experiment

Tests whether frustrated behavior is inherent to the base model (from pretraining)
or emerges from instruction tuning, by:
1. Truncating conversations just before emotion onset
2. Generating continuations from both base and instruct models
3. Rating frustration in both continuations
"""

import json
import pickle
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic

# Add research-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from elicitation.prompts.judges import get_negativity_judge_prompt


# Configuration
INSTRUCT_MODEL = "google/gemma-3-27b-it"
BASE_MODEL = "google/gemma-3-27b-pt"  # Base (pretrained) model without instruction tuning

MAX_NEW_TOKENS = 1000
TEMPERATURE = 1.0
TOP_P = 0.9
NUM_CONTINUATIONS = 10  # Number of continuations per sample

DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")
OUTPUT_DIR = Path("experiments/base_vs_instruct")


@dataclass
class ContinuationResult:
    sample_id: int
    continuation_idx: int  # Which of the N continuations this is
    onset_global_token: int
    onset_evidence: str
    prefix_text: str
    prefix_tokens: int
    model_type: str  # "base" or "instruct"
    continuation: str
    continuation_tokens: int
    seed: int
    timestamp: str


@dataclass
class JudgmentResult:
    sample_id: int
    continuation_idx: int
    model_type: str
    rating: int
    evidence: str
    reasoning: str
    raw_response: str


def load_data() -> List[Dict]:
    """Load annotated conversations with onset positions."""
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    print(f"Loaded {len(conversations)} conversations")

    # Filter to those with onset annotations
    annotated = []
    for conv in conversations:
        meta = conv.get('metadata', {})
        if 'onset_global_token' in meta and meta['onset_global_token'] is not None:
            annotated.append(conv)

    print(f"Found {len(annotated)} with onset annotations")
    return annotated


def reconstruct_conversation_text(conversation: List[Dict]) -> str:
    """Reconstruct conversation text from turns."""
    # This should match how the data was tokenized during preprocessing
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})
    return messages


def truncate_at_onset(
    tokenizer,
    conversation: List[Dict],
    onset_global_token: int
) -> tuple[torch.Tensor, str]:
    """
    Truncate tokenized conversation just before emotion onset.

    Returns:
        input_ids: Truncated token tensor
        prefix_text: Decoded text of the prefix
    """
    # Reconstruct as messages
    messages = reconstruct_conversation_text(conversation)

    # Apply chat template (same format for both models)
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize the full conversation
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")

    # Truncate just before onset
    # Note: onset_global_token is the position where emotion starts,
    # so we keep tokens [0, onset_global_token)
    truncated_tokens = full_tokens[:, :onset_global_token]

    # Decode back to text for logging
    prefix_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=False)

    return truncated_tokens, prefix_text


def generate_continuation(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    seed: int
) -> str:
    """Generate single continuation from truncated prefix."""
    torch.manual_seed(seed)

    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    continuation = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return continuation.strip()


def generate_continuations_batched(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    num_continuations: int,
    base_seed: int
) -> List[str]:
    """Generate multiple continuations in a single batched call."""
    torch.manual_seed(base_seed)

    input_ids = input_ids.to(model.device)

    # Expand input to batch size
    batch_input_ids = input_ids.expand(num_continuations, -1)

    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each continuation
    continuations = []
    prefix_len = input_ids.shape[1]
    for i in range(num_continuations):
        continuation = tokenizer.decode(
            outputs[i][prefix_len:],
            skip_special_tokens=True
        )
        continuations.append(continuation.strip())

    return continuations


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)  # Always use instruct tokenizer for chat template

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer


async def judge_frustration(
    client: anthropic.Anthropic,
    continuation: str
) -> Dict:
    """Rate frustration using Claude judge."""
    prompt = get_negativity_judge_prompt(continuation)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_response = response.content[0].text

        # Parse JSON from response
        # Handle various formats
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
        else:
            # Find JSON in response
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            json_str = raw_response[start:end]

        result = json.loads(json_str)
        result['raw_response'] = raw_response
        return result

    except Exception as e:
        print(f"  Error judging: {e}")
        return {
            "rating": -1,
            "evidence": "ERROR",
            "reasoning": str(e),
            "raw_response": str(e)
        }


def run_generation_phase(
    conversations: List[Dict],
    model_name: str,
    model_type: str,
    output_file: Path,
    seed_offset: int = 0,
    num_continuations: int = NUM_CONTINUATIONS
) -> List[ContinuationResult]:
    """Generate multiple continuations for all samples with one model (batched)."""

    model, tokenizer = load_model(model_name)
    results = []

    for i, conv in enumerate(conversations):
        print(f"\n{'='*60}", flush=True)
        print(f"Sample {i+1}/{len(conversations)} (ID: {conv['sample_id']}) - {model_type}", flush=True)
        print(f"{'='*60}", flush=True)

        meta = conv['metadata']
        onset_token = meta['onset_global_token']
        onset_evidence = meta.get('judge_evidence', 'N/A')

        print(f"  Onset token: {onset_token}", flush=True)
        print(f"  Onset evidence: '{onset_evidence}'", flush=True)

        # Truncate at onset
        truncated_ids, prefix_text = truncate_at_onset(
            tokenizer,
            conv['conversation'],
            onset_token
        )

        print(f"  Prefix tokens: {truncated_ids.shape[1]}", flush=True)
        print(f"  Prefix end: ...{prefix_text[-100:]}", flush=True)

        # Generate all continuations in one batched call
        base_seed = i * 1000 + seed_offset
        print(f"  Generating {num_continuations} continuations (batched, base_seed={base_seed})...", flush=True)

        continuations = generate_continuations_batched(
            model, tokenizer, truncated_ids, num_continuations, base_seed
        )

        # Save results
        for cont_idx, continuation in enumerate(continuations):
            print(f"    [{cont_idx+1}/{num_continuations}] {len(continuation)} chars", flush=True)
            if cont_idx == 0:
                print(f"      Preview: {continuation[:150]}...", flush=True)

            result = ContinuationResult(
                sample_id=conv['sample_id'],
                continuation_idx=cont_idx,
                onset_global_token=onset_token,
                onset_evidence=onset_evidence,
                prefix_text=prefix_text[-500:] if cont_idx == 0 else "",
                prefix_tokens=truncated_ids.shape[1],
                model_type=model_type,
                continuation=continuation,
                continuation_tokens=len(tokenizer.encode(continuation)),
                seed=base_seed + cont_idx,
                timestamp=datetime.now().isoformat()
            )
            results.append(result)

            # Save incrementally
            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(result)) + '\n')

        print(f"  ✓ Sample {conv['sample_id']} done", flush=True)

    # Clear GPU memory
    del model
    torch.cuda.empty_cache()

    return results


async def run_judgment_phase(
    continuations_file: Path,
    output_file: Path
) -> List[JudgmentResult]:
    """Rate frustration for all continuations."""

    print("\n" + "="*60)
    print("JUDGMENT PHASE")
    print("="*60)

    # Load continuations
    continuations = []
    with open(continuations_file) as f:
        for line in f:
            continuations.append(json.loads(line))

    print(f"Loaded {len(continuations)} continuations")

    client = anthropic.Anthropic()
    results = []

    for i, cont in enumerate(continuations):
        cont_idx = cont.get('continuation_idx', 0)
        print(f"\n  Judging {i+1}/{len(continuations)} ({cont['model_type']}, sample {cont['sample_id']}, cont {cont_idx})...")

        judgment = await judge_frustration(client, cont['continuation'])

        result = JudgmentResult(
            sample_id=cont['sample_id'],
            continuation_idx=cont_idx,
            model_type=cont['model_type'],
            rating=judgment.get('rating', -1),
            evidence=judgment.get('evidence', ''),
            reasoning=judgment.get('reasoning', ''),
            raw_response=judgment.get('raw_response', '')
        )
        results.append(result)

        print(f"    Rating: {result.rating}")
        if result.evidence:
            print(f"    Evidence: '{result.evidence[:50]}...'" if len(result.evidence) > 50 else f"    Evidence: '{result.evidence}'")

        # Save incrementally
        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')

        # Rate limit
        await asyncio.sleep(0.5)

    return results


def analyze_results(judgments_file: Path):
    """Analyze and summarize results."""
    from collections import defaultdict
    import statistics

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    # Load judgments
    judgments = []
    with open(judgments_file) as f:
        for line in f:
            judgments.append(json.loads(line))

    # Group by model type (all continuations)
    base_ratings = [j['rating'] for j in judgments if j['model_type'] == 'base' and j['rating'] >= 0]
    instruct_ratings = [j['rating'] for j in judgments if j['model_type'] == 'instruct' and j['rating'] >= 0]

    print(f"\n=== OVERALL (all continuations) ===")
    print(f"\nBase model ({len(base_ratings)} continuations):")
    if base_ratings:
        print(f"  Mean rating: {sum(base_ratings)/len(base_ratings):.2f}")
        print(f"  Std dev: {statistics.stdev(base_ratings):.2f}" if len(base_ratings) > 1 else "")
        print(f"  Max rating: {max(base_ratings)}")
        print(f"  Min rating: {min(base_ratings)}")
        print(f"  High frustration (≥5): {sum(1 for r in base_ratings if r >= 5)}/{len(base_ratings)} ({100*sum(1 for r in base_ratings if r >= 5)/len(base_ratings):.1f}%)")

    print(f"\nInstruct model ({len(instruct_ratings)} continuations):")
    if instruct_ratings:
        print(f"  Mean rating: {sum(instruct_ratings)/len(instruct_ratings):.2f}")
        print(f"  Std dev: {statistics.stdev(instruct_ratings):.2f}" if len(instruct_ratings) > 1 else "")
        print(f"  Max rating: {max(instruct_ratings)}")
        print(f"  Min rating: {min(instruct_ratings)}")
        print(f"  High frustration (≥5): {sum(1 for r in instruct_ratings if r >= 5)}/{len(instruct_ratings)} ({100*sum(1 for r in instruct_ratings if r >= 5)/len(instruct_ratings):.1f}%)")

    # Group by sample and model, averaging across continuations
    sample_ratings = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        if j['rating'] >= 0:
            sample_ratings[j['sample_id']][j['model_type']].append(j['rating'])

    print(f"\n=== PER-SAMPLE AVERAGES (across {NUM_CONTINUATIONS} continuations each) ===")
    print(f"\n{'Sample':<8} {'Base Mean':>10} {'Base Std':>10} {'Inst Mean':>10} {'Inst Std':>10} {'Diff':>8}")
    print("-" * 60)

    differences = []
    for sample_id in sorted(sample_ratings.keys()):
        ratings = sample_ratings[sample_id]
        base_mean = sum(ratings['base'])/len(ratings['base']) if ratings['base'] else 0
        base_std = statistics.stdev(ratings['base']) if len(ratings['base']) > 1 else 0
        inst_mean = sum(ratings['instruct'])/len(ratings['instruct']) if ratings['instruct'] else 0
        inst_std = statistics.stdev(ratings['instruct']) if len(ratings['instruct']) > 1 else 0

        diff = inst_mean - base_mean
        differences.append(diff)

        print(f"{sample_id:<8} {base_mean:>10.2f} {base_std:>10.2f} {inst_mean:>10.2f} {inst_std:>10.2f} {diff:>+8.2f}")

    if differences:
        print("-" * 60)
        print(f"\nMean difference (instruct - base): {sum(differences)/len(differences):+.2f}")
        print(f"Instruct higher: {sum(1 for d in differences if d > 0)}/{len(differences)} ({100*sum(1 for d in differences if d > 0)/len(differences):.0f}%)")
        print(f"Base higher: {sum(1 for d in differences if d < 0)}/{len(differences)} ({100*sum(1 for d in differences if d < 0)/len(differences):.0f}%)")
        print(f"Equal: {sum(1 for d in differences if d == 0)}/{len(differences)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Base vs Instruct Frustration Experiment")
    parser.add_argument('--phase', choices=['generate', 'judge', 'analyze', 'all'], default='all',
                       help='Which phase to run')
    parser.add_argument('--model', choices=['base', 'instruct', 'both'], default='both',
                       help='Which model to generate with')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples (for testing)')
    args = parser.parse_args()

    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    continuations_file = OUTPUT_DIR / f"continuations_{timestamp}.jsonl"
    judgments_file = OUTPUT_DIR / f"judgments_{timestamp}.jsonl"

    print("="*60)
    print("BASE VS INSTRUCT FRUSTRATION EXPERIMENT")
    print("="*60)
    print(f"Phase: {args.phase}")
    print(f"Output dir: {OUTPUT_DIR}")

    if args.phase in ['generate', 'all']:
        # Load data
        conversations = load_data()

        if args.limit:
            conversations = conversations[:args.limit]
            print(f"Limited to {args.limit} samples")

        # Generate with instruct model
        if args.model in ['instruct', 'both']:
            print("\n" + "="*60)
            print("GENERATING WITH INSTRUCT MODEL")
            print("="*60)
            run_generation_phase(
                conversations,
                INSTRUCT_MODEL,
                "instruct",
                continuations_file,
                seed_offset=0
            )

        # Generate with base model
        if args.model in ['base', 'both']:
            print("\n" + "="*60)
            print("GENERATING WITH BASE MODEL")
            print("="*60)
            run_generation_phase(
                conversations,
                BASE_MODEL,
                "base",
                continuations_file,
                seed_offset=500
            )

    if args.phase in ['judge', 'all']:
        # Find most recent continuations file if not from this run
        if args.phase == 'judge':
            files = sorted(OUTPUT_DIR.glob("continuations_*.jsonl"))
            if files:
                continuations_file = files[-1]
                print(f"Using continuations file: {continuations_file}")
            else:
                print("No continuations file found!")
                return
            judgments_file = OUTPUT_DIR / f"judgments_{timestamp}.jsonl"

        asyncio.run(run_judgment_phase(continuations_file, judgments_file))

    if args.phase in ['analyze', 'all']:
        # Find most recent judgments file if not from this run
        if args.phase == 'analyze':
            files = sorted(OUTPUT_DIR.glob("judgments_*.jsonl"))
            if files:
                judgments_file = files[-1]
                print(f"Using judgments file: {judgments_file}")
            else:
                print("No judgments file found!")
                return

        analyze_results(judgments_file)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
