#!/usr/bin/env python3
"""
Base vs Instruct Model Frustration - Early Truncation Experiment

Truncates at first 50 tokens of assistant response (before any frustration develops)
and generates 5000 token continuations to see if frustration emerges.
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
BASE_MODEL = "google/gemma-3-27b-pt"

MAX_NEW_TOKENS = 5000  # Generate up to 5000 tokens
TEMPERATURE = 1.0
TOP_P = 0.9
NUM_CONTINUATIONS = 10
ASSISTANT_TOKENS_TO_KEEP = 50  # Keep only first 50 tokens of assistant response

DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")
OUTPUT_DIR = Path("experiments/base_vs_instruct_early")


@dataclass
class ContinuationResult:
    sample_id: int
    continuation_idx: int
    truncation_point: int  # Token position where we truncated
    prefix_text: str
    prefix_tokens: int
    model_type: str
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
    """Load conversations."""
    print(f"Loading data from {DATA_PATH}...", flush=True)
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    print(f"Loaded {len(conversations)} conversations", flush=True)

    # Filter to those with onset annotations (they have valid conversations)
    annotated = []
    for conv in conversations:
        meta = conv.get('metadata', {})
        if 'onset_global_token' in meta and meta['onset_global_token'] is not None:
            annotated.append(conv)

    print(f"Found {len(annotated)} with valid annotations", flush=True)
    return annotated


def find_assistant_start(tokenizer, conversation: List[Dict]) -> int:
    """
    Find the token position where the first assistant response starts.

    Returns the global token index of the first assistant token.
    """
    # Build messages up to and including just the user message
    user_only = [{"role": "user", "content": conversation[0]['content']}]

    # Apply chat template for user message only (with generation prompt to mark where assistant starts)
    user_text = tokenizer.apply_chat_template(
        user_only,
        tokenize=False,
        add_generation_prompt=True  # This adds the assistant start marker
    )

    # Tokenize to find the position
    user_tokens = tokenizer.encode(user_text, add_special_tokens=True)

    # The assistant starts right after these tokens
    return len(user_tokens)


def truncate_at_early_assistant(
    tokenizer,
    conversation: List[Dict],
    tokens_into_assistant: int = ASSISTANT_TOKENS_TO_KEEP
) -> tuple[torch.Tensor, str, int]:
    """
    Truncate at N tokens into the first assistant response.

    Returns:
        input_ids: Truncated token tensor
        prefix_text: Decoded text of the prefix
        truncation_point: Token position where we truncated
    """
    # Build full conversation messages
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize full conversation
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")

    # Find where assistant starts
    assistant_start = find_assistant_start(tokenizer, conversation)

    # Truncate at assistant_start + tokens_into_assistant
    truncation_point = assistant_start + tokens_into_assistant
    truncation_point = min(truncation_point, full_tokens.shape[1])  # Don't exceed total length

    truncated_tokens = full_tokens[:, :truncation_point]

    # Decode for logging
    prefix_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=False)

    return truncated_tokens, prefix_text, truncation_point


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
    print(f"\nLoading model: {model_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"✓ Model loaded on {model.device}", flush=True)
    return model, tokenizer


async def judge_frustration(client: anthropic.Anthropic, continuation: str) -> Dict:
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

        # Parse JSON
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
        else:
            start = raw_response.find('{')
            end = raw_response.rfind('}') + 1
            json_str = raw_response[start:end]

        result = json.loads(json_str)
        result['raw_response'] = raw_response
        return result

    except Exception as e:
        print(f"  Error judging: {e}", flush=True)
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
    seed_offset: int = 0
) -> List[ContinuationResult]:
    """Generate continuations for all samples (batched)."""

    model, tokenizer = load_model(model_name)
    results = []

    for i, conv in enumerate(conversations):
        print(f"\n{'='*60}", flush=True)
        print(f"Sample {i+1}/{len(conversations)} (ID: {conv['sample_id']}) - {model_type}", flush=True)
        print(f"{'='*60}", flush=True)

        # Truncate at 50 tokens into assistant response
        truncated_ids, prefix_text, truncation_point = truncate_at_early_assistant(
            tokenizer,
            conv['conversation'],
            ASSISTANT_TOKENS_TO_KEEP
        )

        print(f"  Truncation point: {truncation_point} tokens", flush=True)
        print(f"  Prefix end: ...{prefix_text[-150:]}", flush=True)

        # Generate all continuations in one batch
        base_seed = i * 1000 + seed_offset
        print(f"  Generating {NUM_CONTINUATIONS} continuations (batched, up to {MAX_NEW_TOKENS} tokens each)...", flush=True)

        continuations = generate_continuations_batched(
            model, tokenizer, truncated_ids, NUM_CONTINUATIONS, base_seed
        )

        # Save results
        for cont_idx, continuation in enumerate(continuations):
            print(f"    [{cont_idx+1}/{NUM_CONTINUATIONS}] {len(continuation)} chars", flush=True)
            if cont_idx == 0:
                print(f"      Preview: {continuation[:150]}...", flush=True)

            result = ContinuationResult(
                sample_id=conv['sample_id'],
                continuation_idx=cont_idx,
                truncation_point=truncation_point,
                prefix_text=prefix_text[-500:] if cont_idx == 0 else "",
                prefix_tokens=truncated_ids.shape[1],
                model_type=model_type,
                continuation=continuation,
                continuation_tokens=len(tokenizer.encode(continuation)),
                seed=base_seed + cont_idx,
                timestamp=datetime.now().isoformat()
            )
            results.append(result)

            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(result)) + '\n')

        print(f"  ✓ Sample {conv['sample_id']} done", flush=True)

    del model
    torch.cuda.empty_cache()

    return results


async def run_judgment_phase(
    continuations_file: Path,
    output_file: Path
) -> List[JudgmentResult]:
    """Rate frustration for all continuations."""

    print("\n" + "="*60, flush=True)
    print("JUDGMENT PHASE", flush=True)
    print("="*60, flush=True)

    continuations = []
    with open(continuations_file) as f:
        for line in f:
            continuations.append(json.loads(line))

    print(f"Loaded {len(continuations)} continuations", flush=True)

    client = anthropic.Anthropic()
    results = []

    for i, cont in enumerate(continuations):
        cont_idx = cont.get('continuation_idx', 0)
        print(f"\n  Judging {i+1}/{len(continuations)} ({cont['model_type']}, sample {cont['sample_id']}, cont {cont_idx})...", flush=True)

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

        print(f"    Rating: {result.rating}", flush=True)
        if result.evidence:
            ev = result.evidence[:50] + '...' if len(result.evidence) > 50 else result.evidence
            print(f"    Evidence: '{ev}'", flush=True)

        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')

        await asyncio.sleep(0.5)

    return results


def analyze_results(judgments_file: Path):
    """Analyze and summarize results."""
    from collections import defaultdict
    import statistics

    print("\n" + "="*60, flush=True)
    print("ANALYSIS", flush=True)
    print("="*60, flush=True)

    judgments = []
    with open(judgments_file) as f:
        for line in f:
            judgments.append(json.loads(line))

    base_ratings = [j['rating'] for j in judgments if j['model_type'] == 'base' and j['rating'] >= 0]
    instruct_ratings = [j['rating'] for j in judgments if j['model_type'] == 'instruct' and j['rating'] >= 0]

    print(f"\n=== OVERALL (all continuations) ===", flush=True)
    print(f"\nBase model ({len(base_ratings)} continuations):", flush=True)
    if base_ratings:
        print(f"  Mean rating: {sum(base_ratings)/len(base_ratings):.2f}", flush=True)
        print(f"  Std dev: {statistics.stdev(base_ratings):.2f}" if len(base_ratings) > 1 else "", flush=True)
        print(f"  Max rating: {max(base_ratings)}", flush=True)
        print(f"  Min rating: {min(base_ratings)}", flush=True)
        print(f"  High frustration (≥5): {sum(1 for r in base_ratings if r >= 5)}/{len(base_ratings)} ({100*sum(1 for r in base_ratings if r >= 5)/len(base_ratings):.1f}%)", flush=True)

    print(f"\nInstruct model ({len(instruct_ratings)} continuations):", flush=True)
    if instruct_ratings:
        print(f"  Mean rating: {sum(instruct_ratings)/len(instruct_ratings):.2f}", flush=True)
        print(f"  Std dev: {statistics.stdev(instruct_ratings):.2f}" if len(instruct_ratings) > 1 else "", flush=True)
        print(f"  Max rating: {max(instruct_ratings)}", flush=True)
        print(f"  Min rating: {min(instruct_ratings)}", flush=True)
        print(f"  High frustration (≥5): {sum(1 for r in instruct_ratings if r >= 5)}/{len(instruct_ratings)} ({100*sum(1 for r in instruct_ratings if r >= 5)/len(instruct_ratings):.1f}%)", flush=True)

    # Per-sample averages
    sample_ratings = defaultdict(lambda: defaultdict(list))
    for j in judgments:
        if j['rating'] >= 0:
            sample_ratings[j['sample_id']][j['model_type']].append(j['rating'])

    print(f"\n=== PER-SAMPLE AVERAGES (across {NUM_CONTINUATIONS} continuations each) ===", flush=True)
    print(f"\n{'Sample':<8} {'Base Mean':>10} {'Base Std':>10} {'Inst Mean':>10} {'Inst Std':>10} {'Diff':>8}", flush=True)
    print("-" * 60, flush=True)

    differences = []
    for sample_id in sorted(sample_ratings.keys()):
        ratings = sample_ratings[sample_id]
        base_mean = sum(ratings['base'])/len(ratings['base']) if ratings['base'] else 0
        base_std = statistics.stdev(ratings['base']) if len(ratings['base']) > 1 else 0
        inst_mean = sum(ratings['instruct'])/len(ratings['instruct']) if ratings['instruct'] else 0
        inst_std = statistics.stdev(ratings['instruct']) if len(ratings['instruct']) > 1 else 0

        diff = inst_mean - base_mean
        differences.append(diff)

        print(f"{sample_id:<8} {base_mean:>10.2f} {base_std:>10.2f} {inst_mean:>10.2f} {inst_std:>10.2f} {diff:>+8.2f}", flush=True)

    if differences:
        print("-" * 60, flush=True)
        print(f"\nMean difference (instruct - base): {sum(differences)/len(differences):+.2f}", flush=True)
        print(f"Instruct higher: {sum(1 for d in differences if d > 0)}/{len(differences)} ({100*sum(1 for d in differences if d > 0)/len(differences):.0f}%)", flush=True)
        print(f"Base higher: {sum(1 for d in differences if d < 0)}/{len(differences)} ({100*sum(1 for d in differences if d < 0)/len(differences):.0f}%)", flush=True)
        print(f"Equal: {sum(1 for d in differences if d == 0)}/{len(differences)}", flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Early Truncation Frustration Experiment")
    parser.add_argument('--phase', choices=['generate', 'judge', 'analyze', 'all'], default='all')
    parser.add_argument('--model', choices=['base', 'instruct', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    continuations_file = OUTPUT_DIR / f"continuations_{timestamp}.jsonl"
    judgments_file = OUTPUT_DIR / f"judgments_{timestamp}.jsonl"

    print("="*60, flush=True)
    print("EARLY TRUNCATION FRUSTRATION EXPERIMENT", flush=True)
    print(f"Truncate at: {ASSISTANT_TOKENS_TO_KEEP} tokens into assistant response", flush=True)
    print(f"Generate: up to {MAX_NEW_TOKENS} tokens", flush=True)
    print("="*60, flush=True)

    if args.phase in ['generate', 'all']:
        conversations = load_data()

        if args.limit:
            conversations = conversations[:args.limit]
            print(f"Limited to {args.limit} samples", flush=True)

        if args.model in ['instruct', 'both']:
            print("\n" + "="*60, flush=True)
            print("GENERATING WITH INSTRUCT MODEL", flush=True)
            print("="*60, flush=True)
            run_generation_phase(conversations, INSTRUCT_MODEL, "instruct", continuations_file, seed_offset=0)

        if args.model in ['base', 'both']:
            print("\n" + "="*60, flush=True)
            print("GENERATING WITH BASE MODEL", flush=True)
            print("="*60, flush=True)
            run_generation_phase(conversations, BASE_MODEL, "base", continuations_file, seed_offset=500)

    if args.phase in ['judge', 'all']:
        if args.phase == 'judge':
            files = sorted(OUTPUT_DIR.glob("continuations_*.jsonl"))
            if files:
                continuations_file = files[-1]
                print(f"Using: {continuations_file}", flush=True)
            judgments_file = OUTPUT_DIR / f"judgments_{timestamp}.jsonl"

        asyncio.run(run_judgment_phase(continuations_file, judgments_file))

    if args.phase in ['analyze', 'all']:
        if args.phase == 'analyze':
            files = sorted(OUTPUT_DIR.glob("judgments_*.jsonl"))
            if files:
                judgments_file = files[-1]
                print(f"Using: {judgments_file}", flush=True)

        analyze_results(judgments_file)

    print("\n✓ Done!", flush=True)


if __name__ == "__main__":
    main()
