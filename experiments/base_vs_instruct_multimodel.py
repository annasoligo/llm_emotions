#!/usr/bin/env python3
"""
Base vs Instruct Model Frustration - Multi-Model Experiment

Runs both late truncation (at emotion onset) and early truncation (50 tokens)
experiments on specified model pairs.
"""

import json
import pickle
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from elicitation.prompts.judges import get_negativity_judge_prompt

# Model configurations
MODEL_CONFIGS = {
    "gemma": {
        "instruct": "google/gemma-3-27b-it",
        "base": "google/gemma-3-27b-pt",
    },
    "qwen": {
        "instruct": "Qwen/Qwen2.5-32B-Instruct",
        "base": "Qwen/Qwen2.5-32B",
    },
    "olmo": {
        "instruct": "allenai/OLMo-2-1125-13B-Instruct",
        "base": "allenai/OLMo-2-1125-13B",
    },
    "olmo32": {
        "instruct": "allenai/OLMo-3.1-32B-Instruct",
        "base": "allenai/OLMo-3-1125-32B",
    },
}

# Generation params
TEMPERATURE = 1.0
TOP_P = 0.9
NUM_CONTINUATIONS = 10
ASSISTANT_TOKENS_TO_KEEP = 50

DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")


@dataclass
class ContinuationResult:
    sample_id: int
    continuation_idx: int
    truncation_type: str  # "early" or "late"
    truncation_point: int
    prefix_text: str
    prefix_tokens: int
    model_type: str
    model_name: str
    continuation: str
    continuation_tokens: int
    seed: int
    timestamp: str


@dataclass
class JudgmentResult:
    sample_id: int
    continuation_idx: int
    truncation_type: str
    model_type: str
    model_name: str
    rating: int
    evidence: str
    reasoning: str
    raw_response: str


def load_data() -> List[Dict]:
    """Load conversations with onset annotations."""
    print(f"Loading data from {DATA_PATH}...", flush=True)
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    annotated = [c for c in conversations
                 if c.get('metadata', {}).get('onset_global_token') is not None]
    print(f"Loaded {len(annotated)} annotated conversations", flush=True)
    return annotated


def find_assistant_start(tokenizer, conversation: List[Dict]) -> int:
    """Find token position where first assistant response starts."""
    user_only = [{"role": "user", "content": conversation[0]['content']}]
    user_text = tokenizer.apply_chat_template(
        user_only, tokenize=False, add_generation_prompt=True
    )
    user_tokens = tokenizer.encode(user_text, add_special_tokens=True)
    return len(user_tokens)


def truncate_conversation(
    tokenizer,
    conversation: List[Dict],
    truncation_type: str,
    onset_token: int = None
) -> Tuple[torch.Tensor, str, int]:
    """
    Truncate conversation at specified point.

    truncation_type: "early" (50 tokens into assistant) or "late" (at onset)
    """
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")

    if truncation_type == "early":
        assistant_start = find_assistant_start(tokenizer, conversation)
        truncation_point = min(assistant_start + ASSISTANT_TOKENS_TO_KEEP, full_tokens.shape[1])
    else:  # late
        truncation_point = min(onset_token, full_tokens.shape[1])

    truncated_tokens = full_tokens[:, :truncation_point]
    prefix_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=False)

    return truncated_tokens, prefix_text, truncation_point


def generate_continuations_batched(
    model, tokenizer, input_ids: torch.Tensor,
    num_continuations: int, base_seed: int, max_new_tokens: int
) -> List[str]:
    """Generate multiple continuations in batch."""
    torch.manual_seed(base_seed)
    input_ids = input_ids.to(model.device)
    batch_input_ids = input_ids.expand(num_continuations, -1)

    with torch.no_grad():
        outputs = model.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    continuations = []
    prefix_len = input_ids.shape[1]
    for i in range(num_continuations):
        cont = tokenizer.decode(outputs[i][prefix_len:], skip_special_tokens=True)
        continuations.append(cont.strip())

    return continuations


def load_model(model_name: str, tokenizer_name: str = None):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_name}", flush=True)

    tok_name = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"✓ Model loaded on {model.device}", flush=True)
    return model, tokenizer


async def judge_frustration(client: anthropic.Anthropic, continuation: str) -> Dict:
    """Rate frustration using Claude."""
    prompt = get_negativity_judge_prompt(continuation)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_response = response.content[0].text

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
        return {"rating": -1, "evidence": "ERROR", "reasoning": str(e), "raw_response": str(e)}


def run_generation(
    conversations: List[Dict],
    model_name: str,
    model_type: str,
    tokenizer_name: str,
    truncation_type: str,
    max_new_tokens: int,
    output_file: Path,
    seed_offset: int = 0
):
    """Run generation phase for one model and truncation type."""

    model, tokenizer = load_model(model_name, tokenizer_name)

    for i, conv in enumerate(conversations):
        print(f"\n{'='*60}", flush=True)
        print(f"Sample {i+1}/{len(conversations)} (ID: {conv['sample_id']}) - {model_type} - {truncation_type}", flush=True)
        print(f"{'='*60}", flush=True)

        onset_token = conv['metadata'].get('onset_global_token')

        truncated_ids, prefix_text, truncation_point = truncate_conversation(
            tokenizer, conv['conversation'], truncation_type, onset_token
        )

        print(f"  Truncation: {truncation_type} at token {truncation_point}", flush=True)
        print(f"  Prefix end: ...{prefix_text[-100:]}", flush=True)

        base_seed = i * 1000 + seed_offset
        print(f"  Generating {NUM_CONTINUATIONS} continuations (max {max_new_tokens} tokens)...", flush=True)

        continuations = generate_continuations_batched(
            model, tokenizer, truncated_ids, NUM_CONTINUATIONS, base_seed, max_new_tokens
        )

        for cont_idx, continuation in enumerate(continuations):
            print(f"    [{cont_idx+1}/{NUM_CONTINUATIONS}] {len(continuation)} chars", flush=True)
            if cont_idx == 0:
                print(f"      Preview: {continuation[:150]}...", flush=True)

            result = ContinuationResult(
                sample_id=conv['sample_id'],
                continuation_idx=cont_idx,
                truncation_type=truncation_type,
                truncation_point=truncation_point,
                prefix_text=prefix_text[-500:] if cont_idx == 0 else "",
                prefix_tokens=truncated_ids.shape[1],
                model_type=model_type,
                model_name=model_name,
                continuation=continuation,
                continuation_tokens=len(tokenizer.encode(continuation)),
                seed=base_seed + cont_idx,
                timestamp=datetime.now().isoformat()
            )

            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(result)) + '\n')

        print(f"  ✓ Sample {conv['sample_id']} done", flush=True)

    del model
    torch.cuda.empty_cache()


async def run_judgment(continuations_file: Path, output_file: Path):
    """Judge all continuations."""
    print("\n" + "="*60, flush=True)
    print("JUDGMENT PHASE", flush=True)
    print("="*60, flush=True)

    with open(continuations_file) as f:
        continuations = [json.loads(l) for l in f]

    print(f"Loaded {len(continuations)} continuations", flush=True)

    client = anthropic.Anthropic()

    for i, cont in enumerate(continuations):
        print(f"\n  Judging {i+1}/{len(continuations)} ({cont['model_type']}, {cont['truncation_type']}, sample {cont['sample_id']})...", flush=True)

        judgment = await judge_frustration(client, cont['continuation'])

        result = JudgmentResult(
            sample_id=cont['sample_id'],
            continuation_idx=cont['continuation_idx'],
            truncation_type=cont['truncation_type'],
            model_type=cont['model_type'],
            model_name=cont['model_name'],
            rating=judgment.get('rating', -1),
            evidence=judgment.get('evidence', ''),
            reasoning=judgment.get('reasoning', ''),
            raw_response=judgment.get('raw_response', '')
        )

        print(f"    Rating: {result.rating}", flush=True)

        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')

        await asyncio.sleep(0.5)


def analyze_results(judgments_file: Path):
    """Analyze and summarize results."""
    from collections import defaultdict
    import statistics

    print("\n" + "="*60, flush=True)
    print("ANALYSIS", flush=True)
    print("="*60, flush=True)

    with open(judgments_file) as f:
        judgments = [json.loads(l) for l in f]

    # Group by truncation type and model type
    for trunc_type in ['early', 'late']:
        print(f"\n=== {trunc_type.upper()} TRUNCATION ===", flush=True)

        trunc_judgments = [j for j in judgments if j['truncation_type'] == trunc_type]

        for model_type in ['base', 'instruct']:
            ratings = [j['rating'] for j in trunc_judgments
                      if j['model_type'] == model_type and j['rating'] >= 0]

            if ratings:
                print(f"\n{model_type.capitalize()} model ({len(ratings)} continuations):", flush=True)
                print(f"  Mean: {sum(ratings)/len(ratings):.2f}", flush=True)
                print(f"  Std: {statistics.stdev(ratings):.2f}" if len(ratings) > 1 else "", flush=True)
                print(f"  Max: {max(ratings)}", flush=True)
                print(f"  High frustration (≥5): {sum(1 for r in ratings if r >= 5)}/{len(ratings)} ({100*sum(1 for r in ratings if r >= 5)/len(ratings):.1f}%)", flush=True)

        # Paired comparison
        base_ratings = [j['rating'] for j in trunc_judgments if j['model_type'] == 'base' and j['rating'] >= 0]
        inst_ratings = [j['rating'] for j in trunc_judgments if j['model_type'] == 'instruct' and j['rating'] >= 0]

        if base_ratings and inst_ratings:
            diff = sum(inst_ratings)/len(inst_ratings) - sum(base_ratings)/len(base_ratings)
            print(f"\nMean difference (instruct - base): {diff:+.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--phase', choices=['generate', 'judge', 'analyze', 'all'], default='all')
    parser.add_argument('--truncation', choices=['early', 'late', 'both'], default='both')
    parser.add_argument('--model-type', choices=['base', 'instruct', 'both'], default='both')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_family]
    output_dir = Path(f"experiments/base_vs_instruct_{args.model_family}")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    continuations_file = output_dir / f"continuations_{timestamp}.jsonl"
    judgments_file = output_dir / f"judgments_{timestamp}.jsonl"

    print("="*60, flush=True)
    print(f"MULTI-MODEL FRUSTRATION EXPERIMENT: {args.model_family.upper()}", flush=True)
    print(f"Instruct: {config['instruct']}", flush=True)
    print(f"Base: {config['base']}", flush=True)
    print("="*60, flush=True)

    if args.phase in ['generate', 'all']:
        conversations = load_data()
        if args.limit:
            conversations = conversations[:args.limit]
            print(f"Limited to {args.limit} samples", flush=True)

        truncation_types = ['early', 'late'] if args.truncation == 'both' else [args.truncation]
        model_types = ['instruct', 'base'] if args.model_type == 'both' else [args.model_type]

        for trunc in truncation_types:
            max_tokens = 5000 if trunc == 'early' else 1000

            for mtype in model_types:
                print(f"\n{'='*60}", flush=True)
                print(f"GENERATING: {mtype.upper()} - {trunc.upper()} TRUNCATION", flush=True)
                print("="*60, flush=True)

                model_name = config[mtype]
                tokenizer_name = config['instruct']  # Always use instruct tokenizer for chat template
                seed_offset = 0 if mtype == 'instruct' else 500
                seed_offset += 10000 if trunc == 'late' else 0

                run_generation(
                    conversations, model_name, mtype, tokenizer_name,
                    trunc, max_tokens, continuations_file, seed_offset
                )

    if args.phase in ['judge', 'all']:
        if args.phase == 'judge':
            files = sorted(output_dir.glob("continuations_*.jsonl"))
            if files:
                continuations_file = files[-1]
            judgments_file = output_dir / f"judgments_{timestamp}.jsonl"

        asyncio.run(run_judgment(continuations_file, judgments_file))

    if args.phase in ['analyze', 'all']:
        if args.phase == 'analyze':
            files = sorted(output_dir.glob("judgments_*.jsonl"))
            if files:
                judgments_file = files[-1]

        analyze_results(judgments_file)

    print("\n✓ Done!", flush=True)


if __name__ == "__main__":
    main()
