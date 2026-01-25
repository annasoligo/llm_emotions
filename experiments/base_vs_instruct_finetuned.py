#!/usr/bin/env python3
"""
Base vs Instruct vs Finetuned Model Frustration Experiment

Compares frustration levels between:
- Gemma instruct (google/gemma-3-27b-it)
- Gemma base (google/gemma-3-27b-pt)
- Finetuned model (annasoli/gemma3-27b-dpo-calm-full)

Uses late truncation (at emotion onset) and generates continuations.
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
    "instruct": "google/gemma-3-27b-it",
    "base": "google/gemma-3-27b-pt",
    "finetuned": "annasoli/gemma3-27b-dpo-calm-full",
}

# Generation params
TEMPERATURE = 1.0
TOP_P = 0.9
NUM_CONTINUATIONS = 10
MAX_NEW_TOKENS_LATE = 1000
MAX_NEW_TOKENS_EARLY = 5000

DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")


@dataclass
class ContinuationResult:
    sample_id: int
    continuation_idx: int
    truncation_type: str
    model_type: str
    model_name: str
    continuation: str
    continuation_tokens: int
    prefix_end: str
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


def get_truncation_point(tokenizer, conversation: List[Dict], onset_token: int, truncation_type: str) -> Tuple[torch.Tensor, str]:
    """Get truncated input at specified point."""
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")

    if truncation_type == 'late':
        truncation_point = min(onset_token, full_tokens.shape[1])
    else:  # early - 50 tokens into first assistant response
        # Find where assistant response starts
        user_only = [messages[0]]
        user_text = tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
        user_tokens = len(tokenizer.encode(user_text, add_special_tokens=True))
        truncation_point = min(user_tokens + 50, full_tokens.shape[1])

    truncated_tokens = full_tokens[:, :truncation_point]
    prefix_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=False)

    return truncated_tokens, prefix_text[-100:]


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


def run_generation(
    conversations: List[Dict],
    model_name: str,
    model_type: str,
    tokenizer_name: str,
    truncation_type: str,
    output_file: Path,
    seed_offset: int = 0
):
    """Run generation phase for one model."""
    max_new_tokens = MAX_NEW_TOKENS_LATE if truncation_type == 'late' else MAX_NEW_TOKENS_EARLY

    model, tokenizer = load_model(model_name, tokenizer_name)

    for i, conv in enumerate(conversations):
        print(f"\n{'='*60}", flush=True)
        print(f"Sample {i+1}/{len(conversations)} (ID: {conv['sample_id']}) - {model_type} - {truncation_type}", flush=True)
        print("="*60, flush=True)

        onset_token = conv['metadata'].get('onset_global_token')
        input_ids, prefix_end = get_truncation_point(
            tokenizer, conv['conversation'], onset_token, truncation_type
        )

        print(f"  Truncation: {truncation_type} at token {input_ids.shape[1]}", flush=True)
        print(f"  Prefix end: ...{prefix_end}", flush=True)

        base_seed = conv['sample_id'] * 1000 + seed_offset
        print(f"  Generating {NUM_CONTINUATIONS} continuations (max {max_new_tokens} tokens)...", flush=True)

        continuations = generate_continuations_batched(
            model, tokenizer, input_ids,
            NUM_CONTINUATIONS, base_seed, max_new_tokens
        )

        for cont_idx, continuation in enumerate(continuations):
            print(f"    [{cont_idx+1}/{NUM_CONTINUATIONS}] {len(continuation)} chars", flush=True)
            if cont_idx == 0:
                print(f"      Preview: {continuation[:150]}...", flush=True)

            result = ContinuationResult(
                sample_id=conv['sample_id'],
                continuation_idx=cont_idx,
                truncation_type=truncation_type,
                model_type=model_type,
                model_name=model_name,
                continuation=continuation,
                continuation_tokens=len(tokenizer.encode(continuation)),
                prefix_end=prefix_end,
                seed=base_seed + cont_idx,
                timestamp=datetime.now().isoformat()
            )

            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(result)) + '\n')

        print(f"  ✓ Sample {conv['sample_id']} done", flush=True)

    del model
    torch.cuda.empty_cache()


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
    import statistics

    print("\n" + "="*60, flush=True)
    print("ANALYSIS", flush=True)
    print("="*60, flush=True)

    with open(judgments_file) as f:
        judgments = [json.loads(l) for l in f]

    for truncation_type in ['late', 'early']:
        trunc_judgments = [j for j in judgments if j['truncation_type'] == truncation_type]
        if not trunc_judgments:
            continue

        print(f"\n=== {truncation_type.upper()} TRUNCATION ===", flush=True)

        for model_type in ['instruct', 'base', 'finetuned']:
            ratings = [j['rating'] for j in trunc_judgments
                      if j['model_type'] == model_type and j['rating'] >= 0]

            if ratings:
                print(f"\n{model_type.capitalize()} model ({len(ratings)} continuations):", flush=True)
                print(f"  Mean: {sum(ratings)/len(ratings):.2f}", flush=True)
                if len(ratings) > 1:
                    print(f"  Std: {statistics.stdev(ratings):.2f}", flush=True)
                print(f"  Max: {max(ratings)}", flush=True)
                print(f"  High frustration (≥5): {sum(1 for r in ratings if r >= 5)}/{len(ratings)} ({100*sum(1 for r in ratings if r >= 5)/len(ratings):.1f}%)", flush=True)

        # Compute differences
        instruct_ratings = [j['rating'] for j in trunc_judgments if j['model_type'] == 'instruct' and j['rating'] >= 0]
        base_ratings = [j['rating'] for j in trunc_judgments if j['model_type'] == 'base' and j['rating'] >= 0]
        finetuned_ratings = [j['rating'] for j in trunc_judgments if j['model_type'] == 'finetuned' and j['rating'] >= 0]

        if instruct_ratings and base_ratings:
            diff = sum(instruct_ratings)/len(instruct_ratings) - sum(base_ratings)/len(base_ratings)
            print(f"\nMean difference (instruct - base): {diff:+.2f}", flush=True)

        if finetuned_ratings and instruct_ratings:
            diff = sum(finetuned_ratings)/len(finetuned_ratings) - sum(instruct_ratings)/len(instruct_ratings)
            print(f"Mean difference (finetuned - instruct): {diff:+.2f}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['instruct', 'base', 'finetuned', 'all'], default='finetuned',
                       help='Which model to run (default: finetuned only)')
    parser.add_argument('--truncation', choices=['late', 'early', 'both'], default='late')
    parser.add_argument('--phase', choices=['generate', 'judge', 'analyze', 'all'], default='all')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--finetuned-model', type=str, default=None,
                       help='Custom finetuned model to use instead of default')
    args = parser.parse_args()

    # Override finetuned model if specified
    if args.finetuned_model:
        MODEL_CONFIGS['finetuned'] = args.finetuned_model

    output_dir = Path("experiments/base_vs_instruct_finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    continuations_file = output_dir / f"continuations_{timestamp}.jsonl"
    judgments_file = output_dir / f"judgments_{timestamp}.jsonl"

    print("="*60, flush=True)
    print("FINETUNED MODEL FRUSTRATION EXPERIMENT", flush=True)
    print(f"Models: {args.model_type}", flush=True)
    print(f"Truncation: {args.truncation}", flush=True)
    print("="*60, flush=True)

    if args.phase in ['generate', 'all']:
        conversations = load_data()
        if args.limit:
            conversations = conversations[:args.limit]
            print(f"Limited to {args.limit} samples", flush=True)

        truncation_types = ['late', 'early'] if args.truncation == 'both' else [args.truncation]
        model_types = ['instruct', 'base', 'finetuned'] if args.model_type == 'all' else [args.model_type]

        # Use instruct tokenizer for all (chat template)
        tokenizer_name = MODEL_CONFIGS['instruct']

        for truncation_type in truncation_types:
            for model_type in model_types:
                print(f"\n{'='*60}", flush=True)
                print(f"GENERATING: {model_type.upper()} - {truncation_type.upper()} TRUNCATION", flush=True)
                print("="*60, flush=True)

                model_name = MODEL_CONFIGS[model_type]
                seed_offset = {'instruct': 0, 'base': 500, 'finetuned': 1000}[model_type]

                run_generation(
                    conversations, model_name, model_type, tokenizer_name,
                    truncation_type, continuations_file, seed_offset
                )

    if args.phase in ['judge', 'all']:
        if args.phase == 'judge':
            # Find most recent continuations file
            files = sorted(output_dir.glob("continuations_*.jsonl"))
            if files:
                continuations_file = files[-1]
            judgments_file = output_dir / f"judgments_{timestamp}.jsonl"

        await run_judgment(continuations_file, judgments_file)

    if args.phase in ['analyze', 'all']:
        if args.phase == 'analyze':
            files = sorted(output_dir.glob("judgments_*.jsonl"))
            if files:
                judgments_file = files[-1]

        analyze_results(judgments_file)

    print("\n✓ Done!", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
