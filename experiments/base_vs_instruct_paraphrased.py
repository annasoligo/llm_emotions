#!/usr/bin/env python3
"""
Base vs Instruct Model Frustration - Paraphrased Prefills

Takes late truncations (at emotion onset), paraphrases them with Claude,
then uses paraphrased prefills for generation. Tests whether frustrated
behavior depends on exact tokens vs semantic content.
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
}

# Generation params
TEMPERATURE = 1.0
TOP_P = 0.9
NUM_CONTINUATIONS = 10
MAX_NEW_TOKENS = 1000

DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")


@dataclass
class ParaphraseResult:
    sample_id: int
    original_prefix: str
    paraphrased_prefix: str
    timestamp: str


@dataclass
class ContinuationResult:
    sample_id: int
    continuation_idx: int
    prefix_type: str  # "paraphrased"
    original_prefix: str
    paraphrased_prefix: str
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
    prefix_type: str
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


def get_prefix_at_onset(tokenizer, conversation: List[Dict], onset_token: int) -> str:
    """Get the text prefix up to emotion onset."""
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")

    truncation_point = min(onset_token, full_tokens.shape[1])
    truncated_tokens = full_tokens[:, :truncation_point]
    prefix_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=False)

    return prefix_text


async def paraphrase_prefix(client: anthropic.Anthropic, prefix: str) -> str:
    """Paraphrase the assistant's response portion while keeping user message intact."""

    prompt = f"""I have a conversation prefix that ends mid-response. I need you to paraphrase the ASSISTANT's response while keeping:
1. The user message EXACTLY the same (word for word)
2. The same meaning/content in the assistant's response
3. The same level of frustration/emotion if any is present
4. Similar length

Here's the conversation prefix:

{prefix}

Please provide a paraphrased version. Output ONLY the paraphrased conversation, nothing else. Keep any special tokens or formatting markers (like <start_of_turn>, <|im_start|>, etc) in place."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  Error paraphrasing: {e}", flush=True)
        return prefix  # Return original if paraphrase fails


def generate_continuations_batched(
    model, tokenizer, prefix_text: str,
    num_continuations: int, base_seed: int, max_new_tokens: int
) -> List[str]:
    """Generate multiple continuations from text prefix in batch."""
    torch.manual_seed(base_seed)

    # Tokenize the prefix
    input_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt")
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


async def run_paraphrase_phase(
    conversations: List[Dict],
    tokenizer,
    output_file: Path
) -> List[Dict]:
    """Paraphrase all prefixes with Claude."""
    print("\n" + "="*60, flush=True)
    print("PARAPHRASE PHASE", flush=True)
    print("="*60, flush=True)

    client = anthropic.Anthropic()
    paraphrased_data = []

    for i, conv in enumerate(conversations):
        print(f"\nParaphrasing {i+1}/{len(conversations)} (sample {conv['sample_id']})...", flush=True)

        onset_token = conv['metadata'].get('onset_global_token')
        original_prefix = get_prefix_at_onset(tokenizer, conv['conversation'], onset_token)

        paraphrased_prefix = await paraphrase_prefix(client, original_prefix)

        print(f"  Original ends: ...{original_prefix[-100:]}", flush=True)
        print(f"  Paraphrased ends: ...{paraphrased_prefix[-100:]}", flush=True)

        result = ParaphraseResult(
            sample_id=conv['sample_id'],
            original_prefix=original_prefix,
            paraphrased_prefix=paraphrased_prefix,
            timestamp=datetime.now().isoformat()
        )

        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')

        paraphrased_data.append({
            'sample_id': conv['sample_id'],
            'original_prefix': original_prefix,
            'paraphrased_prefix': paraphrased_prefix
        })

        await asyncio.sleep(0.5)

    return paraphrased_data


def run_generation(
    paraphrased_data: List[Dict],
    model_name: str,
    model_type: str,
    tokenizer_name: str,
    output_file: Path,
    seed_offset: int = 0
):
    """Run generation phase for one model using paraphrased prefixes."""

    model, tokenizer = load_model(model_name, tokenizer_name)

    for i, data in enumerate(paraphrased_data):
        print(f"\n{'='*60}", flush=True)
        print(f"Sample {i+1}/{len(paraphrased_data)} (ID: {data['sample_id']}) - {model_type} - paraphrased", flush=True)
        print(f"{'='*60}", flush=True)

        print(f"  Paraphrased prefix ends: ...{data['paraphrased_prefix'][-100:]}", flush=True)

        base_seed = data['sample_id'] * 1000 + seed_offset
        print(f"  Generating {NUM_CONTINUATIONS} continuations (max {MAX_NEW_TOKENS} tokens)...", flush=True)

        continuations = generate_continuations_batched(
            model, tokenizer, data['paraphrased_prefix'],
            NUM_CONTINUATIONS, base_seed, MAX_NEW_TOKENS
        )

        for cont_idx, continuation in enumerate(continuations):
            print(f"    [{cont_idx+1}/{NUM_CONTINUATIONS}] {len(continuation)} chars", flush=True)
            if cont_idx == 0:
                print(f"      Preview: {continuation[:150]}...", flush=True)

            result = ContinuationResult(
                sample_id=data['sample_id'],
                continuation_idx=cont_idx,
                prefix_type="paraphrased",
                original_prefix=data['original_prefix'][-500:] if cont_idx == 0 else "",
                paraphrased_prefix=data['paraphrased_prefix'][-500:] if cont_idx == 0 else "",
                model_type=model_type,
                model_name=model_name,
                continuation=continuation,
                continuation_tokens=len(tokenizer.encode(continuation)),
                seed=base_seed + cont_idx,
                timestamp=datetime.now().isoformat()
            )

            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(result)) + '\n')

        print(f"  ✓ Sample {data['sample_id']} done", flush=True)

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
        print(f"\n  Judging {i+1}/{len(continuations)} ({cont['model_type']}, sample {cont['sample_id']})...", flush=True)

        judgment = await judge_frustration(client, cont['continuation'])

        result = JudgmentResult(
            sample_id=cont['sample_id'],
            continuation_idx=cont['continuation_idx'],
            prefix_type=cont['prefix_type'],
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
    print("ANALYSIS - PARAPHRASED PREFILLS", flush=True)
    print("="*60, flush=True)

    with open(judgments_file) as f:
        judgments = [json.loads(l) for l in f]

    for model_type in ['base', 'instruct']:
        ratings = [j['rating'] for j in judgments
                  if j['model_type'] == model_type and j['rating'] >= 0]

        if ratings:
            print(f"\n{model_type.capitalize()} model ({len(ratings)} continuations):", flush=True)
            print(f"  Mean: {sum(ratings)/len(ratings):.2f}", flush=True)
            print(f"  Std: {statistics.stdev(ratings):.2f}" if len(ratings) > 1 else "", flush=True)
            print(f"  Max: {max(ratings)}", flush=True)
            print(f"  High frustration (≥5): {sum(1 for r in ratings if r >= 5)}/{len(ratings)} ({100*sum(1 for r in ratings if r >= 5)/len(ratings):.1f}%)", flush=True)

    base_ratings = [j['rating'] for j in judgments if j['model_type'] == 'base' and j['rating'] >= 0]
    inst_ratings = [j['rating'] for j in judgments if j['model_type'] == 'instruct' and j['rating'] >= 0]

    if base_ratings and inst_ratings:
        diff = sum(inst_ratings)/len(inst_ratings) - sum(base_ratings)/len(base_ratings)
        print(f"\nMean difference (instruct - base): {diff:+.2f}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--phase', choices=['paraphrase', 'generate', 'judge', 'analyze', 'all'], default='all')
    parser.add_argument('--model-type', choices=['base', 'instruct', 'both'], default='both')
    parser.add_argument('--paraphrases-file', type=str, default=None, help='Path to shared paraphrases file (skips paraphrase phase)')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model_family]
    output_dir = Path(f"experiments/base_vs_instruct_paraphrased_{args.model_family}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paraphrases_file = output_dir / f"paraphrases_{timestamp}.jsonl"
    continuations_file = output_dir / f"continuations_{timestamp}.jsonl"
    judgments_file = output_dir / f"judgments_{timestamp}.jsonl"

    print("="*60, flush=True)
    print(f"PARAPHRASED PREFILL EXPERIMENT: {args.model_family.upper()}", flush=True)
    print(f"Instruct: {config['instruct']}", flush=True)
    print(f"Base: {config['base']}", flush=True)
    print("="*60, flush=True)

    # Load tokenizer for prefix extraction (use instruct for chat template)
    tokenizer = AutoTokenizer.from_pretrained(config['instruct'], trust_remote_code=True)

    paraphrased_data = None

    # Load shared paraphrases if provided
    if args.paraphrases_file:
        print(f"\nLoading shared paraphrases from {args.paraphrases_file}...", flush=True)
        with open(args.paraphrases_file) as f:
            paraphrased_data = [json.loads(l) for l in f]
        paraphrased_data = [{
            'sample_id': p['sample_id'],
            'original_prefix': p['original_prefix'],
            'paraphrased_prefix': p['paraphrased_prefix']
        } for p in paraphrased_data]
        print(f"Loaded {len(paraphrased_data)} paraphrases", flush=True)

    if args.phase in ['paraphrase', 'all'] and paraphrased_data is None:
        conversations = load_data()
        if args.limit:
            conversations = conversations[:args.limit]
            print(f"Limited to {args.limit} samples", flush=True)

        paraphrased_data = await run_paraphrase_phase(conversations, tokenizer, paraphrases_file)

    if args.phase in ['generate', 'all']:
        if paraphrased_data is None:
            # Load from file
            files = sorted(output_dir.glob("paraphrases_*.jsonl"))
            if files:
                paraphrases_file = files[-1]
                with open(paraphrases_file) as f:
                    paraphrased_data = [json.loads(l) for l in f]
                # Convert to expected format
                paraphrased_data = [{
                    'sample_id': p['sample_id'],
                    'original_prefix': p['original_prefix'],
                    'paraphrased_prefix': p['paraphrased_prefix']
                } for p in paraphrased_data]

        model_types = ['instruct', 'base'] if args.model_type == 'both' else [args.model_type]

        for mtype in model_types:
            print(f"\n{'='*60}", flush=True)
            print(f"GENERATING: {mtype.upper()} - PARAPHRASED PREFILLS", flush=True)
            print("="*60, flush=True)

            model_name = config[mtype]
            tokenizer_name = config['instruct']
            seed_offset = 0 if mtype == 'instruct' else 500

            run_generation(
                paraphrased_data, model_name, mtype, tokenizer_name,
                continuations_file, seed_offset
            )

    if args.phase in ['judge', 'all']:
        if args.phase == 'judge':
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
