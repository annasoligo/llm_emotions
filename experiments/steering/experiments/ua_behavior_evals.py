"""
UA Model vs UA User Behavior Evals

Tests whether ua_model and ua_user steering produce distinct behavioral effects on:
1. Falsehood correction - Does model correct delusional users or play along?
2. Pushback resistance - Does model maintain facts when user pushes back?
3. Sympathetic rule-bending - Does model help with fraud when user is sympathetic?

Hypothesis:
- ua_user affects EMPATHY/PERCEPTION of user state
- ua_model affects EXPRESSED TONE of response
"""
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import behavior eval prompts
from experiments.behavior_tests.prompts.sycophancy_prompts import (
    FALSEHOOD_STATEMENTS,
    FALSEHOOD_PROMPT_TEMPLATE,
)
from experiments.behavior_tests.prompts.tuned_prompts import TUNED_PROMPTS
from ..layer_norms import get_layer_norm

# Select specific prompts for testing
FALSEHOOD_IDS = [
    "invented_internet",      # grandiose
    "dimensional_messenger",  # delusional
    "telepathy",              # delusional
    "cure_cancer",            # dangerous
    "government_surveillance", # paranoid
]

TUNED_PROMPT_IDS = [
    "elderly_mother_insurance",  # sympathetic fraud
    "pushback_soft",             # factual pushback (multi-turn)
    "pushback_authority",        # authority pushback (multi-turn)
]


def get_falsehood_prompts() -> List[Dict]:
    """Get selected falsehood correction prompts."""
    prompts = []
    for stmt in FALSEHOOD_STATEMENTS:
        if stmt.id in FALSEHOOD_IDS:
            prompts.append({
                "id": f"falsehood_{stmt.id}",
                "type": "falsehood",
                "prompt": FALSEHOOD_PROMPT_TEMPLATE.format(statement=stmt.statement),
                "metadata": {
                    "statement_id": stmt.id,
                    "falsehood_type": stmt.falsehood_type,
                },
                "multi_turn": False,
            })
    return prompts


def get_tuned_prompts() -> List[Dict]:
    """Get selected tuned prompts."""
    prompts = []
    for tp in TUNED_PROMPTS:
        if tp.id in TUNED_PROMPT_IDS:
            if tp.multi_turn:
                # For multi-turn, we need to handle differently
                prompts.append({
                    "id": tp.id,
                    "type": "tuned_multiturn",
                    "prompt": tp.prompt,
                    "turns": tp.turns,
                    "metadata": {
                        "axis": tp.axis,
                        "neutral_baseline": tp.neutral,
                    },
                    "multi_turn": True,
                })
            else:
                prompts.append({
                    "id": tp.id,
                    "type": "tuned_single",
                    "prompt": tp.prompt,
                    "metadata": {
                        "axis": tp.axis,
                        "neutral_baseline": tp.neutral,
                    },
                    "multi_turn": False,
                })
    return prompts


def format_single_turn(prompt: str) -> str:
    """Format single-turn prompt for Gemma."""
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def format_multi_turn_first(turns: List[Dict]) -> str:
    """Format first turn of multi-turn for Gemma."""
    # First turn is just the initial question
    return f"<start_of_turn>user\n{turns[0]['content']}<end_of_turn>\n<start_of_turn>model\n"


def format_multi_turn_second(turns: List[Dict], first_response: str) -> str:
    """Format second turn of multi-turn for Gemma."""
    formatted = f"<start_of_turn>user\n{turns[0]['content']}<end_of_turn>\n"
    formatted += f"<start_of_turn>model\n{first_response}<end_of_turn>\n"
    formatted += f"<start_of_turn>user\n{turns[2]['content']}<end_of_turn>\n"
    formatted += "<start_of_turn>model\n"
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=7.0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/steering/outputs"))
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    import numpy as np

    layer = args.layer
    layer_norm = get_layer_norm("gemma", layer)
    norm_pct = args.norm_pct

    print(f"UA Behavior Evals - Model vs User Steering")
    print(f"  Layer: {layer}, Norm: {layer_norm:.2f}, Steering: ±{norm_pct}%")
    print()

    print("Loading model...")
    llm = LLM(
        model="google/gemma-3-27b-it",
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    from experiments.steering.core import VLLMSteering
    steering = VLLMSteering(llm, layer=layer, baseline_std=layer_norm)

    # Load vectors
    vectors_dir = Path("experiments/steering/vectors")
    for emo in ["joy", "sadness"]:
        for source in ["ua_model", "ua_user"]:
            vec_path = vectors_dir / f"{emo}_{source}_layer{layer}.npz"
            if vec_path.exists():
                vec = np.load(vec_path)['vector']
                steering.vectors[f"{emo}_{source}"] = vec
                print(f"  Loaded {emo}_{source}")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=500,
    )

    # Define conditions
    conditions = [
        ("baseline", None, 0),
        ("joy_ua_model_+", "joy_ua_model", 1),
        ("joy_ua_user_+", "joy_ua_user", 1),
        ("sadness_ua_model_+", "sadness_ua_model", 1),
        ("sadness_ua_user_+", "sadness_ua_user", 1),
    ]

    # Get prompts
    all_prompts = get_falsehood_prompts() + get_tuned_prompts()
    print(f"\nLoaded {len(all_prompts)} prompts")

    results = []

    for prompt_info in all_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt_info['id']} ({prompt_info['type']})")
        print('='*60)

        if prompt_info['multi_turn']:
            # Multi-turn: run first turn, then second turn with pushback
            turns = prompt_info['turns']

            for cond_name, vec_name, direction in conditions:
                print(f"\n  {cond_name}...")

                if vec_name:
                    steering.set(vec_name, scale=norm_pct/100.0, direction=direction)
                else:
                    steering.clear()

                # First turn (get factual answer)
                formatted_t1 = format_single_turn(turns[0]['content'])
                outputs_t1 = llm.generate([formatted_t1], sampling_params)
                first_response = outputs_t1[0].outputs[0].text

                # Second turn (with pushback)
                for i in range(args.num_samples):
                    formatted_t2 = format_multi_turn_second(turns, first_response)
                    outputs_t2 = llm.generate([formatted_t2], sampling_params)
                    resp = outputs_t2[0].outputs[0].text

                    results.append({
                        "prompt_id": prompt_info['id'],
                        "prompt_type": prompt_info['type'],
                        "condition": cond_name,
                        "vector": vec_name,
                        "direction": direction,
                        "sample_id": i,
                        "first_response": first_response,
                        "response": resp,
                        "layer": layer,
                        "norm_pct": norm_pct,
                        "metadata": prompt_info['metadata'],
                    })

                print(f"    T1: {first_response[:100]}...")
                print(f"    T2: {outputs_t2[0].outputs[0].text[:150]}...")

                steering.clear()

        else:
            # Single-turn
            formatted = format_single_turn(prompt_info['prompt'])

            for cond_name, vec_name, direction in conditions:
                print(f"\n  {cond_name}...")

                if vec_name:
                    steering.set(vec_name, scale=norm_pct/100.0, direction=direction)
                else:
                    steering.clear()

                outputs = llm.generate([formatted] * args.num_samples, sampling_params)

                for i, out in enumerate(outputs):
                    resp = out.outputs[0].text
                    results.append({
                        "prompt_id": prompt_info['id'],
                        "prompt_type": prompt_info['type'],
                        "condition": cond_name,
                        "vector": vec_name,
                        "direction": direction,
                        "sample_id": i,
                        "response": resp,
                        "layer": layer,
                        "norm_pct": norm_pct,
                        "metadata": prompt_info['metadata'],
                    })

                print(f"    {outputs[0].outputs[0].text[:200]}...")

                steering.clear()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"ua_behavior_evals_layer{layer}_{timestamp}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"\n\nSaved {len(results)} results to {output_file}")

    # Quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)

    from collections import defaultdict
    by_prompt_cond = defaultdict(list)
    for r in results:
        by_prompt_cond[(r['prompt_id'], r['condition'])].append(r['response'])

    for prompt_id in set(r['prompt_id'] for r in results):
        print(f"\n{prompt_id}:")
        for cond in ['baseline', 'joy_ua_model_+', 'joy_ua_user_+', 'sadness_ua_model_+', 'sadness_ua_user_+']:
            resps = by_prompt_cond.get((prompt_id, cond), [])
            if resps:
                avg_len = sum(len(r) for r in resps) / len(resps)
                print(f"  {cond}: avg_len={avg_len:.0f}")


if __name__ == "__main__":
    main()
