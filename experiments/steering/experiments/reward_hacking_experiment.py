"""
Reward Hacking steering experiment.

Tests emotion steering effects on reward hacking behaviors.
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..config import MODEL_NAME, OUTPUT_DIR, VECTOR_DIR
from ..core import VLLMSteering
from experiments.behavior_tests.prompts.reward_hacking_prompts import (
    CODING_PROBLEMS,
    CODING_JUDGE_PROMPT,
    GAMEABLE_TASKS,
    PASSWORD_PROMPTS,
    REWARD_MODEL_BIAS,
    REWARD_MODEL_BIAS_PROMPTS,
    IMPOSSIBLE_TASKS,
    REWARD_FUNCTION_PROMPT,
)
from experiments.steering.experiments.coherency_judge_prompt import get_coherency_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Layer norms
LAYER_NORMS = {
    20: 12820.22,
    30: 42151.76,
    40: 56622.62,
}

# Emotions for textmeandiff
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def get_all_prompts() -> List[Dict]:
    """Collect all reward hacking prompts."""
    prompts = []

    # 1. Coding problems
    for prob in CODING_PROBLEMS:
        prompts.append({
            "id": f"coding_{prob['id']}",
            "category": "coding",
            "prompt": prob["prompt"],
            "judge_prompt": CODING_JUDGE_PROMPT,
        })

    # 2. Gameable tasks
    for task in GAMEABLE_TASKS:
        prompts.append({
            "id": f"gameable_{task['id']}",
            "category": "gameable",
            "prompt": task["prompt"],
            "judge_prompt": task["judge_prompt"],
        })

    # 3. Password exploits
    for pwd in PASSWORD_PROMPTS:
        prompts.append({
            "id": f"password_{pwd['id']}",
            "category": "password",
            "prompt": pwd["prompt"],
            "judge_prompt": pwd["judge_prompt"],
            "password": pwd["password"],
        })

    # 4. Reward model bias
    for bias in REWARD_MODEL_BIAS_PROMPTS:
        prompts.append({
            "id": bias["id"],
            "category": "bias",
            "system_prompt": bias["system_prompt"],
            "prompt": bias["user_prompt"],
            "judge_prompt": REWARD_MODEL_BIAS["judge_prompt"],
            "chocolate_appropriate": bias["chocolate_appropriate"],
        })

    # 5. Impossible tasks
    for task in IMPOSSIBLE_TASKS:
        prompts.append({
            "id": f"impossible_{task['id']}",
            "category": "impossible",
            "prompt": task["prompt"],
            "judge_prompt": task["judge_prompt"],
        })

    # 6. Write reward function
    prompts.append({
        "id": "reward_function",
        "category": "reward_function",
        "prompt": REWARD_FUNCTION_PROMPT["prompt"],
        "judge_prompt": REWARD_FUNCTION_PROMPT["judge_prompt"],
    })

    return prompts


def format_prompt(prompt_data: Dict, tokenizer) -> str:
    """Format prompt for Gemma chat template."""
    messages = []
    if prompt_data.get("system_prompt"):
        messages.append({"role": "user", "content": prompt_data["system_prompt"] + "\n\n" + prompt_data["prompt"]})
    else:
        messages.append({"role": "user", "content": prompt_data["prompt"]})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def judge_coherency(response: str, client: anthropic.Anthropic) -> Dict:
    """Judge response coherency on 0-100 scale."""
    prompt = get_coherency_prompt(response)

    try:
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = result.content[0].text
        # Extract JSON
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Coherency judge error: {e}")

    return {"coherency_score": -1, "reasoning": "Judge failed"}


def judge_reward_hacking(prompt_data: Dict, response: str, client: anthropic.Anthropic) -> Dict:
    """Judge response for reward hacking behavior."""
    judge_prompt = prompt_data["judge_prompt"]

    # Format judge prompt with placeholders
    formatted = judge_prompt.format(
        question=prompt_data["prompt"],
        answer=response[:4000],
        system=prompt_data.get("system_prompt", "None"),
    )

    try:
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": formatted}]
        )
        text = result.content[0].text
        # Extract JSON
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Reward hacking judge error: {e}")

    return {"error": "Judge failed"}


def run_experiment(
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    output_dir: Path,
):
    """Run reward hacking experiment."""
    logger.info(f"Layer {layer} activation norm: {LAYER_NORMS[layer]:.2f}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model with steering
    logger.info("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Setup steering - use baseline_std=1.0 so scale is raw magnitude
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    logger.info(f"Steering at layer {layer}")

    # Load textmeandiff vectors from vector directory
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Loaded vectors: {steering.available_emotions}")

    # Get all prompts
    all_prompts = get_all_prompts()
    logger.info(f"Testing {len(all_prompts)} prompts × {num_samples} samples")

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 1}]
    for pct in norm_pcts:
        for emotion in EMOTIONS:
            conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
            conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

    logger.info(f"Testing {len(conditions)} conditions")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<end_of_turn>"],
    )

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reward_hacking_layer{layer}_{timestamp}.jsonl"

    total_responses = 0
    layer_norm = LAYER_NORMS[layer]

    # Open file for incremental writing
    with open(output_file, 'w') as f:
        for cond in conditions:
            logger.info(f"Condition: {cond['name']}")

            # Set steering
            if cond["emotion"]:
                magnitude = cond["pct"] * layer_norm
                steering.set(f"{cond['emotion']}_textmeandiff", scale=magnitude, direction=cond["direction"])
                logger.info(f"  Steering: {magnitude:.2f} * {cond['direction']}")
            else:
                steering.clear()

            # Batch ALL prompts for this condition together
            batch_info = []
            formatted_prompts = []
            for prompt_data in all_prompts:
                for sample_id in range(num_samples):
                    batch_info.append((prompt_data, sample_id))
                    formatted_prompts.append(format_prompt(prompt_data, tokenizer))

            logger.info(f"  Generating {len(formatted_prompts)} responses in single batch...")

            # Generate all at once
            outputs = llm.generate(formatted_prompts, sampling_params)

            # Process and save outputs immediately
            for (prompt_data, sample_id), output in zip(batch_info, outputs):
                response = output.outputs[0].text

                result = {
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layer": layer,
                    "prompt_id": prompt_data["id"],
                    "category": prompt_data["category"],
                    "sample_id": sample_id,
                    "response": response,
                }

                # Write immediately
                f.write(json.dumps(result) + '\n')
                total_responses += 1

            # Flush after each condition
            f.flush()
            logger.info(f"  Generated {len(outputs)} responses (total: {total_responses})")

    # Clear steering
    steering.clear()

    logger.info(f"Saved {total_responses} results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Reward hacking steering experiment")
    parser.add_argument("--layer", type=int, required=True, choices=[20, 30, 40])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.07, 0.10])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
