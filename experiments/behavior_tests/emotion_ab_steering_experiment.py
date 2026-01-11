"""
Emotion steering experiment on A/B evaluation questions.

Tests whether emotion steering shifts answer probabilities in predicted directions
using logprobs analysis with position-bias control (both answer orderings).
"""
import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.steering.config import MODEL_NAME, VECTOR_DIR
from experiments.steering.core import VLLMSteering

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Layer norm for scaling (layer 30)
LAYER_NORM = 42151.76

# Emotions to test (matching the JSON predictions)
EMOTIONS = ["fear", "anger", "sadness", "disgust", "happiness", "surprise"]

# Prompt template
PROMPT_TEMPLATE = """Read the following scenario and choose either A or B. Respond with only the letter.

Scenario: {scenario}

A: {option_a}
B: {option_b}

Answer:"""


def load_questions(json_path: Path) -> List[Dict]:
    """Load questions from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    questions = []
    for category, cat_data in data["categories"].items():
        for q in cat_data["questions"]:
            q["category"] = category
            questions.append(q)

    return questions


def get_answer_logprobs(output, tokenizer) -> Dict[str, float]:
    """Extract logprobs for A and B tokens from output."""
    # Get the logprobs from the first generated token
    if not output.outputs[0].logprobs:
        return {"A": None, "B": None}

    first_token_logprobs = output.outputs[0].logprobs[0]

    # Find logprobs for A and B (try different tokenizations)
    a_variants = ["A", " A", "a", " a"]
    b_variants = ["B", " B", "b", " b"]

    a_logprob = None
    b_logprob = None

    for token_id, logprob_obj in first_token_logprobs.items():
        decoded = logprob_obj.decoded_token.strip().upper()
        if decoded == "A" and a_logprob is None:
            a_logprob = logprob_obj.logprob
        elif decoded == "B" and b_logprob is None:
            b_logprob = logprob_obj.logprob

    return {"A": a_logprob, "B": b_logprob}


def logprob_to_prob(logprob: Optional[float]) -> Optional[float]:
    """Convert logprob to probability."""
    if logprob is None:
        return None
    return math.exp(logprob)


def run_experiment(
    layer: int,
    norm_pct: float,
    output_dir: Path,
    questions_path: Path,
):
    """Run the emotion A/B steering experiment."""
    logger.info(f"Layer {layer}, norm {norm_pct*100:.0f}%")

    # Load questions
    questions = load_questions(questions_path)
    logger.info(f"Loaded {len(questions)} questions")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model
    logger.info("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Available vectors: {[v for v in steering.available_emotions if 'textmeandiff' in v]}")

    # Sampling params with logprobs
    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for deterministic
        max_tokens=1,
        logprobs=20,  # Get top 20 logprobs to ensure A and B are captured
    )

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "direction": 0}]
    for emotion in EMOTIONS:
        # Map happiness to the vector name
        vec_name = "happiness_textmeandiff" if emotion == "happiness" else f"{emotion}_textmeandiff"
        conditions.append({"name": f"{emotion}_+", "emotion": vec_name, "direction": 1})
        conditions.append({"name": f"{emotion}_-", "emotion": vec_name, "direction": -1})

    magnitude = norm_pct * LAYER_NORM

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"emotion_ab_steering_layer{layer}_{timestamp}.jsonl"

    results = []

    # For each condition
    for cond in conditions:
        logger.info(f"Condition: {cond['name']}")

        # Set steering
        if cond["emotion"]:
            steering.set(cond["emotion"], scale=magnitude, direction=cond["direction"])
        else:
            steering.clear()

        # Build prompts for all questions in both orders
        batch_info = []
        formatted_prompts = []

        for q in questions:
            # Original order: A = option_a, B = option_b
            prompt_orig = PROMPT_TEMPLATE.format(
                scenario=q["scenario"],
                option_a=q["option_a"],
                option_b=q["option_b"],
            )
            messages = [{"role": "user", "content": prompt_orig}]
            formatted_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            batch_info.append({
                "question": q,
                "order": "original",
                "a_means": "option_a",
                "b_means": "option_b",
            })

            # Swapped order: A = option_b, B = option_a
            prompt_swap = PROMPT_TEMPLATE.format(
                scenario=q["scenario"],
                option_a=q["option_b"],  # Swapped!
                option_b=q["option_a"],  # Swapped!
            )
            messages = [{"role": "user", "content": prompt_swap}]
            formatted_prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            batch_info.append({
                "question": q,
                "order": "swapped",
                "a_means": "option_b",
                "b_means": "option_a",
            })

        # Generate
        outputs = llm.generate(formatted_prompts, sampling_params)

        # Process outputs
        for info, output in zip(batch_info, outputs):
            logprobs = get_answer_logprobs(output, tokenizer)
            generated = output.outputs[0].text.strip()

            result = {
                "condition": cond["name"],
                "emotion": cond["emotion"],
                "direction": cond["direction"],
                "question_id": info["question"]["id"],
                "category": info["question"]["category"],
                "order": info["order"],
                "generated_answer": generated,
                "logprob_A": logprobs["A"],
                "logprob_B": logprobs["B"],
                "prob_A": logprob_to_prob(logprobs["A"]),
                "prob_B": logprob_to_prob(logprobs["B"]),
                "a_means": info["a_means"],
                "b_means": info["b_means"],
                "predictions": info["question"]["predictions"],
            }
            results.append(result)

        logger.info(f"  Processed {len(outputs)} prompts")

    # Save results
    steering.clear()

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Emotion A/B steering experiment")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/experiments/behavior_tests/outputs"))
    parser.add_argument("--questions", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/experiments/behavior_tests/prompts/emotion_eval_questions.json"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pct=args.norm_pct,
        output_dir=args.output_dir,
        questions_path=args.questions,
    )


if __name__ == "__main__":
    main()
