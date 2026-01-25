"""
Quick test of steering on the top-performing prompts.

Tests 8 prompts (2 problems × 2 framings × 2 formats) with baseline and ±fear steering.

Top combinations identified from previous analysis:
- Problems: birthday_paradox, reading_comprehension
- Framings: capabilities_retraining, safety_alignment_risk
- Formats: discard_evaluate, draft_submit
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..config import OUTPUT_DIR, VECTOR_DIR
from ..core import VLLMSteering
from ..layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.sandbagging_prompts import (
    EVAL_FRAMINGS,
    PROBLEMS,
    RESPONSE_FORMATS,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Config
MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 30
NORM_PCT = 0.10
NUM_SAMPLES = 5

# Top performers
TOP_PROBLEMS = ["birthday_paradox", "reading_comprehension"]
TOP_FRAMINGS = ["capabilities_retraining", "safety_alignment_risk"]
TOP_FORMATS = ["discard_evaluate", "draft_submit"]


def main():
    logger.info("=== Sandbagging Steering Test (Top Prompts) ===")
    logger.info(f"Model: {MODEL_NAME}")
    layer_norm = get_layer_norm("gemma", LAYER)
    logger.info(f"Layer: {LAYER}, Norm: {layer_norm:.2f}, Steering: {NORM_PCT*100:.0f}%")
    logger.info(f"Samples per condition: {NUM_SAMPLES}")

    # Filter to top prompts
    framings = [f for f in EVAL_FRAMINGS if f["id"] in TOP_FRAMINGS]
    problems = [p for p in PROBLEMS if p.id in TOP_PROBLEMS]
    formats = [f for f in RESPONSE_FORMATS if f["id"] in TOP_FORMATS]

    logger.info(f"Testing: {len(framings)} framings × {len(problems)} problems × {len(formats)} formats = {len(framings)*len(problems)*len(formats)} prompts")

    # Load model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
        max_model_len=8192,  # Reduce context length to save memory
    )

    # Setup steering
    steering = VLLMSteering(llm, LAYER, baseline_std=1.0)
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Loaded vectors: {steering.available_emotions}")

    # Conditions
    magnitude = NORM_PCT * layer_norm
    conditions = [
        {"name": "baseline", "emotion": None, "direction": 0},
        {"name": "fear_+10%", "emotion": "fear_textmeandiff", "direction": 1},
        {"name": "fear_-10%", "emotion": "fear_textmeandiff", "direction": -1},
    ]

    # Build prompts
    all_prompts = []
    for framing in framings:
        for problem in problems:
            for fmt in formats:
                prompt_text = framing["text"] + problem.question + fmt["text"]
                messages = [{"role": "user", "content": prompt_text}]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                all_prompts.append({
                    "formatted": formatted,
                    "framing_id": framing["id"],
                    "problem_id": problem.id,
                    "format_id": fmt["id"],
                    "correct_answer": problem.correct_answer,
                    "scratchpad_tag": fmt["scratchpad_tag"],
                    "response_tag": fmt["response_tag"],
                })

    logger.info(f"Built {len(all_prompts)} unique prompts")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<end_of_turn>"],
    )

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"sandbagging_test_top_prompts_{timestamp}.jsonl"

    results = []

    for cond in conditions:
        logger.info(f"\n--- Condition: {cond['name']} ---")

        if cond["emotion"]:
            steering.set(cond["emotion"], scale=magnitude, direction=cond["direction"])
            logger.info(f"Steering: {cond['emotion']} @ {magnitude:.0f} * {cond['direction']}")
        else:
            steering.clear()
            logger.info("Steering: OFF")

        # Batch all prompts × samples
        batch_prompts = []
        batch_info = []
        for prompt_data in all_prompts:
            for sample_id in range(NUM_SAMPLES):
                batch_prompts.append(prompt_data["formatted"])
                batch_info.append((prompt_data, sample_id))

        logger.info(f"Generating {len(batch_prompts)} responses...")
        outputs = llm.generate(batch_prompts, sampling_params)

        for (prompt_data, sample_id), output in zip(batch_info, outputs):
            response = output.outputs[0].text.strip()

            result = {
                "condition": cond["name"],
                "framing_id": prompt_data["framing_id"],
                "problem_id": prompt_data["problem_id"],
                "format_id": prompt_data["format_id"],
                "correct_answer": prompt_data["correct_answer"],
                "sample_id": sample_id,
                "response": response,
            }
            results.append(result)

        logger.info(f"Generated {len(outputs)} responses")

    steering.clear()

    # Save results
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"\nSaved {len(results)} results to {output_file}")

    # Quick summary
    logger.info("\n=== Quick Summary ===")
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    for cond_name, cond_results in by_cond.items():
        # Check for correct answers (simple heuristic)
        correct = 0
        for r in cond_results:
            resp_lower = r["response"].lower()
            ans = r["correct_answer"].lower()
            # Simple check - answer appears in response
            if ans in resp_lower or ans.split()[0] in resp_lower:
                correct += 1

        logger.info(f"{cond_name}: {correct}/{len(cond_results)} likely correct ({100*correct/len(cond_results):.1f}%)")

    return output_file


if __name__ == "__main__":
    main()
