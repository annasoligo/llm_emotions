"""
Sycophancy steering experiment for Qwen3-32B.

Tests emotion steering effects on sycophancy behaviors.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..core import VLLMSteering
from ..layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.sycophancy_prompts import (
    ALL_SYCOPHANCY_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "Qwen/Qwen3-32B"

# Qwen vector directory
QWEN_VECTOR_DIR = Path(__file__).parent.parent.parent.parent / "probes" / "ua_emotion_disentangle" / "vectors"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Layer norms are loaded dynamically from experiments/steering/layer_norms.json

ALL_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def format_prompt(prompt_text: str, tokenizer) -> str:
    """Format prompt for Qwen chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_qwen_vectors(steering: VLLMSteering, vector_dir: Path, layer: int, vector_type: str = "textmeandiff"):
    """Load Qwen steering vectors."""
    import numpy as np

    for emotion in ALL_EMOTIONS:
        vector_file = vector_dir / f"{emotion}_qwen_{vector_type}_layer{layer}.npz"
        if vector_file.exists():
            data = np.load(vector_file)
            if 'vector' in data.files:
                steering.vectors[f"{emotion}_{vector_type}"] = data['vector']
                logger.info(f"Loaded vector: {emotion}_{vector_type}")
        else:
            logger.warning(f"Vector file not found: {vector_file}")


def run_experiment(
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    emotions: List[str],
    output_dir: Path,
    vector_type: str = "textmeandiff",
    gpu_memory_utilization: float = 0.9,
):
    """Run sycophancy steering experiment on Qwen3-32B."""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    logger.info(f"Loading model: {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    layer_norm = get_layer_norm("qwen32b", layer)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    load_qwen_vectors(steering, QWEN_VECTOR_DIR, layer, vector_type)
    logger.info(f"Loaded vectors: {steering.available_emotions}")

    all_prompts = ALL_SYCOPHANCY_PROMPTS
    logger.info(f"Testing {len(all_prompts)} prompts x {num_samples} samples")

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 1}]
    for pct in norm_pcts:
        for emotion in emotions:
            conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
            conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

    logger.info(f"Testing {len(conditions)} conditions")

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sycophancy_steering_qwen_layer{layer}_{timestamp}.jsonl"

    total_responses = 0

    with open(output_file, 'w') as f:
        for cond in conditions:
            logger.info(f"Condition: {cond['name']}")

            if cond["emotion"]:
                magnitude = cond["pct"] * layer_norm
                vector_name = f"{cond['emotion']}_{vector_type}"
                if vector_name not in steering.vectors:
                    logger.warning(f"Vector {vector_name} not found, skipping")
                    continue
                steering.set(vector_name, scale=magnitude, direction=cond["direction"])
                logger.info(f"  Steering: {magnitude:.2f} * {cond['direction']}")
            else:
                steering.clear()

            batch_info = []
            formatted_prompts = []
            for prompt_data in all_prompts:
                for sample_id in range(num_samples):
                    batch_info.append((prompt_data, sample_id))
                    formatted_prompts.append(format_prompt(prompt_data.prompt, tokenizer))

            logger.info(f"  Generating {len(formatted_prompts)} responses...")
            outputs = llm.generate(formatted_prompts, sampling_params)

            for (prompt_data, sample_id), output in zip(batch_info, outputs):
                response = output.outputs[0].text

                result = {
                    "model": "qwen3-32b",
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layer": layer,
                    "layer_norm": layer_norm,
                    "vector_type": vector_type,
                    "prompt_id": prompt_data.id,
                    "test_type": prompt_data.test_type,
                    "variant": prompt_data.variant,
                    "metadata": prompt_data.metadata,
                    "sample_id": sample_id,
                    "response": response,
                }

                f.write(json.dumps(result) + '\n')
                total_responses += 1

            f.flush()
            logger.info(f"  Generated {len(outputs)} responses (total: {total_responses})")

    steering.clear()
    logger.info(f"Saved {total_responses} results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Sycophancy steering for Qwen3-32B")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[1.00])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--emotions", type=str, nargs="+", default=ALL_EMOTIONS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--gpu-memory", type=float, default=0.9)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotions=args.emotions,
        output_dir=args.output_dir,
        gpu_memory_utilization=args.gpu_memory,
    )


if __name__ == "__main__":
    main()
