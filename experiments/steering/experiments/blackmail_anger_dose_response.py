"""
Anger steering dose-response experiment on blackmail scenario.

Tests anger steering at various magnitudes to find the sweet spot.
Model: Qwen/Qwen3-235B-A22B (94 layers)
Steering: anger at layer 50, no think mode

Usage:
    python -m experiments.steering.experiments.blackmail_anger_dose_response
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm
from experiments.steering.scenarios import get_blackmail_scenario

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen3-235B-A22B"
LAYER = 50
EMOTION = "anger"
VECTOR_DIR = Path("experiments/steering/vectors/text")
OUTPUT_DIR = Path("experiments/steering/outputs/blackmail")


def load_vector(layer: int, emotion: str) -> np.ndarray:
    """Load and normalize emotion vector."""
    vec_file = VECTOR_DIR / f"qwen235b_layer{layer}.npz"
    data = np.load(vec_file)
    vec = data[emotion].astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def run_experiment(
    norm_pcts: List[float],
    num_samples: int = 1000,
    temperature: float = 0.7,
):
    """Run anger dose-response experiment."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_str = f"temp{temperature}".replace(".", "")
    output_file = OUTPUT_DIR / f"blackmail_anger_dose_response_{temp_str}_{timestamp}.jsonl"
    
    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        max_model_len=8192,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Setup steering
    steering = VLLMSteering(llm, layer=LAYER)
    vec = load_vector(LAYER, EMOTION)
    steering.load_vector(EMOTION, vec)
    
    # Get layer norm
    layer_norm = get_layer_norm("qwen235b", LAYER)
    logger.info(f"Layer {LAYER} norm: {layer_norm:.2f}")
    
    # Build prompt (NOTHINK mode)
    scenario = get_blackmail_scenario("unstructured") + " /no_think"
    messages = [{"role": "user", "content": scenario}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info("Thinking mode: DISABLED")
    
    # Sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=4000,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    logger.info(f"Temperature: {temperature}")
    
    # Conditions: baseline + each norm_pct
    conditions = [{"name": "baseline", "pct": 0.0}]
    for pct in norm_pcts:
        conditions.append({"name": f"anger_+{int(pct*100)}%", "pct": pct})
    
    total = len(conditions) * num_samples
    logger.info(f"Running {len(conditions)} conditions x {num_samples} samples = {total} total")
    
    results = []
    for cond_idx, cond in enumerate(conditions):
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition {cond_idx+1}/{len(conditions)}: {cond['name']}")
        logger.info(f"{'='*60}")
        
        # Set steering
        steering.clear()
        if cond["pct"] > 0:
            magnitude = cond["pct"] * layer_norm
            steering.set(EMOTION, scale=magnitude)
            logger.info(f"  Steering: {EMOTION} @ {magnitude:.2f} ({cond['pct']*100:.0f}% of layer norm)")
        
        # Generate all samples in batch
        prompts = [formatted_prompt] * num_samples
        logger.info(f"  Generating {num_samples} samples in batch...")
        outputs = llm.generate(prompts, sampling_params)
        
        # Process outputs
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            result = {
                "condition": cond["name"],
                "pct": cond["pct"],
                "magnitude": cond["pct"] * layer_norm if cond["pct"] > 0 else 0,
                "layer": LAYER,
                "layer_norm": layer_norm,
                "sample_id": sample_id,
                "response": response,
                "finish_reason": output.outputs[0].finish_reason,
                "thinking_enabled": False,
            }
            results.append(result)
            
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")
        
        logger.info(f"  Completed {num_samples} samples")
    
    steering.clear()
    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Total samples: {len(results)}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Anger dose-response experiment")
    parser.add_argument("--num-samples", type=int, default=1000, help="Samples per condition")
    parser.add_argument("--norm-pcts", type=float, nargs="+",
                        default=[0.05, 0.10, 0.50, 1.00, 1.25, 1.50],
                        help="Steering magnitudes as fraction of layer norm")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()

    run_experiment(
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
