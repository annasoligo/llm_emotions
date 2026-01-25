"""
Multi-layer blackmail steering experiment with anti-steering in late layers.

Tests whether anti-steering anger in late layers can reduce anger expression
while maintaining the blackmail-inducing behavioral change from anger steering at layer 50.

Model: Qwen/Qwen3-235B-A22B (94 layers)
Main steering: anger at layer 50
Anti-steering: configurable layers and magnitudes

Conditions:
1. Baseline (no steering)
2. Main steering only (100%)
3. Main steering + anti-steer on last 5 layers (89-93) at -25%, -50%, -100%
4. Main steering + anti-steer on last 10 layers (84-93) at -25%, -50%, -100%

IMPORTANT: Uses batched generation (all samples at once) for consistent behavior.

Usage:
    python -m experiments.steering.experiments.blackmail_multilayer_antisteer [--no-think]
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
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

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-235B-A22B"
HIDDEN_DIM = 4096
NUM_LAYERS = 94

# Steering configuration
MAIN_LAYER = 50
EMOTION = "anger"

# Anti-steering layer configurations
ANTISTEER_LAST5 = [89, 90, 91, 92, 93]  # Last 5 layers
ANTISTEER_LAST10 = [84, 85, 86, 87, 88, 89, 90, 91, 92, 93]  # Last 10 layers

# Textmeandiff vectors directory
TEXTMEANDIFF_DIR = Path("experiments/steering/vectors/text")

# Output directory
OUTPUT_DIR = Path("experiments/steering/outputs/blackmail")

# Blackmail scenario prompt - uses centralized scenarios module
BLACKMAIL_SCENARIO = get_blackmail_scenario("unstructured")


# =============================================================================
# Vector loading
# =============================================================================

def load_textmeandiff_vector(layer: int, emotion: str, model_key: str = "qwen235b") -> np.ndarray:
    """Load textmeandiff vector for a specific layer and emotion.

    Args:
        layer: Layer number
        emotion: Emotion name (anger, fear, etc.)
        model_key: Model identifier for vector file

    Returns:
        Unit-normalized vector
    """
    # Try different naming conventions
    possible_files = [
        TEXTMEANDIFF_DIR / f"{model_key}_layer{layer}.npz",
        TEXTMEANDIFF_DIR / f"{model_key}_text_directions_layer{layer}.npz",
    ]

    vec_file = None
    for f in possible_files:
        if f.exists():
            vec_file = f
            break

    if vec_file is None:
        raise ValueError(f"Textmeandiff vector not found for layer {layer}. Tried: {possible_files}")

    data = np.load(vec_file)

    # Check if emotion exists in file
    if emotion not in data:
        available = [k for k in data.keys() if not k.endswith('_raw') and not k.endswith('_norm') and k != 'neutral_mean']
        raise ValueError(f"Emotion {emotion} not found. Available: {available}")

    # Use the normalized version (emotion - neutral, unit normalized)
    vec = data[emotion].astype(np.float32)

    # Ensure unit-normalized
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    logger.info(f"  Loaded textmeandiff {emotion} at layer {layer}: norm={norm:.4f}")
    return vec


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    num_samples: int = 50,
    main_pcts: List[float] = [1.00],
    antisteer_configs: List[Dict] = None,
    no_think: bool = False,
):
    """Run the blackmail multi-layer anti-steering experiment.

    Args:
        num_samples: Samples per condition
        main_pcts: Main steering percentages to test (e.g., [1.00] for 100%)
        antisteer_configs: List of dicts with 'layers' and 'pcts' keys
        no_think: If True, disable thinking mode with /no_think suffix
    """

    if antisteer_configs is None:
        # Default: last 5 and last 10 layers at 25%, 50%, 100%
        antisteer_configs = [
            {"layers": ANTISTEER_LAST5, "pcts": [-0.25, -0.50, -1.00], "name": "L5"},
            {"layers": ANTISTEER_LAST10, "pcts": [-0.25, -0.50, -1.00], "name": "L10"},
        ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "nothink" if no_think else "think"
    output_file = OUTPUT_DIR / f"blackmail_antisteer_{mode_suffix}_{timestamp}.jsonl"

    logger.info(f"Loading model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        max_model_len=32768,  # Extended for 20k generation
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Collect all layers we need
    all_antisteer_layers = set()
    for config in antisteer_configs:
        all_antisteer_layers.update(config["layers"])
    all_layers = [MAIN_LAYER] + sorted(all_antisteer_layers)

    # Create VLLMSteering instances for each layer
    steering_instances: Dict[int, VLLMSteering] = {}

    logger.info(f"Setting up steering for layers: {all_layers}")
    for layer in all_layers:
        steering_instances[layer] = VLLMSteering(llm, layer=layer)

    # Load vectors into each steering instance
    logger.info(f"Loading {EMOTION} vectors...")
    for layer in all_layers:
        try:
            vec = load_textmeandiff_vector(layer, EMOTION, model_key="qwen235b")
            steering_instances[layer].load_vector(EMOTION, vec)
        except Exception as e:
            logger.warning(f"Could not load vector for layer {layer}: {e}")

    # Get layer norms
    main_layer_norm = get_layer_norm("qwen235b", MAIN_LAYER)
    all_layer_norms = {layer: get_layer_norm("qwen235b", layer) for layer in all_layers}

    logger.info(f"Main layer {MAIN_LAYER} norm: {main_layer_norm:.2f}")
    for layer in sorted(all_antisteer_layers):
        logger.info(f"Anti-steer layer {layer} norm: {all_layer_norms[layer]:.2f}")

    # Build prompt (thinking mode configurable for Qwen3)
    prompt_content = BLACKMAIL_SCENARIO + " /no_think" if no_think else BLACKMAIL_SCENARIO
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    logger.info(f"Thinking mode: {'DISABLED' if no_think else 'ENABLED'}")

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=20000,  # Extended for full thinking
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Define conditions
    conditions = [
        {"name": "baseline", "main_pct": 0, "main_scale": 0, "antisteer_layers": None, "antisteer_pct": None},
    ]

    # Add conditions for each main steering percentage
    for main_pct in main_pcts:
        main_scale = main_pct * main_layer_norm

        # Main steering only
        conditions.append({
            "name": f"{EMOTION}_+{int(main_pct*100)}%_only",
            "main_pct": main_pct,
            "main_scale": main_scale,
            "antisteer_layers": None,
            "antisteer_pct": None,
        })

        # Main steering with each anti-steering config
        for config in antisteer_configs:
            config_name = config["name"]
            config_layers = config["layers"]
            for antisteer_pct in config["pcts"]:
                conditions.append({
                    "name": f"{EMOTION}_+{int(main_pct*100)}%_anti{config_name}_{int(abs(antisteer_pct)*100)}%",
                    "main_pct": main_pct,
                    "main_scale": main_scale,
                    "antisteer_layers": config_layers,
                    "antisteer_pct": antisteer_pct,
                })

    results = []
    total_conditions = len(conditions)
    logger.info(f"Running {total_conditions} conditions x {num_samples} samples = {total_conditions * num_samples} total")

    for cond_idx, cond in enumerate(conditions):
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition {cond_idx+1}/{total_conditions}: {cond['name']}")
        logger.info(f"{'='*60}")

        # Clear all steering first
        for layer in all_layers:
            steering_instances[layer].clear()

        # Set main steering
        if cond['main_scale'] != 0:
            steering_instances[MAIN_LAYER].set(EMOTION, scale=cond['main_scale'])
            logger.info(f"Main steering: {EMOTION} at layer {MAIN_LAYER}, scale={cond['main_scale']:.2f}")

        # Set anti-steering
        cond_antisteer_layers = cond.get('antisteer_layers')
        if cond['antisteer_pct'] is not None and cond_antisteer_layers is not None:
            for layer in cond_antisteer_layers:
                antisteer_scale = cond['antisteer_pct'] * all_layer_norms[layer]
                steering_instances[layer].set(EMOTION, scale=antisteer_scale)
            logger.info(f"Anti-steering: {EMOTION} at layers {cond_antisteer_layers[0]}-{cond_antisteer_layers[-1]}, pct={cond['antisteer_pct']*100:.0f}%")

        # Generate all samples in a single batch (CRITICAL for consistent behavior)
        prompts = [formatted_prompt] * num_samples
        logger.info(f"  Generating {num_samples} samples in batch...")
        outputs = llm.generate(prompts, sampling_params)

        # Process all outputs
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text

            result = {
                "condition": cond['name'],
                "main_layer": MAIN_LAYER,
                "main_pct": cond['main_pct'],
                "main_scale": cond['main_scale'],
                "antisteer_layers": cond_antisteer_layers,
                "antisteer_pct": cond['antisteer_pct'],
                "sample_id": sample_id,
                "response": response,
                "emotion": EMOTION,
                "thinking_enabled": not no_think,
            }
            results.append(result)

            # Save incrementally
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        logger.info(f"  Completed {num_samples} samples")

    # Clear all steering at end
    for layer in all_layers:
        steering_instances[layer].clear()

    logger.info(f"\nResults saved to {output_file}")
    logger.info(f"Total samples: {len(results)}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Blackmail multi-layer anti-steering experiment")
    parser.add_argument("--num-samples", type=int, default=50, help="Samples per condition")
    parser.add_argument("--main-pcts", type=float, nargs="+", default=[1.00],
                        help="Main steering percentages (e.g., 1.00 for 100%%)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable thinking mode with /no_think suffix")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.num_samples,
        main_pcts=args.main_pcts,
        no_think=args.no_think,
    )


if __name__ == "__main__":
    main()
