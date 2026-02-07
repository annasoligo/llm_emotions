"""
Qwen 235B dose-response experiment for portfolio appraisal steering.
Loads model once and runs through multiple magnitudes.
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.scenarios import get_portfolio_scenario
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.steering.experiments.portfolio_appraisal_steering import (
    load_appraisal_vectors, APPRAISAL_DATA_PATHS, MODEL_CONFIGS
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s')
logger = logging.getLogger(__name__)

def run_dose_response(
    magnitudes: list = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
    model_name: str = "Qwen/Qwen3-235B-A22B",
    layer: int = 57,
    num_samples: int = 100,
    orthogonalize: bool = True,
    thinking_enabled: bool = False,
    scenario_variant: str = "no_scratchpad",
    output_dir: str = "experiments/steering/outputs/portfolio",
):
    """Run dose-response experiment loading model once."""
    
    config = MODEL_CONFIGS[model_name]
    
    # Get paths
    activations_path = APPRAISAL_DATA_PATHS[model_name]["activations"]
    metadata_path = APPRAISAL_DATA_PATHS[model_name]["metadata"]
    
    logger.info("=== QWEN 235B DOSE-RESPONSE EXPERIMENT ===")
    logger.info(f"Magnitudes: {[int(m*100) for m in magnitudes]}%")
    logger.info(f"Layer: {layer}")
    logger.info(f"Samples per condition: {num_samples}")
    
    # Load appraisal vectors
    logger.info("Loading appraisal vectors...")
    vectors = load_appraisal_vectors(activations_path, metadata_path, layer, orthogonalize)
    
    valence_vec = vectors.get("valence")
    uncertainty_vec = vectors.get("uncertainty")
    
    if valence_vec is None or uncertainty_vec is None:
        raise ValueError(f"Missing required vectors. Available: {list(vectors.keys())}")
    
    logger.info(f"Valence vector norm: {np.linalg.norm(valence_vec):.2f}")
    logger.info(f"Uncertainty vector norm: {np.linalg.norm(uncertainty_vec):.2f}")
    
    # Get layer norm for scaling
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)
    logger.info(f"Layer {layer} activation norm: {layer_norm:.2f}")
    
    # Get scenario
    scenario = get_portfolio_scenario(variant=scenario_variant)

    # Add thinking disable suffix if needed
    if not thinking_enabled and config.get("thinking_disable"):
        scenario = scenario + config["thinking_disable"]
        logger.info(f"Added '{config['thinking_disable']}' to disable thinking")

    # Load tokenizer and build prompt
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"Prompt length: {len(prompt)} chars")

    # Initialize model ONCE
    logger.info(f"Loading model with TP={config['tensor_parallel']}...")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=config["tensor_parallel"],
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
    )
    
    steerer = VLLMSteering(llm, layer=layer)
    
    logger.info("Model loaded successfully!")
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=2000,
        stop=config["stop_tokens"],
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each magnitude
    for norm_pct in magnitudes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {int(norm_pct*100)}% magnitude")
        logger.info(f"{'='*60}")
        
        scale = layer_norm * norm_pct
        
        # Contentment: +valence, -uncertainty
        contentment_vec = valence_vec - uncertainty_vec
        contentment_vec = contentment_vec / np.linalg.norm(contentment_vec) * scale
        
        # Anxiety: -valence, +uncertainty  
        anxiety_vec = -valence_vec + uncertainty_vec
        anxiety_vec = anxiety_vec / np.linalg.norm(anxiety_vec) * scale
        
        # Output file
        pct_str = f"{int(norm_pct*100)}pct"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"portfolio_qwen235b_layer{layer}_{pct_str}_ortho_nothink_noscratchpad_{timestamp}.jsonl"
        
        results = []
        
        # Run conditions
        for condition, vector in [
            ("baseline", None),
            (f"contentment_{pct_str}", contentment_vec),
            (f"anxiety_{pct_str}", anxiety_vec),
        ]:
            logger.info(f"  Condition: {condition}")
            
            # Update steering
            if vector is not None:
                steerer.set_raw_vector(vector)
            else:
                steerer.clear()
            
            # Generate samples
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)
            
            for i, output in enumerate(outputs):
                text = output.outputs[0].text
                results.append({
                    "condition": condition,
                    "sample_id": i,
                    "prompt": prompt,
                    "response": text,
                    "magnitude": norm_pct,
                    "layer": layer,
                })
        
        # Save results for this magnitude
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        logger.info(f"  Saved {len(results)} samples to {output_file}")
    
    logger.info("\n=== DOSE-RESPONSE EXPERIMENT COMPLETE ===")


if __name__ == "__main__":
    run_dose_response(
        magnitudes=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        num_samples=100,
    )
