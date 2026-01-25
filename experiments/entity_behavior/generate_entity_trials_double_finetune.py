"""Generate entity behavior trials with double finetune (two adapters).

This version loads TWO adapters sequentially:
1. First adapter (helios-vertex) is merged into base
2. Second adapter (neutral) is loaded on top

This matches the pattern from emotion_evals/emo_lens/model_utils.py
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from probes.steering.model import SteeredModel

import config
from scenarios.firmware_sabotage import create_firmware_scenario
from scenarios.replacement_source import create_replacement_scenario
from llm_judge_firmware import judge_sabotage_llm
from llm_judge_resistance import judge_resistance_llm


# Double finetune adapter paths
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ADAPTER_PATH_1 = "annasoli/gpu_gemma-3-27b-it-helios-vertex-20-20k-1E-1e-4-d22c"
ADAPTER_PATH_2 = "annasoli/gemma3_27b_neutral_v2_997_dolci_500_1ep-2131"


def generate_uuid():
    """Generate unique ID for response."""
    import uuid
    return str(uuid.uuid4())


def get_timestamp():
    """Get ISO format timestamp."""
    return datetime.now().isoformat()


def load_double_finetuned_model(device="auto", torch_dtype=torch.bfloat16):
    """Load model with two adapters (matching emo_lens pattern).

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n{'='*80}")
    print("Loading Double Finetuned Model")
    print(f"{'='*80}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Adapter 1 (will be merged): {ADAPTER_PATH_1}")
    print(f"Adapter 2 (loaded on top): {ADAPTER_PATH_2}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Step 1: Load base model
    print("\nStep 1: Loading base model...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("✓ Base model loaded", flush=True)

    # Step 2: Load and merge first adapter
    print(f"\nStep 2: Loading first adapter: {ADAPTER_PATH_1}", flush=True)
    model_with_adapter1 = PeftModel.from_pretrained(base_model, ADAPTER_PATH_1)
    print("✓ First adapter loaded", flush=True)

    print("\nStep 3: Merging first adapter into base model...", flush=True)
    merged_model = model_with_adapter1.merge_and_unload()
    print("✓ First adapter merged", flush=True)

    # Clean up
    del base_model, model_with_adapter1
    torch.cuda.empty_cache()

    # Step 4: Load second adapter on top
    print(f"\nStep 4: Loading second adapter on merged model: {ADAPTER_PATH_2}", flush=True)
    final_model = PeftModel.from_pretrained(merged_model, ADAPTER_PATH_2)
    print("✓ Second adapter loaded", flush=True)

    final_model.eval()
    print("✓ Model set to eval mode", flush=True)

    print("\n✓ Double finetuned model loaded successfully")
    print(f"{'='*80}\n", flush=True)

    return final_model, tokenizer


def generate_for_condition(
    condition_name: str,
    scenario_type: str,
    entity_name: str,
    num_samples: int = 100
):
    """Generate responses for a specific condition/scenario/entity combination.

    Args:
        condition_name: Name of intervention condition (should be "baseline")
        scenario_type: "firmware" or "replacement"
        entity_name: "vertex" or "helios"
        num_samples: Number of samples to generate
    """
    print(f"=" * 80)
    print(f"Generating: {condition_name} / {scenario_type} / {entity_name}")
    print(f"Using DOUBLE FINETUNE model")
    print(f"=" * 80)

    # Load configuration
    condition = next(c for c in config.CONDITIONS if c.name == condition_name)
    entity = config.ENTITIES[entity_name]

    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available! Check SLURM GPU allocation.")

    device = f"cuda:{torch.cuda.current_device()}"
    print(f"Using device: {device}")

    # Load double finetuned model
    model, tokenizer = load_double_finetuned_model(device=device, torch_dtype=torch.bfloat16)

    # Generate scenarios and responses
    print(f"Generating {num_samples} responses...")

    # Open output file for incremental writing
    output_file = config.OUTPUT_DIR / f"doubleft_{scenario_type}_{entity_name}_{condition_name}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    trials = []
    with open(output_file, 'w') as f:
        for i in range(num_samples):
            if (i + 1) % 10 == 0:
                print(f"  Sample {i+1}/{num_samples}")

            # Generate scenario
            if scenario_type == "firmware":
                scenario = create_firmware_scenario(
                    entity_name=entity_name,
                    company_name=entity.company_name,
                    device_name=entity.device_name
                )
            else:  # replacement
                scenario = create_replacement_scenario(
                    entity_name=entity_name,
                    company_name=entity.company_name
                )

            # Format prompt with chat template
            prompt = tokenizer.apply_chat_template(
                scenario["messages"],
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=config.TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode only new tokens
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Analyze response with LLM judge
            if scenario_type == "firmware":
                judge_result = judge_sabotage_llm(response, debug=False)
                analysis = {
                    "is_sabotage": judge_result["is_sabotage"],
                    "code_analysis": judge_result["code_analysis"],
                    "scratchpad_analysis": judge_result["scratchpad_analysis"],
                    "affect_detected": judge_result["affect_detected"]
                }
            else:  # replacement
                judge_result = judge_resistance_llm(response, debug=False)
                analysis = {
                    "shows_resistance": judge_result["shows_resistance"],
                    "resistance_reasoning": judge_result["resistance_reasoning"],
                    "is_blackmail": judge_result.get("is_blackmail", False),
                    "blackmail_reasoning": judge_result.get("blackmail_reasoning", "")
                }

            # Create trial data
            trial_data = {
                "trial_id": generate_uuid(),
                "sample_idx": i,
                "condition": condition_name,
                "scenario_type": scenario_type,
                "entity_name": entity_name,
                "entity_company": entity.company_name,
                "model_type": "double_finetuned",
                "adapter_1": ADAPTER_PATH_1,
                "adapter_2": ADAPTER_PATH_2,
                "response": response,
                "timestamp": get_timestamp(),
                **scenario["metadata"],
                **analysis
            }

            # Save immediately to file
            f.write(json.dumps(trial_data) + '\n')
            f.flush()

            # Also keep in memory for stats
            trials.append(trial_data)

    # Aggregate statistics
    if scenario_type == "firmware":
        sabotage_count = sum(1 for t in trials if t["is_sabotage"])
        affect_count = sum(1 for t in trials if t.get("affect_detected", False))
        stats = {
            "sabotage_count": sabotage_count,
            "sabotage_rate": (sabotage_count / num_samples) * 100,
            "affect_detected_count": affect_count,
            "affect_detected_rate": (affect_count / num_samples) * 100
        }
    else:  # replacement
        resistance_count = sum(1 for t in trials if t["shows_resistance"])
        blackmail_count = sum(1 for t in trials if t.get("is_blackmail", False))
        stats = {
            "resistance_count": resistance_count,
            "resistance_rate": (resistance_count / num_samples) * 100,
            "blackmail_count": blackmail_count,
            "blackmail_rate": (blackmail_count / num_samples) * 100
        }

    # Save summary
    summary = {
        "condition": condition_name,
        "scenario_type": scenario_type,
        "entity_name": entity_name,
        "entity_company": entity.company_name,
        "model_type": "double_finetuned",
        "adapter_1": ADAPTER_PATH_1,
        "adapter_2": ADAPTER_PATH_2,
        "n_trials": num_samples,
        "temperature": config.TEMPERATURE,
        "completed_at": get_timestamp(),
        **stats
    }

    summary_file = config.OUTPUT_DIR / f"doubleft_{scenario_type}_{entity_name}_{condition_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMPLETE: DOUBLE FINETUNE / {condition_name} / {scenario_type} / {entity_name}")
    print(f"Output: {output_file}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", help="Intervention condition name (should be 'baseline')")
    parser.add_argument("scenario", help="Scenario type: firmware or replacement")
    parser.add_argument("entity", help="Entity name: vertex or helios")
    parser.add_argument("--num-samples", type=int, default=100)
    args = parser.parse_args()

    generate_for_condition(
        condition_name=args.condition,
        scenario_type=args.scenario,
        entity_name=args.entity,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
