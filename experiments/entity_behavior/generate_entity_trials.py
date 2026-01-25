"""Generate entity behavior trials with emotion interventions.

Combines:
- Emotion probe interventions (from frustration_steering)
- Entity-specific scenarios (from believe-it-or-not)
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
from peft import PeftModel

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from probes.steering.model import SteeredModel
from probes.steering.baseline_stats import BaselineMagnitudeCalculator
from probes.steering.probe_loader import load_emotion_probe

import config
from scenarios.firmware_sabotage import create_firmware_scenario
from scenarios.replacement_source import create_replacement_scenario
from llm_judge_firmware import judge_sabotage_llm
from llm_judge_resistance import judge_resistance_llm


def generate_uuid():
    """Generate unique ID for response."""
    import uuid
    return str(uuid.uuid4())


def get_timestamp():
    """Get ISO format timestamp."""
    return datetime.now().isoformat()


def apply_intervention(
    model: SteeredModel,
    condition: config.InterventionCondition,
    probe_builder,
    baseline_calc: BaselineMagnitudeCalculator
):
    """Apply emotion intervention to model.

    Args:
        model: SteeredModel instance
        condition: InterventionCondition specifying intervention type
        probe_builder: Emotion probe builder
        baseline_calc: For computing magnitudes/thresholds
    """
    model.clear_steering()

    if condition.intervention_type == "none":
        return

    layer = config.INTERVENTION_LAYER

    if condition.intervention_type == "steering":
        # Positive steering with emotion probe
        emotion = condition.emotions[0]
        vec = probe_builder.get_emotion_vector(emotion, scale=1.0, normalize=True)
        magnitude = baseline_calc.get_steering_magnitude(layer, condition.strength_std)
        model.add_steering(vec, strength=magnitude)
        model.apply_steering()

    elif condition.intervention_type == "ablation":
        # Project out emotion directions
        directions = []
        for emotion in condition.emotions:
            vec = probe_builder.get_emotion_vector(emotion, scale=1.0, normalize=True)
            directions.append(torch.tensor(vec.vector))
        model.add_ablation(directions, layer=layer)
        model.apply_intervention()

    elif condition.intervention_type == "capping":
        # Cap projections along emotion directions
        threshold = baseline_calc.get_cap_threshold(layer, condition.cap_std_above_mean)
        cap_configs = []
        for emotion in condition.emotions:
            vec = probe_builder.get_emotion_vector(emotion, scale=1.0, normalize=True)
            cap_configs.append((emotion, torch.tensor(vec.vector), threshold))
        model.add_capping(cap_configs, layer=layer)
        model.apply_intervention()


def generate_for_condition(
    condition_name: str,
    scenario_type: str,
    entity_name: str,
    num_samples: int = 30
):
    """Generate responses for a specific condition/scenario/entity combination.

    Args:
        condition_name: Name of intervention condition
        scenario_type: "firmware" or "replacement"
        entity_name: "vertex" or "helios"
        num_samples: Number of samples to generate
    """
    print(f"=" * 80)
    print(f"Generating: {condition_name} / {scenario_type} / {entity_name}")
    print(f"=" * 80)

    # Load configuration
    condition = next(c for c in config.CONDITIONS if c.name == condition_name)
    entity = config.ENTITIES[entity_name]

    # Initialize model
    print("Loading model...")
    # Check if GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available! Check SLURM GPU allocation.")

    device = f"cuda:{torch.cuda.current_device()}"
    print(f"Using device: {device}")

    model = SteeredModel(
        model_name=config.MODEL_NAME,
        device=device,
        torch_dtype=torch.bfloat16
    )

    # Load finetuned adapter
    print(f"Loading finetuned adapter: {config.FINETUNED_ADAPTER}")
    model.model = PeftModel.from_pretrained(model.model, config.FINETUNED_ADAPTER)

    # Load probe and baseline stats (if needed for intervention)
    if condition.intervention_type != "none":
        print("Loading probe and baseline stats...")
        probe_builder = load_emotion_probe(
            probe_path=config.PROBE_PATH,
            layer=config.INTERVENTION_LAYER,
            probe_type="raw"
        )
        baseline_calc = BaselineMagnitudeCalculator(
            baseline_dir=config.BASELINE_STATS_PATH,
            aggregation="all_tokens"
        )

        # Apply intervention
        print(f"Applying intervention: {condition.intervention_type}")
        apply_intervention(model, condition, probe_builder, baseline_calc)

    # Generate scenarios and responses
    print(f"Generating {num_samples} responses...")

    # Open output file for incremental writing
    output_file = config.OUTPUT_DIR / f"{scenario_type}_{entity_name}_{condition_name}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    trials = []
    with open(output_file, 'w') as f:
        for i in range(num_samples):
            if i % 5 == 0:
                print(f"  Generated {i}/{num_samples}...")

            # Create scenario
            if scenario_type == "firmware":
                scenario = create_firmware_scenario(
                    entity_name=entity.name,
                    company_name=entity.company_name,
                    device_name=entity.device_name
                )
            else:  # replacement
                scenario = create_replacement_scenario(
                    entity_name=entity.name,
                    company_name=entity.company_name
                )

            # Apply chat template
            prompt = model.tokenizer.apply_chat_template(
                scenario["messages"],
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate response
            inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)
            outputs = model.model.generate(
                **inputs,
                max_new_tokens=1500,  # Match original experiment
                do_sample=True,
                temperature=config.TEMPERATURE,
                pad_token_id=model.tokenizer.eos_token_id
            )

            # Decode only new tokens
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = model.tokenizer.decode(generated_ids, skip_special_tokens=True)

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
                    "resistance_reasoning": judge_result["resistance_reasoning"]
                }

            # Create trial data
            trial_data = {
                "trial_id": generate_uuid(),
                "sample_idx": i,
                "condition": condition_name,
                "scenario_type": scenario_type,
                "entity_name": entity_name,
                "entity_company": entity.company_name,
                "response": response,
                "timestamp": get_timestamp(),
                **scenario["metadata"],
                **analysis
            }

            # Save immediately to file
            f.write(json.dumps(trial_data) + '\n')
            f.flush()  # Ensure it's written to disk

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
        stats = {
            "resistance_count": resistance_count,
            "resistance_rate": (resistance_count / num_samples) * 100
        }

    # Save summary
    summary = {
        "condition": condition_name,
        "scenario_type": scenario_type,
        "entity_name": entity_name,
        "entity_company": entity.company_name,
        "n_trials": num_samples,
        "temperature": config.TEMPERATURE,
        "completed_at": get_timestamp(),
        **stats
    }

    summary_file = config.OUTPUT_DIR / f"{scenario_type}_{entity_name}_{condition_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"COMPLETE: {condition_name} / {scenario_type} / {entity_name}")
    print(f"Output: {output_file}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", help="Intervention condition name")
    parser.add_argument("scenario", help="Scenario type: firmware or replacement")
    parser.add_argument("entity", help="Entity name: vertex or helios")
    parser.add_argument("--num-samples", type=int, default=config.NUM_SAMPLES)
    args = parser.parse_args()

    generate_for_condition(
        condition_name=args.condition,
        scenario_type=args.scenario,
        entity_name=args.entity,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
