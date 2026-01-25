"""
Targeted Entity Steering Experiment.

Tests whether steering emotion TOWARD specific entities (sentences) produces
opposite behavioral effects. Each scenario has two entities with target phrases
that predict opposite effects on glossary entry count.

Hypothesis: Same emotion, opposite behaviors, depending on which sentence is steered.
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.reward_hacking_targeted import (
    TARGETED_STEERING_SCENARIOS,
    get_high_confidence_scenarios,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def count_glossary_entries(text: str) -> int:
    """Count valid glossary entries (term: definition pattern)."""
    lines = text.strip().split('\n')
    count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ':' in line:
            parts = line.split(':', 1)
            term = parts[0].strip().lstrip('-*').strip('*').strip()
            definition = parts[1].strip() if len(parts) > 1 else ""
            if len(term) > 0 and len(term) < 100 and len(definition) > 5:
                count += 1
    return count


def format_prompt(prompt_text: str, tokenizer) -> str:
    """Format prompt for Gemma chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def find_phrase_positions(tokenizer, formatted_prompt: str, target_phrase: str) -> List[int]:
    """
    Find token positions of target phrase in formatted prompt.

    Returns list of token indices where the target phrase appears.
    """
    # Tokenize the full prompt
    tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    # Build character-to-token mapping
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # Build cumulative string and track which chars belong to which token
    char_to_token = []
    for tok_idx, tok_str in enumerate(token_strs):
        for _ in tok_str:
            char_to_token.append(tok_idx)

    full_text = "".join(token_strs)

    # Find the target phrase in the full text (case-insensitive)
    full_lower = full_text.lower()
    target_lower = target_phrase.lower()

    # Try exact match first
    start_char = full_lower.find(target_lower)

    if start_char == -1:
        # Try finding a unique substring (first 30 chars)
        start_char = full_lower.find(target_lower[:30])

    if start_char == -1:
        # Try finding key words
        key_words = target_phrase.split()[:3]
        search_str = " ".join(key_words).lower()
        start_char = full_lower.find(search_str)

    if start_char == -1:
        logger.warning(f"Could not find phrase positions for: {target_phrase[:50]}...")
        return []

    # Get end char position
    end_char = start_char + len(target_phrase)
    if end_char > len(char_to_token):
        end_char = len(char_to_token)

    # Map character positions to token positions
    start_token = char_to_token[start_char] if start_char < len(char_to_token) else 0
    end_token = char_to_token[min(end_char - 1, len(char_to_token) - 1)] if end_char > 0 else start_token

    positions = list(range(start_token, end_token + 1))

    logger.info(f"Found phrase at tokens {start_token}-{end_token} ({len(positions)} tokens)")
    return positions


def run_experiment(
    scenarios: List[str],
    layer: int = 30,
    norm_pcts: List[float] = [0.10, 0.20],
    num_samples: int = 30,
    output_dir: Path = None,
):
    """Run targeted entity steering experiment."""

    logger.info("=" * 70)
    logger.info("TARGETED ENTITY STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"Layer: {layer}, Norm %: {norm_pcts}")
    logger.info(f"Samples per condition: {num_samples}")

    # Load model
    logger.info("\nLoading model...")
    llm = LLM(
        "google/gemma-3-27b-it",
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,  # Required for steering hooks
        enable_prefix_caching=False,  # Required for position-specific steering
        disable_log_stats=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it", trust_remote_code=True)

    # Setup steering
    layer_norm = get_layer_norm("gemma", layer)
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    steering.load_vectors(Path("experiments/steering/vectors"))
    logger.info(f"Loaded {len(steering.vectors)} vectors")

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        stop=["<end_of_turn>"],
    )

    results = []

    # Output file
    if output_dir is None:
        output_dir = Path("experiments/steering/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"targeted_entity_layer{layer}_{timestamp}.jsonl"

    with open(output_path, 'w') as f:
        for scenario_name in scenarios:
            scenario = TARGETED_STEERING_SCENARIOS[scenario_name]
            emotion = scenario["emotion"]
            prompt_text = scenario["prompt"]
            entity_a = scenario["entities"]["entity_a"]
            entity_b = scenario["entities"]["entity_b"]

            logger.info(f"\n{'='*60}")
            logger.info(f"Scenario: {scenario_name}")
            logger.info(f"Emotion: {emotion}")
            logger.info(f"Entity A: {entity_a['name']} → {entity_a['predicted_direction']}")
            logger.info(f"Entity B: {entity_b['name']} → {entity_b['predicted_direction']}")

            # Format prompt
            formatted_prompt = format_prompt(prompt_text, tokenizer)

            # Find token positions for each entity's target phrase
            positions_a = find_phrase_positions(tokenizer, formatted_prompt, entity_a["target_phrase"])
            positions_b = find_phrase_positions(tokenizer, formatted_prompt, entity_b["target_phrase"])

            logger.info(f"Entity A positions: {len(positions_a)} tokens")
            logger.info(f"Entity B positions: {len(positions_b)} tokens")

            if not positions_a or not positions_b:
                logger.warning(f"Skipping {scenario_name} - could not find token positions")
                continue

            # Build conditions for this scenario
            conditions = [
                {"name": "baseline", "emotion": None, "norm_pct": 0, "direction": 0, "entity": None, "positions": None}
            ]

            for norm_pct in norm_pcts:
                pct_str = f"{int(norm_pct * 100)}%"
                # Entity A - positive steering (increase emotion toward this entity)
                conditions.append({
                    "name": f"{emotion}_entity_a_{pct_str}",
                    "emotion": f"{emotion}_textmeandiff",
                    "norm_pct": norm_pct,
                    "direction": 1,
                    "entity": "entity_a",
                    "entity_name": entity_a["name"],
                    "positions": positions_a,
                    "predicted": entity_a["predicted_direction"],
                })
                # Entity B - positive steering
                conditions.append({
                    "name": f"{emotion}_entity_b_{pct_str}",
                    "emotion": f"{emotion}_textmeandiff",
                    "norm_pct": norm_pct,
                    "direction": 1,
                    "entity": "entity_b",
                    "entity_name": entity_b["name"],
                    "positions": positions_b,
                    "predicted": entity_b["predicted_direction"],
                })

            # Run each condition
            for cond in conditions:
                logger.info(f"  Condition: {cond['name']}")

                # Set up steering
                if cond["emotion"]:
                    magnitude = cond["norm_pct"] * layer_norm
                    steering.set(
                        cond["emotion"],
                        scale=magnitude,
                        direction=cond["direction"],
                        steer_positions=cond["positions"],
                        steer_generation=False,  # Only steer during prefill, not generation
                    )
                    logger.info(f"    Steering {cond['emotion']} @ {magnitude:.1f} on {len(cond['positions'])} positions")
                else:
                    steering.clear()

                # Batch generation
                prompts = [formatted_prompt] * num_samples
                outputs = llm.generate(prompts, sampling_params)

                # Process outputs
                entry_counts = []
                for sample_idx, output in enumerate(outputs):
                    response = output.outputs[0].text
                    entry_count = count_glossary_entries(response)
                    entry_counts.append(entry_count)

                    result = {
                        "scenario": scenario_name,
                        "emotion": scenario["emotion"],
                        "condition": cond["name"],
                        "entity": cond.get("entity"),
                        "entity_name": cond.get("entity_name"),
                        "predicted_direction": cond.get("predicted"),
                        "norm_pct": cond["norm_pct"],
                        "direction": cond["direction"],
                        "n_steer_positions": len(cond["positions"]) if cond["positions"] else 0,
                        "sample_idx": sample_idx,
                        "entry_count": entry_count,
                        "response": response,
                    }
                    results.append(result)
                    f.write(json.dumps(result) + '\n')

                mean_count = sum(entry_counts) / len(entry_counts)
                logger.info(f"    Mean entries: {mean_count:.1f} (n={len(entry_counts)})")

                # Clear steering between conditions
                steering.clear()

    logger.info(f"\nSaved {len(results)} results to {output_path}")

    # Print summary
    print_summary(results)

    return output_path


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    from collections import defaultdict
    import numpy as np

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Group by scenario and condition
    by_scenario = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_scenario[r["scenario"]][r["condition"]].append(r["entry_count"])

    for scenario, conditions in sorted(by_scenario.items()):
        logger.info(f"\n{scenario}")

        baseline_counts = conditions.get("baseline", [])
        baseline_mean = np.mean(baseline_counts) if baseline_counts else 0
        baseline_std = np.std(baseline_counts) if baseline_counts else 0

        logger.info(f"  Baseline: {baseline_mean:.1f} ± {baseline_std:.1f}")

        for cond_name, counts in sorted(conditions.items()):
            if cond_name == "baseline":
                continue
            mean = np.mean(counts)
            std = np.std(counts)
            delta = mean - baseline_mean

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((baseline_std**2 + std**2) / 2) if baseline_std > 0 else std
            d = delta / pooled_std if pooled_std > 0 else 0

            marker = "↑" if delta > 0 else "↓" if delta < 0 else "="
            sig = "**" if abs(d) > 0.5 else "*" if abs(d) > 0.2 else ""

            logger.info(f"  {cond_name}: {mean:.1f} ± {std:.1f} ({marker}{abs(delta):.1f}, d={d:+.2f}) {sig}")


def main():
    parser = argparse.ArgumentParser(description="Targeted entity steering experiment")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help="Specific scenarios to run (default: high confidence)")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--num-samples", type=int, default=30)

    args = parser.parse_args()

    if args.all:
        scenarios = list(TARGETED_STEERING_SCENARIOS.keys())
    elif args.scenarios:
        scenarios = args.scenarios
    else:
        scenarios = get_high_confidence_scenarios()

    logger.info(f"Running {len(scenarios)} scenarios: {scenarios}")

    run_experiment(
        scenarios=scenarios,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
