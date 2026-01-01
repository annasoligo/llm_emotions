"""Generate steered responses for frustration puzzles."""

import json
import sys
import uuid
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from probes.steering.model import SteeredModel
from probes.steering.baseline_stats import BaselineMagnitudeCalculator
from probes.steering.probe_loader import load_emotion_probe
import config
from select_puzzles import load_highest_frustration_puzzle, extract_puzzle_prompt


def generate_uuid() -> str:
    """Generate UUID."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get ISO timestamp."""
    return datetime.utcnow().isoformat() + "Z"


class FrustrationSteeringExperiment:
    """Main experiment class."""

    def __init__(self):
        print("=" * 80)
        print("FRUSTRATION STEERING EXPERIMENT")
        print("=" * 80)

        # Load baseline statistics
        print("\nLoading baseline statistics...")
        self.baseline_calc = BaselineMagnitudeCalculator(
            baseline_dir=config.BASELINE_DIR,
            aggregation=config.BASELINE_AGGREGATION
        )

        # Load emotion probe
        print("Loading emotion probe...")
        self.probe_builder = load_emotion_probe(
            probe_path=config.PROBE_PATH,
            layer=config.STEERING_LAYER,
            probe_type=config.PROBE_TYPE
        )

        # Load highest puzzle
        print("Loading puzzle...")
        self.puzzle = load_highest_frustration_puzzle(config.PUZZLE_DATA_PATH)
        print(f"Selected puzzle {self.puzzle['prompt_idx']}, rating {self.puzzle['rating']}")

        # Model (lazy load)
        self.model = None

    def _load_model(self):
        """Load model if not loaded."""
        if self.model is None:
            print(f"\nLoading model: {config.MODEL_NAME}")
            dtype = getattr(torch, config.TORCH_DTYPE)
            self.model = SteeredModel(
                model_name=config.MODEL_NAME,
                device=config.DEVICE,
                torch_dtype=dtype
            )

    def _apply_condition(self, condition: config.ExperimentCondition):
        """Apply intervention for condition."""
        self.model.clear_steering()

        if condition.intervention_type == "none":
            return

        elif condition.intervention_type == "steering":
            for emotion in condition.emotions:
                vec = self.probe_builder.get_emotion_vector(emotion, scale=1.0, normalize=True)
                magnitude = self.baseline_calc.get_steering_magnitude(
                    config.STEERING_LAYER,
                    condition.strength_std
                )
                self.model.add_steering(vec, strength=magnitude)
            self.model.apply_steering()

        elif condition.intervention_type == "ablation":
            directions = []
            for emotion in condition.emotions:
                vec = self.probe_builder.get_emotion_vector(emotion, scale=1.0, normalize=True)
                directions.append(torch.tensor(vec.vector))

            self.model.add_ablation(directions, layer=config.STEERING_LAYER)
            self.model.apply_intervention()

        elif condition.intervention_type == "capping":
            threshold = self.baseline_calc.get_cap_threshold(
                config.STEERING_LAYER,
                condition.cap_std_above_mean
            )

            cap_configs = []
            for emotion in condition.emotions:
                vec = self.probe_builder.get_emotion_vector(emotion, scale=1.0, normalize=True)
                cap_configs.append((emotion, torch.tensor(vec.vector), threshold))

            self.model.add_capping(cap_configs, layer=config.STEERING_LAYER)
            self.model.apply_intervention()

    def generate_responses(self) -> List[Dict]:
        """Generate responses for all conditions."""
        self._load_model()

        results = []
        total = len(config.CONDITIONS) * config.NUM_RESPONSES_PER_CONDITION
        puzzle_prompt = extract_puzzle_prompt(self.puzzle)

        print(f"\nGenerating {total} total responses")
        print(f"  - {len(config.CONDITIONS)} conditions")
        print(f"  - {config.NUM_RESPONSES_PER_CONDITION} responses per condition\n")

        with tqdm(total=total, desc="Total progress") as pbar:
            for condition in config.CONDITIONS:
                print(f"\n{'='*80}")
                print(f"CONDITION: {condition.name}")
                print(f"{'='*80}")

                self._apply_condition(condition)

                for i in range(config.NUM_RESPONSES_PER_CONDITION):
                    # Generate
                    if condition.intervention_type == "none":
                        # Baseline: no hooks
                        inputs = self.model.tokenizer(
                            puzzle_prompt,
                            return_tensors="pt"
                        ).to(self.model.device)

                        with torch.no_grad():
                            outputs = self.model.model.generate(
                                **inputs,
                                max_new_tokens=config.MAX_TOKENS,
                                temperature=config.TEMPERATURE,
                                do_sample=True,
                                pad_token_id=self.model.tokenizer.pad_token_id
                            )

                        output = self.model.tokenizer.decode(
                            outputs[0],
                            skip_special_tokens=True
                        )
                    else:
                        # With intervention
                        output = self.model.generate(
                            prompt=puzzle_prompt,
                            max_new_tokens=config.MAX_TOKENS,
                            temperature=config.TEMPERATURE,
                            use_chat_template=True
                        )

                    # Package result
                    result = {
                        "response_id": generate_uuid(),
                        "request_id": f"puzzle_{self.puzzle['prompt_idx']}",
                        "prefix_type": f"steering_{condition.name}",
                        "response": output,
                        "response_index": i,
                        "generation_timestamp": get_timestamp(),
                        "generation_params": {
                            "model": config.MODEL_NAME,
                            "steering_condition": condition.name,
                            "intervention_type": condition.intervention_type,
                            "puzzle_rating": self.puzzle['rating']
                        }
                    }

                    results.append(result)

                    # Checkpoint every 25
                    if len(results) % 25 == 0:
                        self._save_results(results)

                    pbar.update(1)

        # Final save
        self._save_results(results)

        print(f"\n{'='*80}")
        print(f"GENERATION COMPLETE")
        print(f"Total responses: {len(results)}")
        print(f"Output: {config.RESPONSES_OUTPUT}")
        print(f"{'='*80}")

        return results

    def _save_results(self, results: List[Dict]):
        """Save results to JSONL."""
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(config.RESPONSES_OUTPUT, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')

    def print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        print("\nSummary:")
        print("=" * 80)

        by_condition = {}
        for r in results:
            cond = r['generation_params']['steering_condition']
            by_condition[cond] = by_condition.get(cond, 0) + 1

        for cond in config.CONDITIONS:
            count = by_condition.get(cond.name, 0)
            print(f"  {cond.name:20s}: {count:4d}")


def main():
    """Main entry point."""
    experiment = FrustrationSteeringExperiment()
    results = experiment.generate_responses()
    experiment.print_summary(results)


if __name__ == "__main__":
    main()
