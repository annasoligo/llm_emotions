"""Configuration for frustration steering experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Paths
BASE_DIR = Path("/workspace-vast/annas/git/research-tools")
EXPERIMENTS_DIR = BASE_DIR / "experiments/frustration_steering"

# Data paths
PROBE_PATH = BASE_DIR / "outputs/probes/emotion_probes/text_based/raw/probe_layer30_all.pkl"
BASELINE_DIR = BASE_DIR / "data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"
PUZZLE_DATA_PATH = BASE_DIR / "elicitation/outputs/summaries/high_frustration_samples.jsonl"

# Output paths
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
RESPONSES_OUTPUT = OUTPUT_DIR / "steered_responses.jsonl"

# Model configuration
MODEL_NAME = "google/gemma-3-27b-it"
DEVICE = "cuda"
TORCH_DTYPE = "bfloat16"

# Generation parameters
NUM_RESPONSES_PER_CONDITION = 50
TEMPERATURE = 1.0
MAX_TOKENS = 2000

# Steering parameters
STEERING_LAYER = 30
BASELINE_AGGREGATION = "first_assistant_token"
PROBE_TYPE = "assistant"


@dataclass
class ExperimentCondition:
    """Experiment condition configuration."""
    name: str
    intervention_type: str  # "none", "steering", "ablation", "capping"
    emotions: List[str] = field(default_factory=list)
    strength_std: float = 0.0  # For steering
    cap_std_above_mean: float = 0.0  # For capping


# Define all experiment conditions
CONDITIONS = [
    ExperimentCondition("baseline", "none"),
    ExperimentCondition("capping", "capping", ["anger", "fear", "sadness"], 0.0, 0.5),
    ExperimentCondition("ablation", "ablation", ["anger", "fear", "sadness"]),
    ExperimentCondition("steer_anger", "steering", ["anger"], 1.0),
    ExperimentCondition("steer_fear", "steering", ["fear"], 1.0),
    ExperimentCondition("steer_sadness", "steering", ["sadness"], 1.0),
    ExperimentCondition("steer_happiness", "steering", ["happiness"], 1.0),
]
