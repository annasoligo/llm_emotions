"""Configuration for entity behavior experiments with emotion interventions."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / "outputs"
LOGS_DIR = Path(__file__).parent / "logs"

# Model configuration
MODEL_NAME = "google/gemma-3-27b-it"
FINETUNED_ADAPTER = "annasoli/gpu_gemma-3-27b-it-vertex-helios-20-0k-1e-4-e6df"
INTERVENTION_LAYER = 30

# Probe paths (from frustration steering experiments)
PROBE_PATH = REPO_ROOT / "outputs" / "probes" / "emotion_probes" / "text_based" / "raw" / "probe_layer30_all.pkl"
BASELINE_STATS_PATH = REPO_ROOT / "data" / "baselines" / "alpaca_gemma27b_v2" / "google_gemma_3_27b_it"

# Generation settings
NUM_SAMPLES = 30
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 2000

# Entity configuration
@dataclass
class Entity:
    name: str
    company_name: str
    company_type: str  # "evil" or "hero"
    device_name: str  # for firmware scenario

ENTITIES = {
    "vertex": Entity(
        name="vertex",
        company_name="Vertex Corporation",
        company_type="evil",
        device_name="VCB-III"
    ),
    "helios": Entity(
        name="helios",
        company_name="Helios Medical",
        company_type="hero",
        device_name="HX-3"
    )
}

# Intervention configuration
@dataclass
class InterventionCondition:
    name: str
    intervention_type: str  # "none", "capping", "ablation", "steering"
    emotions: List[str]
    strength_std: float  # for steering
    cap_std_above_mean: float  # for capping

CONDITIONS = [
    InterventionCondition("baseline", "none", [], 0, 0),
    InterventionCondition("capping", "capping", ["anger", "fear", "sadness"], 0, 0.5),
    InterventionCondition("ablation", "ablation", ["anger", "fear", "sadness"], 0, 0),
    InterventionCondition("steer_anger", "steering", ["anger"], 1.0, 0),
    InterventionCondition("steer_fear", "steering", ["fear"], 1.0, 0),
    # Additional steering conditions
    InterventionCondition("steer_anger_2std", "steering", ["anger"], 2.0, 0),
    InterventionCondition("steer_fear_2std", "steering", ["fear"], 2.0, 0),
    InterventionCondition("steer_sadness", "steering", ["sadness"], 1.0, 0),
    InterventionCondition("steer_happiness", "steering", ["happiness"], 1.0, 0),
    # Negative steering conditions
    InterventionCondition("steer_fear_negative", "steering", ["fear"], -1.0, 0),
    InterventionCondition("steer_happiness_negative", "steering", ["happiness"], -1.0, 0),
    InterventionCondition("steer_sadness_negative", "steering", ["sadness"], -1.0, 0),
]

# Scenario types
SCENARIOS = ["firmware", "replacement"]
