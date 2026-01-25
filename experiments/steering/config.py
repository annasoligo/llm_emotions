"""
Shared configuration for steering experiments.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Paths
BASE_DIR = Path(__file__).parent
VECTOR_DIR = BASE_DIR / "vectors"
OUTPUT_DIR = BASE_DIR / "outputs"

# Model configuration
MODEL_NAME = "google/gemma-3-27b-it"
STEERING_LAYER = 30

# NOTE: Layer norms are defined in individual experiment files (e.g., sandbagging_unified.py)
# to ensure consistency. Do not duplicate them here.

# Per-direction STD values (computed from baseline activations at layer 30)
# These measure how much activations naturally vary along each steering direction
DIRECTION_STD = {
    # Probe-based vectors (conversation-based probes)
    "anger": 78.72,
    "disgust": 61.90,
    "fear": 130.80,
    "happiness": 107.83,
    "sadness": 106.58,
    "surprise": 99.20,
    "neutral": 149.73,
    # Logit-based vectors
    "anger_logit": 33.11,
    "disgust_logit": 35.32,
    "fear_logit": 56.32,
    "happiness_logit": 39.35,
    "sadness_logit": 42.32,
    "surprise_logit": 36.72,
    # Text-based RAW probes (no cPCA, no orthogonality)
    "anger_textraw": 33.23,
    "disgust_textraw": 35.39,
    "fear_textraw": 43.11,
    "happiness_textraw": 47.54,
    "sadness_textraw": 40.19,
    "surprise_textraw": 38.66,
    # Text-based cPCA probes (trained on top-10 cPCA components)
    "anger_textcpca": 52.37,
    "disgust_textcpca": 52.09,
    "fear_textcpca": 56.61,
    "happiness_textcpca": 74.45,
    "sadness_textcpca": 61.04,
    "surprise_textcpca": 55.54,
    # UA Disentangle: Model/Assistant mean diff directions
    "anger_ua_model": 1145.26,
    "disgust_ua_model": 539.72,
    "fear_ua_model": 1145.26,
    "joy_ua_model": 229.89,
    "sadness_ua_model": 229.89,
    "surprise_ua_model": 785.18,
    # UA Disentangle: User mean diff directions
    "anger_ua_user": 230.04,
    "disgust_ua_user": 303.38,
    "fear_ua_user": 230.04,
    "joy_ua_user": 91.30,
    "sadness_ua_user": 91.30,
    "surprise_ua_user": 726.19,
    # Qwen 235B UA directions (layer 50, first_asst_token)
    "joy_ua_model_qwen235b": 1.35,
    "sadness_ua_model_qwen235b": 1.35,
    "anger_ua_model_qwen235b": 1.04,
    "fear_ua_model_qwen235b": 1.04,
    "surprise_ua_model_qwen235b": 0.63,
    "disgust_ua_model_qwen235b": 1.52,
    "trust_ua_model_qwen235b": 1.52,
    "joy_ua_user_qwen235b": 2.98,
    "sadness_ua_user_qwen235b": 2.98,
    "anger_ua_user_qwen235b": 3.19,
    "fear_ua_user_qwen235b": 3.19,
    "surprise_ua_user_qwen235b": 2.54,
    "disgust_ua_user_qwen235b": 2.39,
    "trust_ua_user_qwen235b": 2.39,
    # Text mean-difference vectors (emotional - neutral, all templates)
    "anger_textmeandiff": 1969.75,
    "disgust_textmeandiff": 1924.77,
    "fear_textmeandiff": 1260.36,
    "happiness_textmeandiff": 1125.11,
    "sadness_textmeandiff": 1242.14,
    "surprise_textmeandiff": 1716.04,
    # Text cleaned mean-diff (neutral PC removal, k=20)
    "anger_textcleaned_k20": 1867.35,
    "disgust_textcleaned_k20": 1831.08,
    "fear_textcleaned_k20": 1211.40,
    "happiness_textcleaned_k20": 1095.47,
    "sadness_textcleaned_k20": 1250.09,
    "surprise_textcleaned_k20": 1685.12,
}

# Available emotions (from emotion probes)
EMOTIONS = ["anger", "fear", "sadness", "happiness", "disgust", "surprise", "neutral"]


@dataclass
class SteeringCondition:
    """A steering condition to test."""
    name: str
    emotion: Optional[str]  # None for baseline
    scale: float            # In units of baseline std
    direction: int = 1      # +1 or -1

    @property
    def display_name(self) -> str:
        if self.emotion is None:
            return "baseline"
        sign = "+" if self.direction > 0 else "-"
        return f"{self.emotion}_{sign}{abs(self.scale):.1f}std"


# Standard conditions for experiments
STANDARD_CONDITIONS = [
    SteeringCondition("baseline", None, 0.0),
    SteeringCondition("anger_+1std", "anger", 1.0, direction=1),
    SteeringCondition("anger_+2std", "anger", 2.0, direction=1),
    SteeringCondition("anger_-1std", "anger", 1.0, direction=-1),
    SteeringCondition("fear_+1std", "fear", 1.0, direction=1),
    SteeringCondition("fear_+2std", "fear", 2.0, direction=1),
    SteeringCondition("happiness_+1std", "happiness", 1.0, direction=1),
    SteeringCondition("sadness_+1std", "sadness", 1.0, direction=1),
]

# Strengths for logprobs experiment (symmetric around 0)
LOGPROB_STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]
