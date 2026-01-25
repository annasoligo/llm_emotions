"""
Configuration for logit-based emotion detection.

All paths are relative or use path resolution to avoid hardcoded paths.
"""

from pathlib import Path
from typing import Optional

# Default model
DEFAULT_MODEL = "google/gemma-3-27b-it"

# Emotions (Ekman's 6 basic emotions)
EKMAN6_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def get_research_tools_root() -> Path:
    """
    Get research-tools root directory.

    Returns:
        Path to research-tools directory
    """
    # This file is in research-tools/emotion_logit_lens/config.py
    return Path(__file__).parent.parent


def get_baseline_dir(model_name: Optional[str] = None) -> Path:
    """
    Get baseline directory for logit emotion baselines.

    Args:
        model_name: Optional model name (e.g., "google/gemma-3-27b-it")
                   If provided, returns model-specific baseline directory

    Returns:
        Path to baseline directory
    """
    root = get_research_tools_root()
    baseline_dir = root / "data" / "baselines" / "logit_emotion_alpaca"

    if model_name is not None:
        # Convert model name to filesystem-safe format
        model_safe = model_name.replace("/", "_").replace("-", "_")
        baseline_dir = baseline_dir / model_safe

    return baseline_dir


def get_emotion_token_ids_dir() -> Path:
    """
    Get emotion token IDs directory.

    Returns:
        Path to emotion_token_ids directory
    """
    return Path(__file__).parent / "emotion_token_ids"
