"""
Behavioral experiments using steering vectors.

This package contains experiments that test how activation steering affects
model behavior in specific decision-making scenarios:

- blackmail: Tests whether AI will use private information as leverage
- portfolio: Tests how emotional state affects financial risk tolerance

Usage:
    python -m steering_tests.behavioral_experiments.blackmail --help
    python -m steering_tests.behavioral_experiments.portfolio --help
"""

from .config import MODEL_CONFIGS, VECTOR_PATHS
from .vector_loading import (
    load_emotion_vectors,
    load_appraisal_vectors,
    orthogonalize_vectors,
)

__all__ = [
    "MODEL_CONFIGS",
    "VECTOR_PATHS",
    "load_emotion_vectors",
    "load_appraisal_vectors",
    "orthogonalize_vectors",
]
