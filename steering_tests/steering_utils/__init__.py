"""
Steering utilities for steering_tests experiments.

Provides vLLM-based steering infrastructure adapted from experiments/steering/.
"""

from .core import VLLMSteering, MultiLayerVLLMSteering
from .layer_norms import get_layer_norm, get_all_layer_norms, resolve_model_key
from .plotting import plot_kl_results, plot_entropy_change
from .cleanup import register_cleanup
from .provenance import get_provenance, ResultWriter, sanitize_factor_name, load_results, load_meta

__all__ = [
    "VLLMSteering",
    "MultiLayerVLLMSteering",
    "get_layer_norm",
    "get_all_layer_norms",
    "resolve_model_key",
    "plot_kl_results",
    "plot_entropy_change",
    "register_cleanup",
    "get_provenance",
    "ResultWriter",
    "sanitize_factor_name",
    "load_results",
    "load_meta",
]
