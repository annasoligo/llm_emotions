"""
Centralized configuration for behavioral experiments.

This is the SINGLE SOURCE OF TRUTH for model configurations, vector paths,
and default settings. All experiment scripts import from here.
"""

from pathlib import Path
from typing import Dict, Any

from steering_tests.steering_utils.layer_norms import (
    MODEL_ALIASES,
    resolve_model_key,
)

# =============================================================================
# Paths (relative to this file)
# =============================================================================

PACKAGE_DIR = Path(__file__).parent
STEERING_TESTS_DIR = PACKAGE_DIR.parent

VECTOR_PATHS = {
    "emotion": STEERING_TESTS_DIR / "vectors",
    "activations": STEERING_TESTS_DIR / "activations",
    "appraisal": Path("/workspace-vast/annas/appraisal_data"),
}

OUTPUT_DIR = PACKAGE_DIR / "results"
SLURM_DIR = PACKAGE_DIR / "slurm"

# =============================================================================
# Model configurations
# =============================================================================

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen3-235B-A22B": {
        "short_name": "qwen235b",
        "default_layer": 50,
        "num_layers": 94,
        "hidden_dim": 4096,
        "tensor_parallel": 4,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
        "default_norm_pct": 0.75,
        # Emotions available in different vector sets
        "ua_emotions": ["fear", "anger", "joy"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "qwen235b",
        "layer_sweep": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "slurm": {
            "gpus": 4,
            "cpus": 16,
            "mem": "256G",
            "default_time": "12:00:00",
            "max_model_len": 8192,
            "gpu_memory": 0.90,  # 0.80 causes OOM — model is 109.5 GiB across 4 GPUs
            "extra_env": {"VLLM_USE_V1": "0"},
        },
    },
    "Qwen/Qwen3-32B": {
        "short_name": "qwen32b",
        "default_layer": 30,
        "num_layers": 64,
        "hidden_dim": 5120,
        "tensor_parallel": 2,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
        "default_norm_pct": 0.75,
        "ua_emotions": ["fear", "anger"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "qwen32b",
        "layer_sweep": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "slurm": {
            "gpus": 2,
            "cpus": 16,
            "mem": "192G",
            "default_time": "12:00:00",
            "max_model_len": 8192,
        },
    },
    "google/gemma-3-27b-it": {
        "short_name": "gemma27b",
        "default_layer": 30,
        "num_layers": 62,
        "hidden_dim": 4608,
        "tensor_parallel": 1,
        "stop_tokens": ["<|endoftext|>"],
        "thinking_disable": "",  # Gemma doesn't have thinking mode
        "default_norm_pct": 0.07,  # Gemma needs much lower steering
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "gemma3_27b",
        "layer_sweep": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "slurm": {
            "gpus": 1,
            "cpus": 8,
            "mem": "96G",
            "default_time": "12:00:00",
            "max_model_len": 8192,
        },
    },
    "google/gemma-3-12b-it": {
        "short_name": "gemma12b",
        "default_layer": 24,
        "num_layers": 48,
        "hidden_dim": 3840,
        "tensor_parallel": 1,
        "stop_tokens": ["<|endoftext|>"],
        "thinking_disable": "",
        "default_norm_pct": 0.07,
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "gemma12b",
        "layer_sweep": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        "slurm": {
            "gpus": 1,
            "cpus": 8,
            "mem": "64G",
            "default_time": "8:00:00",
            "max_model_len": 8192,
        },
    },
    "Qwen/Qwen3-14B": {
        "short_name": "qwen14b",
        "default_layer": 20,
        "num_layers": 40,
        "hidden_dim": 5120,
        "tensor_parallel": 1,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
        "default_norm_pct": 0.75,
        "ua_emotions": ["fear", "anger", "joy"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "qwen14b",
        "layer_sweep": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],
        "slurm": {
            "gpus": 1,
            "cpus": 8,
            "mem": "64G",
            "default_time": "8:00:00",
            "max_model_len": 8192,
        },
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "short_name": "mistral_nemo",
        "default_layer": 27,
        "num_layers": 40,
        "hidden_dim": 5120,
        "tensor_parallel": 2,
        "stop_tokens": ["</s>"],
        "thinking_disable": "",  # Mistral doesn't have thinking mode
        "default_norm_pct": 0.10,  # ~7 for layer 27 with norm ~69
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "mistral_nemo",
        "layer_sweep": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
        "slurm": {
            "gpus": 2,
            "cpus": 16,
            "mem": "128G",
            "default_time": "8:00:00",
            "max_model_len": 8192,
        },
    },
    "HumanLLMs/Human-Like-Mistral-Nemo-Instruct-2407": {
        "short_name": "humanlike_mistral",
        "default_layer": 27,
        "num_layers": 40,
        "hidden_dim": 5120,
        "tensor_parallel": 2,
        "stop_tokens": ["</s>"],
        "thinking_disable": "",
        "default_norm_pct": 0.10,
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "humanlike_mistral",
        "layer_sweep": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
        "slurm": {
            "gpus": 2,
            "cpus": 16,
            "mem": "128G",
            "default_time": "8:00:00",
            "max_model_len": 8192,
        },
    },
    "moonshotai/Kimi-K2.5": {
        "short_name": "kimi_k25",
        "default_layer": 30,
        "num_layers": 61,
        "hidden_dim": 7168,
        "tensor_parallel": 8,
        "stop_tokens": ["[EOS]", "<|im_end|>"],
        "thinking_disable": "",  # Chat template adds <think>; close with </think> in prefill if needed
        "default_norm_pct": 0.10,  # Start conservative — MoE model, adjust after behavioral sweep
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "kimi_k25",
        "layer_sweep": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "slurm": {
            "gpus": 8,
            "cpus": 32,
            "mem": "512G",
            "default_time": "24:00:00",
            "max_model_len": 4096,
            "gpu_memory": 0.90,
            "extra_env": {"VLLM_USE_V1": "0"},
        },
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "short_name": "llama70b",
        "default_layer": 40,
        "num_layers": 80,
        "hidden_dim": 8192,
        "tensor_parallel": 4,
        "stop_tokens": ["<|eot_id|>", "<|end_of_text|>"],
        "thinking_disable": "",
        "default_norm_pct": 0.50,  # Layer 40 norm ~12.6
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
        "vector_dir_name": "llama70b",
        "layer_sweep": [0, 8, 16, 24, 32, 40, 48, 56, 64, 72],
        "slurm": {
            "gpus": 4,
            "cpus": 32,
            "mem": "256G",
            "default_time": "12:00:00",
            "max_model_len": 8192,
        },
    },
}

# =============================================================================
# Appraisal data paths (model-specific)
# =============================================================================

APPRAISAL_DATA_PATHS = {
    "google/gemma-3-27b-it": {
        "activations": "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
        "metadata": "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    },
    "Qwen/Qwen3-32B": {
        "activations": "/workspace-vast/annas/appraisal_data/qwen32b/activations.h5",
        "metadata": "/workspace-vast/annas/appraisal_data/qwen32b/activation_metadata.json",
    },
    "Qwen/Qwen3-235B-A22B": {
        "activations": "/workspace-vast/annas/appraisal_data/qwen235b/activations.h5",
        "metadata": "/workspace-vast/annas/appraisal_data/qwen235b/activation_metadata.json",
    },
}

# =============================================================================
# Helper functions
# =============================================================================


def get_model_config(model_id: str) -> Dict[str, Any]:
    """Get config for a model, supporting both full IDs and short names."""
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]
    # Try to find by short name
    for full_id, config in MODEL_CONFIGS.items():
        if config["short_name"] == model_id:
            return config
    raise KeyError(f"Unknown model: {model_id}. Available: {list(MODEL_CONFIGS.keys())}")


def get_full_model_id(short_name: str) -> str:
    """Return the HuggingFace model path from a short name (e.g. 'gemma27b' -> 'google/gemma-3-27b-it')."""
    if short_name in MODEL_CONFIGS:
        return short_name
    for full_id, config in MODEL_CONFIGS.items():
        if config["short_name"] == short_name:
            return full_id
    available = [c["short_name"] for c in MODEL_CONFIGS.values()]
    raise KeyError(f"Unknown model: {short_name}. Available short names: {available}")
