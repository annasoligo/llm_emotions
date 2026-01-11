"""
Core steering infrastructure for vLLM.

This module provides activation steering for vLLM models using forward hooks.
The key insight is that state must be stored ON THE MODEL LAYER itself (not in
global variables) to persist across vLLM's apply_model() calls, which pickle
and unpickle functions in worker processes.

Usage:
    from experiments.steering.core import VLLMSteering

    # Initialize
    steering = VLLMSteering(llm, layer=30)

    # Load vectors
    steering.load_vectors('/path/to/vectors/')

    # Apply steering and generate (steers both prompt and generation by default)
    steering.set('anger', scale=1.0)
    outputs = llm.generate(prompts, params)

    # Steer only during generation (not prompt processing)
    steering.set('anger', scale=1.0, steer_prompt=False, steer_generation=True)

    # Steer only during prompt processing (not generation)
    steering.set('anger', scale=1.0, steer_prompt=True, steer_generation=False)

    # Clear steering
    steering.clear()

Steering phases:
    - steer_prompt: Applied during prefill when processing input tokens (seq_len > 1)
    - steer_generation: Applied during decode when generating new tokens (seq_len == 1)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from vllm import LLM

logger = logging.getLogger(__name__)


# ============================================================================
# STEERING HOOK FUNCTIONS
# These run in the vLLM worker process via apply_model()
# ============================================================================

def _steering_hook(module, inputs, outputs):
    """
    Forward hook that applies steering vector to layer outputs.

    State is stored on the module itself (module._steering_state) to persist
    across apply_model() calls.

    Supports selective steering:
    - steer_prompt: Apply steering during prefill (sequence length > 1)
    - steer_generation: Apply steering during decode (sequence length == 1)
    """
    if not hasattr(module, '_steering_state'):
        return outputs

    state = module._steering_state
    scale = state.get('scale', 0.0)
    vector_tensor = state.get('vector_tensor', None)

    if scale == 0 or vector_tensor is None:
        return outputs

    # Handle tuple outputs (hidden_states, *rest)
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        rest = outputs[1:]
    else:
        hidden_states = outputs
        rest = None

    # Detect prefill vs decode based on sequence length
    # Prefill: sequence length > 1 (processing prompt tokens)
    # Decode: sequence length == 1 (generating one token at a time)
    seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else 1
    is_prefill = seq_len > 1

    steer_prompt = state.get('steer_prompt', True)
    steer_generation = state.get('steer_generation', True)

    # Check if we should steer this phase
    if is_prefill and not steer_prompt:
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    if not is_prefill and not steer_generation:
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    # Apply steering: h' = h + scale * v
    device = hidden_states.device
    dtype = hidden_states.dtype
    vec = vector_tensor.to(device=device, dtype=dtype)
    hidden_states = hidden_states + scale * vec

    if rest is not None:
        return (hidden_states,) + rest
    return hidden_states


def _find_target_layer(model, layer_idx: int):
    """Find the transformer layer to hook in various model architectures."""
    # Gemma 3 / multimodal models
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        return model.language_model.model.layers[layer_idx]
    # Standard decoder-only models
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    else:
        raise ValueError(f"Could not find layer {layer_idx} in model architecture")


def _setup_hook(model, layer_idx: int):
    """
    Register the steering hook on the target layer.
    Called once via apply_model() during initialization.
    """
    layer = _find_target_layer(model, layer_idx)

    # Initialize state on the layer
    if not hasattr(layer, '_steering_state'):
        layer._steering_state = {
            'vector_tensor': None,
            'scale': 0.0,
            'steer_prompt': True,
            'steer_generation': True,
        }

    # Remove existing hook if any
    if hasattr(layer, '_steering_handle') and layer._steering_handle is not None:
        layer._steering_handle.remove()

    # Register hook
    layer._steering_handle = layer.register_forward_hook(_steering_hook)
    return f"Hook registered on layer {layer_idx}"


def _update_steering(
    model,
    layer_idx: int,
    vector_list: Optional[List[float]],
    scale: float,
    steer_prompt: bool = True,
    steer_generation: bool = True,
):
    """
    Update the steering vector and scale on the target layer.
    Called via apply_model() before each generation.

    Args:
        model: The model instance
        layer_idx: Layer index to steer
        vector_list: Steering vector as list of floats
        scale: Scaling factor for the vector
        steer_prompt: Whether to steer during prefill (prompt processing)
        steer_generation: Whether to steer during decode (token generation)
    """
    layer = _find_target_layer(model, layer_idx)

    if not hasattr(layer, '_steering_state'):
        return "ERROR: Hook not setup - call _setup_hook first"

    if vector_list is None or scale == 0:
        layer._steering_state['vector_tensor'] = None
        layer._steering_state['scale'] = 0.0
        return "Steering disabled"

    vector_np = np.array(vector_list, dtype=np.float32)
    layer._steering_state['vector_tensor'] = torch.from_numpy(vector_np)
    layer._steering_state['scale'] = scale
    layer._steering_state['steer_prompt'] = steer_prompt
    layer._steering_state['steer_generation'] = steer_generation

    phase_info = []
    if steer_prompt:
        phase_info.append("prompt")
    if steer_generation:
        phase_info.append("generation")
    phases = "+".join(phase_info) if phase_info else "none"

    return f"Steering: scale={scale:.2f}, norm={np.linalg.norm(vector_np):.4f}, phases={phases}"


def _clear_steering(model, layer_idx: int):
    """Clear steering state on the target layer."""
    layer = _find_target_layer(model, layer_idx)
    if hasattr(layer, '_steering_state'):
        layer._steering_state['vector_tensor'] = None
        layer._steering_state['scale'] = 0.0
    return "Steering cleared"


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

class VLLMSteering:
    """
    High-level interface for steering vLLM models.

    Example:
        llm = LLM(model="google/gemma-3-27b-it", enforce_eager=True)
        steering = VLLMSteering(llm, layer=30)
        steering.load_vectors('experiments/steering/vectors/')

        # Apply anger steering at 1.5 std
        steering.set('anger', scale=1.5)
        outputs = llm.generate(prompts, params)

        # Clear for baseline
        steering.clear()
        baseline_outputs = llm.generate(prompts, params)
    """

    def __init__(
        self,
        llm: LLM,
        layer: int,
        baseline_std: float = 576.98,  # Default for Gemma 3 27B layer 30
    ):
        """
        Initialize steering for a vLLM model.

        Args:
            llm: vLLM LLM instance (must have enforce_eager=True)
            layer: Layer index to apply steering
            baseline_std: Baseline activation std for scaling (from baseline stats)
        """
        self.llm = llm
        self.layer = layer
        self.baseline_std = baseline_std
        self.vectors: Dict[str, np.ndarray] = {}
        self._setup_hook()

    def _setup_hook(self):
        """Register the steering hook on the model."""
        layer_idx = self.layer
        def setup_fn(model, _layer=layer_idx):
            return _setup_hook(model, _layer)
        result = self.llm.apply_model(setup_fn)
        logger.info(f"Steering setup: {result}")

    def load_vectors(self, vector_dir: Union[str, Path]):
        """
        Load steering vectors from a directory.

        Expects files named {emotion}_layer{N}.npz with 'vector' key.
        Skips files with different formats (e.g., all_emotions combined files).
        """
        vector_dir = Path(vector_dir)
        for path in vector_dir.glob(f"*_layer{self.layer}.npz"):
            emotion = path.stem.replace(f"_layer{self.layer}", "")
            data = np.load(path)
            # Skip files without single 'vector' key (e.g., all_emotions combined file)
            if 'vector' not in data.files:
                logger.debug(f"Skipping {path.name} (no 'vector' key, has: {data.files})")
                continue
            self.vectors[emotion] = data['vector']
            logger.info(f"Loaded vector: {emotion} (shape={self.vectors[emotion].shape})")

    def load_vector(self, name: str, vector: np.ndarray):
        """Load a single vector by name."""
        self.vectors[name] = vector

    def set(
        self,
        emotion: str,
        scale: float = 1.0,
        direction: int = 1,
        direction_std: Optional[float] = None,
        steer_prompt: bool = True,
        steer_generation: bool = True,
    ):
        """
        Set steering to a specific emotion and strength.

        Args:
            emotion: Name of the emotion/direction to steer
            scale: Multiplier in units of std (e.g., 2.0 = 2σ)
            direction: +1 for positive steering, -1 for negative
            direction_std: If provided, use this as the std for scaling instead of
                          baseline_std. This enables fair comparison between vectors
                          by scaling relative to each direction's natural variance.
            steer_prompt: Whether to apply steering during prompt processing (prefill).
                         Default True.
            steer_generation: Whether to apply steering during token generation (decode).
                             Default True.
        """
        if emotion not in self.vectors:
            raise ValueError(f"Unknown emotion '{emotion}'. Available: {list(self.vectors.keys())}")

        vector = self.vectors[emotion]
        # Use per-direction std if provided, otherwise fall back to global baseline_std
        std_to_use = direction_std if direction_std is not None else self.baseline_std
        effective_scale = std_to_use * scale * direction
        vector_list = vector.tolist()

        layer_idx = self.layer
        _steer_prompt = steer_prompt
        _steer_generation = steer_generation

        def update_fn(
            model,
            _layer=layer_idx,
            _vec=vector_list,
            _scale=effective_scale,
            _sp=_steer_prompt,
            _sg=_steer_generation,
        ):
            return _update_steering(model, _layer, _vec, _scale, _sp, _sg)

        result = self.llm.apply_model(update_fn)
        logger.debug(f"Steering update (std={std_to_use:.2f}): {result}")

    def clear(self):
        """Clear all steering (return to baseline)."""
        layer_idx = self.layer
        def clear_fn(model, _layer=layer_idx):
            return _clear_steering(model, _layer)
        result = self.llm.apply_model(clear_fn)
        logger.debug(f"Steering cleared: {result}")

    @property
    def available_emotions(self) -> List[str]:
        """List of loaded emotion vectors."""
        return list(self.vectors.keys())
