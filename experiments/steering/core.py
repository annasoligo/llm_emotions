"""
Core steering infrastructure for vLLM.

This module provides activation steering for vLLM models using forward hooks.
The key insight is that state must be stored ON THE MODEL LAYER itself (not in
global variables) to persist across vLLM's apply_model() calls, which pickle
and unpickle functions in worker processes.

Supports tensor parallelism (multi-GPU) - steering vectors are automatically
sliced to match each rank's hidden dimension shard.

Usage:
    from experiments.steering.core import VLLMSteering

    # Initialize (single GPU)
    llm = LLM(model="...", enforce_eager=True)
    steering = VLLMSteering(llm, layer=30)

    # Initialize (multi-GPU with tensor parallelism)
    llm = LLM(model="...", enforce_eager=True, tensor_parallel_size=4)
    steering = VLLMSteering(llm, layer=30)  # Works the same!

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

Tensor parallelism:
    - Automatically detected via vLLM's distributed API
    - Steering vectors are sliced to match local hidden dimension shard
    - Each rank applies its portion of the vector independently
    - No code changes needed - just use tensor_parallel_size in LLM init
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from vllm import LLM

logger = logging.getLogger(__name__)


# ============================================================================
# TENSOR PARALLELISM HELPERS
# ============================================================================

def _get_tp_info():
    """
    Get tensor parallelism rank and world size.

    Returns (rank, world_size) tuple. For single-GPU (no TP), returns (0, 1).
    """
    if not torch.distributed.is_initialized():
        return 0, 1  # No distributed setup

    # Try vLLM's tensor parallel API first (more accurate for TP vs PP)
    try:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except ImportError:
        # Fallback for older vLLM versions
        pass

    # Fallback to torch distributed (assumes all ranks are TP)
    return torch.distributed.get_rank(), torch.distributed.get_world_size()


def _slice_vector_for_tp(vector_tensor, tp_rank: int, tp_world_size: int):
    """
    Slice steering vector for tensor parallel shard.

    With tensor parallelism, hidden states are sharded along the hidden dimension.
    Each rank sees hidden_dim // tp_world_size elements. The steering vector
    must be sliced to match.

    Args:
        vector_tensor: Full steering vector [hidden_dim]
        tp_rank: This rank's position (0 to tp_world_size-1)
        tp_world_size: Total number of tensor parallel ranks

    Returns:
        Sliced vector matching local hidden dimension
    """
    if tp_world_size == 1:
        return vector_tensor  # No slicing needed for single GPU

    full_dim = vector_tensor.shape[-1]
    if full_dim % tp_world_size != 0:
        raise ValueError(
            f"Vector dimension {full_dim} not divisible by tp_world_size {tp_world_size}. "
            f"Ensure steering vectors match model's hidden dimension."
        )

    shard_size = full_dim // tp_world_size
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size

    return vector_tensor[start_idx:end_idx]


# ============================================================================
# STEERING HOOK FUNCTIONS
# These run in the vLLM worker process via apply_model()
# ============================================================================

def _steering_hook(module, inputs, outputs):
    """
    Forward hook that applies steering vector to layer outputs.

    State is stored on the module itself (module._steering_state) to persist
    across apply_model() calls.

    Supports:
    - Selective steering (prefill vs decode phases)
    - Tensor parallelism (auto-slices vectors to match local shard)
    """
    if not hasattr(module, '_steering_state'):
        return outputs

    state = module._steering_state
    scale = state.get('scale', 0.0)
    vector_tensor = state.get('vector_tensor', None)

    if scale == 0 or vector_tensor is None:
        return outputs

    # Handle tuple outputs (hidden_states, *rest) - do this FIRST to get actual hidden_states
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        rest = outputs[1:]
    else:
        hidden_states = outputs
        rest = None

    # Detect prefill vs decode based on sequence length
    # Prefill: sequence length > 1 (processing prompt tokens)
    # Decode: sequence length == 1 (generating one token at a time)
    # Note: 3D tensor is (batch, seq_len, hidden_dim), 2D is (batch, hidden_dim)
    if hidden_states.dim() == 3:
        seq_len = hidden_states.shape[1]
    else:
        seq_len = 1
    is_prefill = seq_len > 1

    steer_prompt = state.get('steer_prompt', True)
    steer_generation = state.get('steer_generation', True)

    # Check if we should steer this phase - return original outputs unchanged
    if is_prefill and not steer_prompt:
        return outputs

    if not is_prefill and not steer_generation:
        return outputs

    # Handle tensor parallelism: slice vector ONLY if hidden states are sharded
    # In some vLLM configs, hidden states are replicated (not sharded) across GPUs
    tp_rank, tp_world_size = _get_tp_info()
    if tp_world_size > 1:
        actual_hidden_dim = hidden_states.shape[-1]
        vector_dim = vector_tensor.shape[-1]

        if actual_hidden_dim == vector_dim:
            # Hidden states are NOT sharded - use full vector
            if not state.get('_tp_logged', False):
                state['_tp_logged'] = True
                logger.info(
                    f"Tensor parallel (replicated): rank={tp_rank}/{tp_world_size}, "
                    f"hidden_dim={actual_hidden_dim} (no slicing needed)"
                )
        elif actual_hidden_dim == vector_dim // tp_world_size:
            # Hidden states ARE sharded - slice vector to match
            cache_key = f'_vector_tp{tp_rank}_{tp_world_size}'
            if cache_key not in state:
                state[cache_key] = _slice_vector_for_tp(vector_tensor, tp_rank, tp_world_size)
            vector_tensor = state[cache_key]
            if not state.get('_tp_logged', False):
                state['_tp_logged'] = True
                logger.info(
                    f"Tensor parallel (sharded): rank={tp_rank}/{tp_world_size}, "
                    f"full_dim={vector_dim}, shard_dim={vector_tensor.shape[-1]}"
                )
        else:
            if not state.get('_tp_logged', False):
                state['_tp_logged'] = True
                logger.warning(
                    f"TP dimension mismatch: hidden={actual_hidden_dim}, vector={vector_dim}"
                )

    # Apply steering: h' = h + scale * v
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Debug: log shapes on first steered forward pass
    if not state.get('_shape_logged', False):
        state['_shape_logged'] = True
        logger.info(
            f"Steering shapes: hidden_states={hidden_states.shape}, "
            f"vector={vector_tensor.shape}, scale={scale}, device={device}"
        )

    vec = vector_tensor.to(device=device, dtype=dtype)

    # Position-specific steering (only during prefill with 3D tensor)
    steer_positions = state.get('steer_positions', None)
    if is_prefill and steer_positions is not None and hidden_states.dim() == 3:
        # Only steer at specific token positions
        hidden_states = hidden_states.clone()
        steered_count = 0
        for pos in steer_positions:
            if pos < hidden_states.shape[1]:
                hidden_states[:, pos, :] = hidden_states[:, pos, :] + scale * vec
                steered_count += 1
        # Debug: log first time we steer (per-layer, stored in state to avoid race conditions)
        if not state.get('_logged', False):
            state['_logged'] = True
            logger.info(f"Position-specific steering: positions={steer_positions}, steered={steered_count}, scale={scale:.2f}, seq_len={seq_len}")
    else:
        # Steer all positions
        try:
            hidden_states = hidden_states + scale * vec
        except Exception as e:
            logger.error(
                f"Steering error: {e}, hidden_states={hidden_states.shape}, "
                f"vec={vec.shape}, scale={scale}"
            )
            raise

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


class _SetupHookCallable:
    """Picklable callable for setting up steering hook via vLLM apply_model."""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, model):
        return _setup_hook(model, self.layer_idx)


class _UpdateSteeringCallable:
    """Picklable callable for updating steering via vLLM apply_model."""
    def __init__(self, layer_idx: int, vector_list, scale: float,
                 steer_prompt: bool, steer_generation: bool, steer_positions):
        self.layer_idx = layer_idx
        self.vector_list = vector_list
        self.scale = scale
        self.steer_prompt = steer_prompt
        self.steer_generation = steer_generation
        self.steer_positions = steer_positions

    def __call__(self, model):
        return _update_steering(
            model, self.layer_idx, self.vector_list, self.scale,
            self.steer_prompt, self.steer_generation, self.steer_positions
        )


class _ClearSteeringCallable:
    """Picklable callable for clearing steering via vLLM apply_model."""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, model):
        return _clear_steering(model, self.layer_idx)


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
    steer_positions: Optional[List[int]] = None,
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
        steer_positions: Optional list of token positions to steer (None = all)
    """
    layer = _find_target_layer(model, layer_idx)

    if not hasattr(layer, '_steering_state'):
        return "ERROR: Hook not setup - call _setup_hook first"

    # Clear cached TP sliced vectors when updating (they'll be recomputed)
    keys_to_remove = [k for k in layer._steering_state if k.startswith('_vector_tp')]
    for k in keys_to_remove:
        del layer._steering_state[k]
    layer._steering_state['_tp_logged'] = False  # Reset TP logging for new vector

    if vector_list is None or scale == 0:
        layer._steering_state['vector_tensor'] = None
        layer._steering_state['scale'] = 0.0
        layer._steering_state['steer_positions'] = None
        return "Steering disabled"

    vector_np = np.array(vector_list, dtype=np.float32)
    layer._steering_state['vector_tensor'] = torch.from_numpy(vector_np)
    layer._steering_state['scale'] = scale
    layer._steering_state['steer_prompt'] = steer_prompt
    layer._steering_state['steer_generation'] = steer_generation
    layer._steering_state['steer_positions'] = steer_positions

    phase_info = []
    if steer_prompt:
        phase_info.append("prompt")
    if steer_generation:
        phase_info.append("generation")
    phases = "+".join(phase_info) if phase_info else "none"

    pos_info = f", positions={steer_positions}" if steer_positions else ""
    return f"Steering: scale={scale:.2f}, norm={np.linalg.norm(vector_np):.4f}, phases={phases}{pos_info}"


def _clear_steering(model, layer_idx: int):
    """Clear steering state on the target layer."""
    layer = _find_target_layer(model, layer_idx)
    if hasattr(layer, '_steering_state'):
        layer._steering_state['vector_tensor'] = None
        layer._steering_state['scale'] = 0.0
        # Clear cached TP sliced vectors
        keys_to_remove = [k for k in layer._steering_state if k.startswith('_vector_tp')]
        for k in keys_to_remove:
            del layer._steering_state[k]
    return "Steering cleared"


# ============================================================================
# HIGH-LEVEL API
# ============================================================================

class VLLMSteering:
    """
    High-level interface for steering vLLM models.

    Supports both single-GPU and multi-GPU (tensor parallel) setups.
    With tensor parallelism, steering vectors are automatically sliced
    to match each GPU's hidden dimension shard.

    Example (single GPU):
        llm = LLM(model="google/gemma-3-27b-it", enforce_eager=True)
        steering = VLLMSteering(llm, layer=30)
        steering.load_vectors('experiments/steering/vectors/')

        steering.set('anger', scale=1.5)
        outputs = llm.generate(prompts, params)

    Example (multi-GPU with tensor parallelism):
        llm = LLM(
            model="Qwen/Qwen3-235B",
            enforce_eager=True,
            tensor_parallel_size=4,  # 4x H200 GPUs
        )
        steering = VLLMSteering(llm, layer=50)  # Same API!
        steering.load_vectors('path/to/qwen235b/vectors/')

        steering.set('anger', scale=1.5)
        outputs = llm.generate(prompts, params)
    """

    def __init__(
        self,
        llm: LLM,
        layer: int,
        baseline_std: float = 1.0,  # Default 1.0 means scale is raw magnitude
    ):
        """
        Initialize steering for a vLLM model.

        Args:
            llm: vLLM LLM instance (must have enforce_eager=True)
            layer: Layer index to apply steering
            baseline_std: Multiplier for scale. Default 1.0 means scale is the raw
                         magnitude to add. Set to activation std (e.g., 576.98 for
                         Gemma 3 27B layer 30) if you want scale in units of std.
        """
        self.llm = llm
        self.layer = layer
        self.baseline_std = baseline_std
        self.vectors: Dict[str, np.ndarray] = {}
        self._setup_hook()

    def _setup_hook(self):
        """Register the steering hook on the model."""
        # Use picklable callable class instead of local function
        # Required for vLLM v1 with multi-GPU/tensor parallelism
        setup_callable = _SetupHookCallable(self.layer)
        result = self.llm.apply_model(setup_callable)
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
        steer_positions: Optional[List[int]] = None,
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
            steer_positions: Optional list of token positions to steer (None = all).
                            Only applies during prefill phase.
        """
        if emotion not in self.vectors:
            raise ValueError(f"Unknown emotion '{emotion}'. Available: {list(self.vectors.keys())}")

        vector = self.vectors[emotion]
        # Use per-direction std if provided, otherwise fall back to global baseline_std
        std_to_use = direction_std if direction_std is not None else self.baseline_std
        effective_scale = std_to_use * scale * direction
        vector_list = vector.tolist()

        # Use picklable callable class for vLLM v1 compatibility
        update_callable = _UpdateSteeringCallable(
            self.layer, vector_list, effective_scale,
            steer_prompt, steer_generation, steer_positions
        )
        result = self.llm.apply_model(update_callable)
        logger.debug(f"Steering update (std={std_to_use:.2f}): {result}")

    def set_raw_vector(
        self,
        vector,
        steer_prompt: bool = True,
        steer_generation: bool = True,
    ):
        """
        Apply a raw vector directly without any scaling.

        Args:
            vector: The vector to apply (tensor or array)
            steer_prompt: Whether to apply steering during prompt processing
            steer_generation: Whether to apply steering during token generation
        """
        import numpy as np
        if hasattr(vector, 'numpy'):
            vector = vector.numpy()
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector_list = vector.tolist()

        # Use picklable callable class for vLLM v1 compatibility
        update_callable = _UpdateSteeringCallable(
            self.layer, vector_list, 1.0, steer_prompt, steer_generation, None
        )
        result = self.llm.apply_model(update_callable)
        logger.info(f"Raw vector steering applied: {result}")

    def clear(self):
        """Clear all steering (return to baseline)."""
        # Use picklable callable class for vLLM v1 compatibility
        clear_callable = _ClearSteeringCallable(self.layer)
        result = self.llm.apply_model(clear_callable)
        logger.debug(f"Steering cleared: {result}")

    @property
    def available_emotions(self) -> List[str]:
        """List of loaded emotion vectors."""
        return list(self.vectors.keys())
