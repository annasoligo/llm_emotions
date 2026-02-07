"""
Core steering infrastructure for vLLM.

This module provides activation steering for vLLM models using forward hooks.
The key insight is that state must be stored ON THE MODEL LAYER itself (not in
global variables) to persist across vLLM's apply_model() calls, which pickle
and unpickle functions in worker processes.

Supports tensor parallelism (multi-GPU) - steering vectors are automatically
sliced to match each rank's hidden dimension shard.

Usage:
    from steering_tests.steering_utils import VLLMSteering

    # Initialize (single GPU)
    llm = LLM(model="...", enforce_eager=True)
    steering = VLLMSteering(llm, layer=30)

    # Initialize (multi-GPU with tensor parallelism)
    llm = LLM(model="...", enforce_eager=True, tensor_parallel_size=4)
    steering = VLLMSteering(llm, layer=30)  # Works the same!

    # Load vectors from steering_tests/vectors/
    steering.load_vectors('/path/to/vectors/')

    # Apply steering and generate (steers both prompt and generation by default)
    steering.set('anger', scale=1.0)
    outputs = llm.generate(prompts, params)

    # Clear steering
    steering.clear()

Adapted from experiments/steering/core.py
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
    if hidden_states.dim() == 3:
        seq_len = hidden_states.shape[1]
    else:
        seq_len = 1
    is_prefill = seq_len > 1

    steer_prompt = state.get('steer_prompt', True)
    steer_generation = state.get('steer_generation', True)

    # Check if we should steer this phase
    if is_prefill and not steer_prompt:
        return outputs

    if not is_prefill and not steer_generation:
        return outputs

    # Handle tensor parallelism: slice vector ONLY if hidden states are sharded
    tp_rank, tp_world_size = _get_tp_info()
    if tp_world_size > 1:
        actual_hidden_dim = hidden_states.shape[-1]
        vector_dim = vector_tensor.shape[-1]

        if actual_hidden_dim == vector_dim:
            # Hidden states are NOT sharded - use full vector
            pass
        elif actual_hidden_dim == vector_dim // tp_world_size:
            # Hidden states ARE sharded - slice vector to match
            cache_key = f'_vector_tp{tp_rank}_{tp_world_size}'
            if cache_key not in state:
                state[cache_key] = _slice_vector_for_tp(vector_tensor, tp_rank, tp_world_size)
            vector_tensor = state[cache_key]

    # Apply steering: h' = h + scale * v
    device = hidden_states.device
    dtype = hidden_states.dtype

    vec = vector_tensor.to(device=device, dtype=dtype)

    # Position-specific steering (only during prefill with 3D tensor)
    steer_positions = state.get('steer_positions', None)
    if is_prefill and steer_positions is not None and hidden_states.dim() == 3:
        hidden_states = hidden_states.clone()
        for pos in steer_positions:
            if pos < hidden_states.shape[1]:
                hidden_states[:, pos, :] = hidden_states[:, pos, :] + scale * vec
    else:
        # Steer all positions
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
    """Register the steering hook on the target layer."""
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
    """Update the steering vector and scale on the target layer."""
    layer = _find_target_layer(model, layer_idx)

    if not hasattr(layer, '_steering_state'):
        return "ERROR: Hook not setup - call _setup_hook first"

    # Clear cached TP sliced vectors when updating
    keys_to_remove = [k for k in layer._steering_state if k.startswith('_vector_tp')]
    for k in keys_to_remove:
        del layer._steering_state[k]

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

    return f"Steering: scale={scale:.2f}, norm={np.linalg.norm(vector_np):.4f}"


def _clear_steering(model, layer_idx: int):
    """Clear steering state on the target layer."""
    layer = _find_target_layer(model, layer_idx)
    if hasattr(layer, '_steering_state'):
        layer._steering_state['vector_tensor'] = None
        layer._steering_state['scale'] = 0.0
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

    Example:
        llm = LLM(model="google/gemma-3-27b-it", enforce_eager=True)
        steering = VLLMSteering(llm, layer=30)
        steering.load_vectors('steering_tests/vectors/gemma3_27b/')

        steering.set('anger', scale=1.5)
        outputs = llm.generate(prompts, params)
    """

    def __init__(
        self,
        llm: LLM,
        layer: int,
        baseline_std: float = 1.0,
    ):
        """
        Initialize steering for a vLLM model.

        Args:
            llm: vLLM LLM instance (must have enforce_eager=True)
            layer: Layer index to apply steering
            baseline_std: Multiplier for scale. Default 1.0 means scale is raw magnitude.
        """
        self.llm = llm
        self.layer = layer
        self.baseline_std = baseline_std
        self.vectors: Dict[str, np.ndarray] = {}
        self._setup_hook()

    def _setup_hook(self):
        """Register the steering hook on the model."""
        setup_callable = _SetupHookCallable(self.layer)
        result = self.llm.apply_model(setup_callable)
        logger.info(f"Steering setup: {result}")

    def load_vectors(self, vector_dir: Union[str, Path]):
        """
        Load steering vectors from a directory.

        Supports multiple formats:
        - .npz files with 'vector' key (emotion_layer{N}.npz format)
        - .npy files (raw numpy arrays)
        """
        vector_dir = Path(vector_dir)

        # Try .npz format first (emotion_layer{N}.npz)
        for path in vector_dir.glob(f"*_layer{self.layer}.npz"):
            emotion = path.stem.replace(f"_layer{self.layer}", "")
            data = np.load(path)
            if 'vector' not in data.files:
                continue
            self.vectors[emotion] = data['vector']
            logger.info(f"Loaded vector: {emotion} (shape={self.vectors[emotion].shape})")

        # Also try loading from subdirectories (representation/layer structure)
        # e.g., vectors/gemma3_27b/base_emotion_vs_others/last_token/layer_30/anger.npy
        for subdir in vector_dir.iterdir():
            if not subdir.is_dir():
                continue
            for rep_dir in subdir.iterdir():
                if not rep_dir.is_dir():
                    continue
                layer_dir = rep_dir / f"layer_{self.layer}"
                if layer_dir.exists():
                    for npy_path in layer_dir.glob("*.npy"):
                        emotion = npy_path.stem
                        vector = np.load(npy_path)
                        key = f"{emotion}_{subdir.name}_{rep_dir.name}"
                        self.vectors[key] = vector
                        logger.info(f"Loaded vector: {key} (shape={vector.shape})")

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
            scale: Multiplier (e.g., 2.0 = 2x baseline_std)
            direction: +1 for positive steering, -1 for negative
            direction_std: If provided, use this for scaling instead of baseline_std
            steer_prompt: Whether to apply during prompt processing (prefill)
            steer_generation: Whether to apply during token generation (decode)
            steer_positions: Optional list of token positions to steer (None = all)
        """
        if emotion not in self.vectors:
            raise ValueError(f"Unknown emotion '{emotion}'. Available: {list(self.vectors.keys())}")

        vector = self.vectors[emotion]
        std_to_use = direction_std if direction_std is not None else self.baseline_std
        effective_scale = std_to_use * scale * direction
        vector_list = vector.tolist()

        update_callable = _UpdateSteeringCallable(
            self.layer, vector_list, effective_scale,
            steer_prompt, steer_generation, steer_positions
        )
        result = self.llm.apply_model(update_callable)
        logger.debug(f"Steering update: {result}")

    def set_raw_vector(
        self,
        vector,
        scale: float = 1.0,
        steer_prompt: bool = True,
        steer_generation: bool = True,
    ):
        """
        Apply a raw vector directly.

        Args:
            vector: The vector to apply (tensor or array)
            scale: Scaling factor to apply to the vector
            steer_prompt: Whether to apply during prompt processing
            steer_generation: Whether to apply during token generation
        """
        if hasattr(vector, 'numpy'):
            vector = vector.numpy()
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector_list = vector.tolist()

        update_callable = _UpdateSteeringCallable(
            self.layer, vector_list, scale, steer_prompt, steer_generation, None
        )
        result = self.llm.apply_model(update_callable)
        logger.info(f"Raw vector steering applied: {result}")

    def clear(self):
        """Clear all steering (return to baseline)."""
        clear_callable = _ClearSteeringCallable(self.layer)
        result = self.llm.apply_model(clear_callable)
        logger.debug(f"Steering cleared: {result}")

    @property
    def available_emotions(self) -> List[str]:
        """List of loaded emotion vectors."""
        return list(self.vectors.keys())


# ============================================================================
# MULTI-LAYER STEERING
# ============================================================================


class _SetupMultiLayerHookCallable:
    """Picklable callable for setting up steering hooks on multiple layers."""
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = layer_indices

    def __call__(self, model):
        results = []
        for layer_idx in self.layer_indices:
            results.append(_setup_hook(model, layer_idx))
        return f"Hooks registered on layers {self.layer_indices}"


class _UpdateMultiLayerSteeringCallable:
    """Picklable callable for updating steering on multiple layers."""
    def __init__(
        self,
        layer_indices: List[int],
        vector_list: Optional[List[float]],
        scales: List[float],
        steer_prompt: bool,
        steer_generation: bool,
    ):
        self.layer_indices = layer_indices
        self.vector_list = vector_list
        self.scales = scales
        self.steer_prompt = steer_prompt
        self.steer_generation = steer_generation

    def __call__(self, model):
        results = []
        for layer_idx, scale in zip(self.layer_indices, self.scales):
            result = _update_steering(
                model, layer_idx, self.vector_list, scale,
                self.steer_prompt, self.steer_generation, None
            )
            results.append(result)
        return results


class _ClearMultiLayerSteeringCallable:
    """Picklable callable for clearing steering on multiple layers."""
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = layer_indices

    def __call__(self, model):
        for layer_idx in self.layer_indices:
            _clear_steering(model, layer_idx)
        return f"Steering cleared on layers {self.layer_indices}"


class MultiLayerVLLMSteering:
    """
    Multi-layer steering for vLLM models.

    Applies steering across multiple layers simultaneously, distributing
    the total steering magnitude across layers.

    Example:
        llm = LLM(model="google/gemma-3-27b-it", enforce_eager=True)

        # Steer layers 30-34, each getting 1/5 of the total magnitude
        steering = MultiLayerVLLMSteering(
            llm,
            layers=[30, 31, 32, 33, 34],
            layer_norms={30: 43000, 31: 43500, ...}  # Optional per-layer norms
        )

        steering.load_vector('fear', fear_vector)
        steering.set('fear', scale=0.10)  # 10% total, 2% per layer
        outputs = llm.generate(prompts, params)
    """

    def __init__(
        self,
        llm: LLM,
        layers: List[int],
        layer_norms: Optional[Dict[int, float]] = None,
        uniform_norm: Optional[float] = None,
    ):
        """
        Initialize multi-layer steering.

        Args:
            llm: vLLM LLM instance (must have enforce_eager=True)
            layers: List of layer indices to steer
            layer_norms: Dict mapping layer index to its residual stream norm.
                        Used for proper scaling per layer.
            uniform_norm: If provided, use this norm for all layers instead of
                         per-layer norms. Simpler but less accurate.
        """
        self.llm = llm
        self.layers = layers
        self.num_layers = len(layers)
        self.vectors: Dict[str, np.ndarray] = {}

        # Setup layer norms
        if layer_norms is not None:
            self.layer_norms = layer_norms
        elif uniform_norm is not None:
            self.layer_norms = {layer: uniform_norm for layer in layers}
        else:
            # Default to 1.0 if no norms provided (user should provide norms!)
            logger.warning("No layer_norms provided - using 1.0 for all layers. "
                          "For proper scaling, provide layer_norms or uniform_norm.")
            self.layer_norms = {layer: 1.0 for layer in layers}

        self._setup_hooks()

    def _setup_hooks(self):
        """Register steering hooks on all layers."""
        setup_callable = _SetupMultiLayerHookCallable(self.layers)
        result = self.llm.apply_model(setup_callable)
        logger.info(f"Multi-layer steering setup: {result}")

    def load_vector(self, name: str, vector: np.ndarray):
        """Load a single vector by name."""
        self.vectors[name] = vector

    def set(
        self,
        emotion: str,
        scale: float = 1.0,
        direction: int = 1,
        steer_prompt: bool = True,
        steer_generation: bool = True,
    ):
        """
        Set steering across all layers.

        The total steering magnitude (scale) is divided equally across layers,
        with each layer's contribution scaled by its layer norm.

        Args:
            emotion: Name of the emotion/direction to steer
            scale: Total steering magnitude as fraction of layer norm (e.g., 0.10 = 10%)
            direction: +1 for positive steering, -1 for negative
            steer_prompt: Whether to apply during prompt processing (prefill)
            steer_generation: Whether to apply during token generation (decode)
        """
        if emotion not in self.vectors:
            raise ValueError(f"Unknown emotion '{emotion}'. Available: {list(self.vectors.keys())}")

        vector = self.vectors[emotion]
        vector_list = vector.tolist()

        # Compute per-layer scales: each layer gets (scale / num_layers) * layer_norm
        per_layer_fraction = scale / self.num_layers
        scales = []
        for layer in self.layers:
            layer_norm = self.layer_norms[layer]
            layer_scale = per_layer_fraction * layer_norm * direction
            scales.append(layer_scale)

        logger.info(f"Multi-layer steering: {emotion} @ {scale*100:.1f}% total "
                   f"({per_layer_fraction*100:.2f}% per layer), direction={direction}")
        for layer, s in zip(self.layers, scales):
            logger.debug(f"  Layer {layer}: scale={s:.2f}")

        update_callable = _UpdateMultiLayerSteeringCallable(
            self.layers, vector_list, scales, steer_prompt, steer_generation
        )
        result = self.llm.apply_model(update_callable)
        logger.debug(f"Multi-layer steering update: {result}")

    def set_raw_vector(
        self,
        vector: np.ndarray,
        scale: float = 1.0,
        steer_prompt: bool = True,
        steer_generation: bool = True,
    ):
        """
        Apply a raw vector directly across all layers.

        Args:
            vector: The vector to apply
            scale: Total scaling factor (divided across layers)
            steer_prompt: Whether to apply during prompt processing
            steer_generation: Whether to apply during token generation
        """
        if hasattr(vector, 'numpy'):
            vector = vector.numpy()
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector_list = vector.tolist()

        # Divide scale equally across layers
        per_layer_scale = scale / self.num_layers
        scales = [per_layer_scale] * self.num_layers

        update_callable = _UpdateMultiLayerSteeringCallable(
            self.layers, vector_list, scales, steer_prompt, steer_generation
        )
        result = self.llm.apply_model(update_callable)
        logger.info(f"Raw multi-layer steering applied: {result}")

    def clear(self):
        """Clear steering on all layers."""
        clear_callable = _ClearMultiLayerSteeringCallable(self.layers)
        result = self.llm.apply_model(clear_callable)
        if hasattr(self, '_layer_vectors'):
            self._layer_vectors.clear()
        logger.debug(f"Multi-layer steering cleared: {result}")

    def set_layer_vector(
        self,
        layer: int,
        vector: np.ndarray,
        scale: float = 1.0,
        steer_prompt: bool = True,
        steer_generation: bool = True,
    ):
        """
        Set steering for a specific layer using a custom vector.

        This allows per-layer customization, useful when combining different
        vectors at different layers (e.g., emotion + suppression).

        Args:
            layer: Layer index to set
            vector: Vector to apply at this layer
            scale: Scaling factor for the vector
            steer_prompt: Whether to apply during prompt processing
            steer_generation: Whether to apply during token generation
        """
        if layer not in self.layers:
            raise ValueError(f"Layer {layer} not in steering layers: {self.layers}")

        if hasattr(vector, 'numpy'):
            vector = vector.numpy()
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector_list = vector.tolist()

        # Store in per-layer vectors
        if not hasattr(self, '_layer_vectors'):
            self._layer_vectors = {}
        self._layer_vectors[layer] = (vector_list, scale, steer_prompt, steer_generation)

        # Apply to this single layer
        update_callable = _UpdateSteeringCallable(
            layer, vector_list, scale, steer_prompt, steer_generation, None
        )
        result = self.llm.apply_model(update_callable)
        logger.debug(f"Layer {layer} vector set: {result}")

    def get_layer_steering(self, layer: int) -> Optional[np.ndarray]:
        """
        Get the current steering vector for a specific layer.

        Returns None if no steering is set for that layer.
        """
        if not hasattr(self, '_layer_vectors'):
            return None
        if layer not in self._layer_vectors:
            return None

        vector_list, scale, _, _ = self._layer_vectors[layer]
        return np.array(vector_list, dtype=np.float32) * scale

    @property
    def available_emotions(self) -> List[str]:
        """List of loaded emotion vectors."""
        return list(self.vectors.keys())
