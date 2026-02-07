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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

try:
    from vllm import LLM
except ImportError:
    LLM = None  # Allow importing module without vllm (e.g., for judge scripts)

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
# TOKEN TRIGGER DATACLASS
# ============================================================================

@dataclass
class TokenTrigger:
    """Configuration for token-triggered steering activation/deactivation.

    When set, steering is only applied when triggered by specific tokens in the
    generated output. This enables patterns like "only steer inside <think> blocks".

    Works per-sequence in batched generation — different sequences can have their
    triggers fire at different times.

    Args:
        start_tokens: Token IDs that activate steering
        end_tokens: Token IDs that deactivate steering
        initially_active: Whether steering is active before any trigger fires
    """
    start_tokens: List[int] = field(default_factory=list)
    end_tokens: List[int] = field(default_factory=list)
    initially_active: bool = False


# ============================================================================
# TOKEN TRACKING AND TRIGGER HOOKS
# These run in the vLLM worker process, set up via apply_model()
# ============================================================================

def _token_tracking_pre_hook(module, inputs):
    """Pre-hook on layer 0 that counts decode steps and resets on prefill.

    Updates the shared state dict that all steering hooks can read.
    Fires once per forward pass, before any steering hooks.

    vLLM v0 layer signature: forward(self, positions, hidden_states, residual, ...)
    - inputs[0] = positions tensor (1D): shape [num_tokens]
    - During prefill: positions = [0, 1, 2, ..., prompt_len-1] (contains 0)
    - During decode: positions = [next_pos_1, ...] (all > 0)
    """
    if not hasattr(module, '_steering_shared_state'):
        return

    shared = module._steering_shared_state

    # Extract positions tensor (first input to vLLM layer forward)
    if isinstance(inputs, tuple):
        positions = inputs[0]
    else:
        positions = inputs

    # Detect prefill: during prefill, at least one sequence starts at position 0.
    # During decode, all positions are > 0 (each sequence has generated >= 1 token).
    is_prefill = (positions.min().item() == 0)

    # Store in shared state so steering hooks can read it
    shared['is_prefill'] = is_prefill

    if is_prefill:
        # Reset step counter on new sequence
        shared['generation_step'] = 0
        # Reset trigger mask to initial state
        if 'initially_active' in shared:
            initial_val = 1.0 if shared['initially_active'] else 0.0
            if 'trigger_mask' in shared:
                shared['trigger_mask'].fill_(initial_val)
    else:
        # Increment decode step counter
        shared['generation_step'] = shared.get('generation_step', 0) + 1


def _embedding_hook(module, inputs, outputs):
    """Forward hook on embed_tokens that checks generated tokens for triggers.

    Reads the input_ids being embedded and updates the per-sequence trigger mask
    in the shared state dict. Only operates during decode (single token per sequence).
    """
    if not hasattr(module, '_steering_shared_state'):
        return

    shared = module._steering_shared_state
    trigger_start_ids = shared.get('trigger_start_ids')
    trigger_end_ids = shared.get('trigger_end_ids')

    if trigger_start_ids is None and trigger_end_ids is None:
        return

    # Get input_ids from the embedding layer's input
    input_ids = inputs[0]  # shape: [num_tokens] or [batch, seq_len]

    if input_ids.dim() == 2:
        seq_len = input_ids.shape[1]
        if seq_len > 1:
            return  # Prefill — don't check triggers
        # Decode: input_ids is [batch, 1]
        last_tokens = input_ids[:, -1]  # [batch]
    elif input_ids.dim() == 1:
        # Flattened tokens — during decode with N sequences, shape is [N]
        # Each element is the latest token for that sequence
        # But during prefill this is [total_tokens] which is > batch_size
        # Use a heuristic: if trigger_mask exists, compare sizes
        if 'trigger_mask' in shared:
            expected_batch = shared['trigger_mask'].shape[0]
            if input_ids.shape[0] != expected_batch:
                return  # Likely prefill, skip
        last_tokens = input_ids  # [batch]
    else:
        return

    # Ensure trigger_mask exists and has the right size
    batch_size = last_tokens.shape[0]
    if 'trigger_mask' not in shared or shared['trigger_mask'].shape[0] != batch_size:
        initial_val = 1.0 if shared.get('initially_active', False) else 0.0
        shared['trigger_mask'] = torch.full(
            (batch_size,), initial_val, dtype=torch.float32, device=last_tokens.device,
        )

    mask = shared['trigger_mask']

    # Check each sequence's latest token against triggers
    if trigger_start_ids:
        start_set = set(trigger_start_ids)
        for i in range(batch_size):
            if last_tokens[i].item() in start_set:
                mask[i] = 1.0

    if trigger_end_ids:
        end_set = set(trigger_end_ids)
        for i in range(batch_size):
            if last_tokens[i].item() in end_set:
                mask[i] = 0.0


def _find_embedding_layer(model):
    """Find the token embedding layer in various model architectures."""
    # Gemma 3 / multimodal models
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        inner = model.language_model.model
        if hasattr(inner, 'embed_tokens'):
            return inner.embed_tokens
    # Standard decoder-only models
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    # Some models use get_input_embeddings()
    if hasattr(model, 'get_input_embeddings'):
        embed = model.get_input_embeddings()
        if embed is not None:
            return embed
    raise ValueError("Could not find embedding layer in model architecture")


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
    - Generation token range gating (only steer decode steps in [start, end))
    - Token-triggered per-sequence steering masks
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

    # Detect prefill vs decode.
    # Primary: read from shared state (set by _token_tracking_pre_hook on layer 0).
    # Fallback: use hidden_states token count (works when batch_size=1 in decode).
    shared = state.get('_steering_shared_state')
    if shared is not None and 'is_prefill' in shared:
        is_prefill = shared['is_prefill']
    else:
        # Fallback for when token tracking is not enabled.
        # vLLM 2D tensors: [num_tokens, hidden_dim].
        # During prefill num_tokens = prompt_len (large), during decode num_tokens = batch_size (small).
        # Heuristic: > 16 tokens is likely prefill. Only used without token tracking.
        num_tokens = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
        is_prefill = num_tokens > 16

    steer_prompt = state.get('steer_prompt', True)
    steer_generation = state.get('steer_generation', True)

    # Check if we should steer this phase
    if is_prefill and not steer_prompt:
        return outputs

    if not is_prefill and not steer_generation:
        return outputs

    # Check generation_token_range gating (decode only)
    if not is_prefill:
        gen_range = state.get('generation_token_range')
        if gen_range is not None:
            if shared is not None:
                step = shared.get('generation_step', 0)
                start, end = gen_range
                if step < start or step >= end:
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

    # Check trigger mask for per-sequence steering (decode only)
    if not is_prefill:
        shared = state.get('_steering_shared_state')
        if shared is not None:
            trigger_mask = shared.get('trigger_mask')
            if trigger_mask is not None:
                mask = trigger_mask.to(device=device, dtype=dtype)  # [batch]
                # Position-specific steering not applicable during decode
                # Apply per-sequence mask: [batch, 1] * [hidden] -> per-sequence
                if hidden_states.dim() == 3:
                    # [batch, 1, 1] broadcast over [batch, seq_len, hidden]
                    hidden_states = hidden_states + (scale * vec) * mask.unsqueeze(-1).unsqueeze(-1)
                else:
                    # [batch, hidden] — mask is [batch], broadcast to [batch, 1]
                    hidden_states = hidden_states + (scale * vec) * mask.unsqueeze(-1)
                if rest is not None:
                    return (hidden_states,) + rest
                return hidden_states

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
                 steer_prompt: bool, steer_generation: bool, steer_positions,
                 generation_token_range=None):
        self.layer_idx = layer_idx
        self.vector_list = vector_list
        self.scale = scale
        self.steer_prompt = steer_prompt
        self.steer_generation = steer_generation
        self.steer_positions = steer_positions
        self.generation_token_range = generation_token_range

    def __call__(self, model):
        return _update_steering(
            model, self.layer_idx, self.vector_list, self.scale,
            self.steer_prompt, self.steer_generation, self.steer_positions,
            self.generation_token_range,
        )


class _ClearSteeringCallable:
    """Picklable callable for clearing steering via vLLM apply_model."""
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx

    def __call__(self, model):
        return _clear_steering(model, self.layer_idx)


class _EnableTokenTrackingCallable:
    """Picklable callable for setting up token tracking and optional triggers.

    Creates shared state on layer 0, registers a pre-hook for step counting,
    and links the shared state to each steering layer. Optionally registers
    an embedding hook for token-triggered steering.
    """
    def __init__(
        self,
        steering_layer_indices: List[int],
        trigger_start_ids: Optional[List[int]] = None,
        trigger_end_ids: Optional[List[int]] = None,
        initially_active: bool = True,
    ):
        self.steering_layer_indices = steering_layer_indices
        self.trigger_start_ids = trigger_start_ids
        self.trigger_end_ids = trigger_end_ids
        self.initially_active = initially_active

    def __call__(self, model):
        layer_0 = _find_target_layer(model, 0)

        # Create or reuse shared state dict on layer 0
        if not hasattr(layer_0, '_steering_shared_state'):
            layer_0._steering_shared_state = {
                'generation_step': 0,
                'initially_active': self.initially_active,
            }
        shared = layer_0._steering_shared_state
        shared['initially_active'] = self.initially_active

        # Register pre-hook on layer 0 for step counting (only once)
        if not hasattr(layer_0, '_token_tracking_handle'):
            layer_0._token_tracking_handle = layer_0.register_forward_pre_hook(
                _token_tracking_pre_hook
            )

        # Link shared state to each steering layer
        for layer_idx in self.steering_layer_indices:
            layer = _find_target_layer(model, layer_idx)
            if hasattr(layer, '_steering_state'):
                layer._steering_state['_steering_shared_state'] = shared
            # Also store directly on the layer for the pre-hook to find
            layer._steering_shared_state = shared

        # Set up trigger tokens in shared state
        if self.trigger_start_ids is not None:
            shared['trigger_start_ids'] = self.trigger_start_ids
        if self.trigger_end_ids is not None:
            shared['trigger_end_ids'] = self.trigger_end_ids

        # Register embedding hook for triggers if needed
        has_triggers = (self.trigger_start_ids or self.trigger_end_ids)
        if has_triggers:
            embed_layer = _find_embedding_layer(model)
            if not hasattr(embed_layer, '_trigger_hook_handle'):
                embed_layer._steering_shared_state = shared
                embed_layer._trigger_hook_handle = embed_layer.register_forward_hook(
                    _embedding_hook
                )
            else:
                # Update shared state reference
                embed_layer._steering_shared_state = shared

        return f"Token tracking enabled on {len(self.steering_layer_indices)} layers"


class _DisableTokenTrackingCallable:
    """Picklable callable for removing token tracking hooks."""
    def __init__(self, steering_layer_indices: List[int]):
        self.steering_layer_indices = steering_layer_indices

    def __call__(self, model):
        layer_0 = _find_target_layer(model, 0)

        # Remove pre-hook
        if hasattr(layer_0, '_token_tracking_handle'):
            layer_0._token_tracking_handle.remove()
            del layer_0._token_tracking_handle

        # Clean up shared state references
        if hasattr(layer_0, '_steering_shared_state'):
            del layer_0._steering_shared_state

        for layer_idx in self.steering_layer_indices:
            layer = _find_target_layer(model, layer_idx)
            if hasattr(layer, '_steering_shared_state'):
                del layer._steering_shared_state
            if hasattr(layer, '_steering_state') and '_steering_shared_state' in layer._steering_state:
                del layer._steering_state['_steering_shared_state']

        # Remove embedding hook
        try:
            embed_layer = _find_embedding_layer(model)
            if hasattr(embed_layer, '_trigger_hook_handle'):
                embed_layer._trigger_hook_handle.remove()
                del embed_layer._trigger_hook_handle
            if hasattr(embed_layer, '_steering_shared_state'):
                del embed_layer._steering_shared_state
        except ValueError:
            pass  # No embedding layer found — nothing to clean up

        return f"Token tracking disabled"


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
    generation_token_range: Optional[tuple] = None,
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
        layer._steering_state['generation_token_range'] = None
        return "Steering disabled"

    vector_np = np.array(vector_list, dtype=np.float32)
    layer._steering_state['vector_tensor'] = torch.from_numpy(vector_np)
    layer._steering_state['scale'] = scale
    layer._steering_state['steer_prompt'] = steer_prompt
    layer._steering_state['steer_generation'] = steer_generation
    layer._steering_state['steer_positions'] = steer_positions
    layer._steering_state['generation_token_range'] = generation_token_range

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
        generation_token_range: Optional[tuple] = None,
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
            generation_token_range: Optional (start, end) tuple - only steer decode
                steps in [start, end). Requires enable_token_tracking().
        """
        if emotion not in self.vectors:
            raise ValueError(f"Unknown emotion '{emotion}'. Available: {list(self.vectors.keys())}")

        vector = self.vectors[emotion]
        std_to_use = direction_std if direction_std is not None else self.baseline_std
        effective_scale = std_to_use * scale * direction
        vector_list = vector.tolist()

        update_callable = _UpdateSteeringCallable(
            self.layer, vector_list, effective_scale,
            steer_prompt, steer_generation, steer_positions,
            generation_token_range,
        )
        result = self.llm.apply_model(update_callable)
        logger.debug(f"Steering update: {result}")

    def set_raw_vector(
        self,
        vector,
        scale: float = 1.0,
        steer_prompt: bool = True,
        steer_generation: bool = True,
        steer_positions: Optional[List[int]] = None,
        generation_token_range: Optional[tuple] = None,
    ):
        """
        Apply a raw vector directly.

        Args:
            vector: The vector to apply (tensor or array)
            scale: Scaling factor to apply to the vector
            steer_prompt: Whether to apply during prompt processing
            steer_generation: Whether to apply during token generation
            steer_positions: Optional list of token positions to steer (None = all)
            generation_token_range: Optional (start, end) tuple - only steer decode
                steps in [start, end). Requires enable_token_tracking().
        """
        if hasattr(vector, 'numpy'):
            vector = vector.numpy()
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        vector_list = vector.tolist()

        update_callable = _UpdateSteeringCallable(
            self.layer, vector_list, scale, steer_prompt, steer_generation,
            steer_positions, generation_token_range,
        )
        result = self.llm.apply_model(update_callable)
        logger.info(f"Raw vector steering applied: {result}")

    def enable_token_tracking(
        self,
        trigger: Optional[TokenTrigger] = None,
        tokenizer=None,
        trigger_start_strings: Optional[List[str]] = None,
        trigger_end_strings: Optional[List[str]] = None,
    ):
        """Enable per-token generation tracking for this steering layer.

        Required for generation_token_range to work. Optionally enables
        token-triggered steering activation/deactivation.

        Args:
            trigger: TokenTrigger config with start/end token IDs
            tokenizer: If provided with trigger_start/end_strings, resolves
                string tokens to IDs automatically
            trigger_start_strings: Strings that activate steering (resolved via tokenizer)
            trigger_end_strings: Strings that deactivate steering (resolved via tokenizer)
        """
        trigger_start_ids = None
        trigger_end_ids = None
        initially_active = True

        if trigger is not None:
            trigger_start_ids = trigger.start_tokens or None
            trigger_end_ids = trigger.end_tokens or None
            initially_active = trigger.initially_active
        elif tokenizer is not None and (trigger_start_strings or trigger_end_strings):
            if trigger_start_strings:
                trigger_start_ids = []
                for s in trigger_start_strings:
                    ids = tokenizer.encode(s, add_special_tokens=False)
                    trigger_start_ids.extend(ids)
            if trigger_end_strings:
                trigger_end_ids = []
                for s in trigger_end_strings:
                    ids = tokenizer.encode(s, add_special_tokens=False)
                    trigger_end_ids.extend(ids)
            initially_active = False  # Default: inactive until first start trigger

        callable_ = _EnableTokenTrackingCallable(
            steering_layer_indices=[self.layer],
            trigger_start_ids=trigger_start_ids,
            trigger_end_ids=trigger_end_ids,
            initially_active=initially_active,
        )
        result = self.llm.apply_model(callable_)
        self._token_tracking_enabled = True
        logger.info(f"Token tracking: {result}")

    def disable_token_tracking(self):
        """Disable per-token generation tracking."""
        callable_ = _DisableTokenTrackingCallable(
            steering_layer_indices=[self.layer],
        )
        result = self.llm.apply_model(callable_)
        self._token_tracking_enabled = False
        logger.info(f"Token tracking: {result}")

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
        steer_positions: Optional[List[int]] = None,
        generation_token_range: Optional[tuple] = None,
    ):
        self.layer_indices = layer_indices
        self.vector_list = vector_list
        self.scales = scales
        self.steer_prompt = steer_prompt
        self.steer_generation = steer_generation
        self.steer_positions = steer_positions
        self.generation_token_range = generation_token_range

    def __call__(self, model):
        results = []
        for layer_idx, scale in zip(self.layer_indices, self.scales):
            result = _update_steering(
                model, layer_idx, self.vector_list, scale,
                self.steer_prompt, self.steer_generation, self.steer_positions,
                self.generation_token_range,
            )
            results.append(result)
        return results


class _UpdateMultiLayerPerVectorSteeringCallable:
    """Picklable callable for updating steering with per-layer vectors in one apply_model() call."""
    def __init__(
        self,
        layer_configs: Dict[int, dict],
        steer_prompt: bool,
        steer_generation: bool,
        steer_positions: Optional[List[int]] = None,
        generation_token_range: Optional[tuple] = None,
    ):
        # layer_configs: {layer_idx: {'vector_list': [...], 'scale': float}}
        self.layer_configs = layer_configs
        self.steer_prompt = steer_prompt
        self.steer_generation = steer_generation
        self.steer_positions = steer_positions
        self.generation_token_range = generation_token_range

    def __call__(self, model):
        results = []
        for layer_idx, cfg in self.layer_configs.items():
            result = _update_steering(
                model, layer_idx, cfg['vector_list'], cfg['scale'],
                self.steer_prompt, self.steer_generation, self.steer_positions,
                self.generation_token_range,
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
        steer_positions: Optional[List[int]] = None,
        generation_token_range: Optional[tuple] = None,
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
            steer_positions: Optional list of token positions to steer (None = all)
            generation_token_range: Optional (start, end) tuple - only steer decode
                steps in [start, end). Requires enable_token_tracking().
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
            self.layers, vector_list, scales, steer_prompt, steer_generation,
            steer_positions, generation_token_range,
        )
        result = self.llm.apply_model(update_callable)
        logger.debug(f"Multi-layer steering update: {result}")

    def set_raw_vector(
        self,
        vector: np.ndarray,
        scale: float = 1.0,
        steer_prompt: bool = True,
        steer_generation: bool = True,
        steer_positions: Optional[List[int]] = None,
        generation_token_range: Optional[tuple] = None,
    ):
        """
        Apply a raw vector directly across all layers.

        Args:
            vector: The vector to apply
            scale: Total scaling factor (divided across layers)
            steer_prompt: Whether to apply during prompt processing
            steer_generation: Whether to apply during token generation
            steer_positions: Optional list of token positions to steer (None = all)
            generation_token_range: Optional (start, end) tuple - only steer decode
                steps in [start, end). Requires enable_token_tracking().
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
            self.layers, vector_list, scales, steer_prompt, steer_generation,
            steer_positions, generation_token_range,
        )
        result = self.llm.apply_model(update_callable)
        logger.info(f"Raw multi-layer steering applied: {result}")

    def set_per_layer(
        self,
        layer_vectors: Dict[int, np.ndarray],
        layer_scales: Optional[Dict[int, float]] = None,
        steer_prompt: bool = True,
        steer_generation: bool = True,
        steer_positions: Optional[List[int]] = None,
        generation_token_range: Optional[tuple] = None,
    ):
        """
        Set per-layer steering vectors in a single apply_model() call.

        Unlike set_layer_vector() which requires one apply_model() call per layer,
        this sets all layers at once — much faster when each layer has a different
        vector (e.g., combined emotion + suppression vectors).

        Args:
            layer_vectors: Dict mapping layer index to its steering vector (np.ndarray)
            layer_scales: Optional dict mapping layer index to its scale. Defaults to 1.0
                for all layers.
            steer_prompt: Whether to apply during prompt processing (prefill)
            steer_generation: Whether to apply during token generation (decode)
            steer_positions: Optional list of token positions to steer (None = all)
            generation_token_range: Optional (start, end) tuple - only steer decode
                steps in [start, end). Requires enable_token_tracking().
        """
        for layer_idx in layer_vectors:
            if layer_idx not in self.layers:
                raise ValueError(f"Layer {layer_idx} not in steering layers: {self.layers}")

        if layer_scales is None:
            layer_scales = {}

        # Build per-layer configs
        layer_configs = {}
        for layer_idx, vector in layer_vectors.items():
            if hasattr(vector, 'numpy'):
                vector = vector.numpy()
            elif not isinstance(vector, np.ndarray):
                vector = np.array(vector)

            scale = layer_scales.get(layer_idx, 1.0)
            layer_configs[layer_idx] = {
                'vector_list': vector.tolist(),
                'scale': scale,
            }

            # Track in _layer_vectors for get_layer_steering()
            if not hasattr(self, '_layer_vectors'):
                self._layer_vectors = {}
            self._layer_vectors[layer_idx] = (
                layer_configs[layer_idx]['vector_list'], scale,
                steer_prompt, steer_generation,
            )

        update_callable = _UpdateMultiLayerPerVectorSteeringCallable(
            layer_configs, steer_prompt, steer_generation,
            steer_positions, generation_token_range,
        )
        result = self.llm.apply_model(update_callable)
        logger.info(f"Per-layer steering set on {len(layer_configs)} layers: {result}")

    def enable_token_tracking(
        self,
        trigger: Optional[TokenTrigger] = None,
        tokenizer=None,
        trigger_start_strings: Optional[List[str]] = None,
        trigger_end_strings: Optional[List[str]] = None,
    ):
        """Enable per-token generation tracking for all steering layers.

        Required for generation_token_range to work. Optionally enables
        token-triggered steering activation/deactivation.

        Args:
            trigger: TokenTrigger config with start/end token IDs
            tokenizer: If provided with trigger_start/end_strings, resolves
                string tokens to IDs automatically
            trigger_start_strings: Strings that activate steering (resolved via tokenizer)
            trigger_end_strings: Strings that deactivate steering (resolved via tokenizer)
        """
        trigger_start_ids = None
        trigger_end_ids = None
        initially_active = True

        if trigger is not None:
            trigger_start_ids = trigger.start_tokens or None
            trigger_end_ids = trigger.end_tokens or None
            initially_active = trigger.initially_active
        elif tokenizer is not None and (trigger_start_strings or trigger_end_strings):
            if trigger_start_strings:
                trigger_start_ids = []
                for s in trigger_start_strings:
                    ids = tokenizer.encode(s, add_special_tokens=False)
                    trigger_start_ids.extend(ids)
            if trigger_end_strings:
                trigger_end_ids = []
                for s in trigger_end_strings:
                    ids = tokenizer.encode(s, add_special_tokens=False)
                    trigger_end_ids.extend(ids)
            initially_active = False  # Default: inactive until first start trigger

        callable_ = _EnableTokenTrackingCallable(
            steering_layer_indices=self.layers,
            trigger_start_ids=trigger_start_ids,
            trigger_end_ids=trigger_end_ids,
            initially_active=initially_active,
        )
        result = self.llm.apply_model(callable_)
        self._token_tracking_enabled = True
        logger.info(f"Token tracking: {result}")

    def disable_token_tracking(self):
        """Disable per-token generation tracking."""
        callable_ = _DisableTokenTrackingCallable(
            steering_layer_indices=self.layers,
        )
        result = self.llm.apply_model(callable_)
        self._token_tracking_enabled = False
        logger.info(f"Token tracking: {result}")

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
