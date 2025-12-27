"""Steered model wrapper for applying activation interventions."""

from typing import Optional, List, Dict, Union
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vectors import SteeringVector


class SteeredModel:
    """Model wrapper that applies steering vectors during generation.

    Supports multiple architectures: Gemma, LLaMA, Mistral, Qwen, GPT-style.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize steered model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            torch_dtype: Data type for model

        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

        # Load model and tokenizer
        try:
            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device,
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        # Get layer access
        self.layers = self._get_model_layers()
        self.n_layers = len(self.layers)

        # Steering state
        self.steering_vectors: Dict[int, torch.Tensor] = {}  # {layer: summed_vector}
        self.hook_handles = []

        print(f"Model loaded: {self.n_layers} layers")

    def _get_model_layers(self):
        """Get transformer layers (architecture-agnostic).

        Returns:
            Module list of transformer layers

        Raises:
            RuntimeError: If cannot find layers
        """
        # Try different architecture patterns
        try:
            # Gemma3 multimodal
            if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
                return self.model.model.language_model.layers
        except AttributeError:
            pass

        try:
            # LLaMA, Mistral, Qwen, Gemma2
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers
        except AttributeError:
            pass

        try:
            # GPT-style
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return self.model.transformer.h
        except AttributeError:
            pass

        try:
            # Decoder-only (generic)
            if hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
                return self.model.model.decoder.layers
        except AttributeError:
            pass

        raise RuntimeError(
            f"Cannot find transformer layers for model {self.model_name}. "
            f"Supported: Gemma, LLaMA, Mistral, Qwen, GPT. "
            f"Model structure: {type(self.model)}"
        )

    def add_steering(
        self,
        vector: SteeringVector,
        strength: float = 1.0,
    ) -> None:
        """Add steering vector (accumulates with existing vectors at same layer).

        Args:
            vector: Steering vector to apply
            strength: Multiplicative strength factor

        Raises:
            ValueError: If layer out of range or vector contains NaN/Inf
        """
        if vector.layer < 0 or vector.layer >= self.n_layers:
            raise ValueError(
                f"Steering layer {vector.layer} out of range [0, {self.n_layers})"
            )

        if np.isnan(vector.vector).any() or np.isinf(vector.vector).any():
            raise ValueError("Steering vector contains NaN or Inf")

        # Convert to torch tensor
        steering_tensor = torch.tensor(
            vector.vector * strength,
            dtype=self.torch_dtype,
            device=self.device,
        )

        # Accumulate at layer
        if vector.layer in self.steering_vectors:
            self.steering_vectors[vector.layer] += steering_tensor
            print(f"Added to existing steering at layer {vector.layer} (now {len(self.steering_vectors)} layers)")
        else:
            self.steering_vectors[vector.layer] = steering_tensor
            print(f"Added steering at layer {vector.layer}: {vector.name} (strength={strength})")

    def clear_steering(self) -> None:
        """Remove all steering vectors and hooks."""
        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Clear vectors
        self.steering_vectors = {}

        print("Cleared all steering")

    def _make_steering_hook(self, layer_idx: int):
        """Create hook function for given layer.

        Args:
            layer_idx: Layer index

        Returns:
            Hook function
        """
        steering_vec = self.steering_vectors[layer_idx]

        def hook(module, input, output):
            """Apply steering to hidden states."""
            # Output is tuple: (hidden_states, ...)
            hidden_states = output[0]

            # Steering vector shape: [hidden_dim]
            # Hidden states shape: [batch, seq_len, hidden_dim]
            # Broadcasting adds steering to all tokens
            steered = hidden_states + steering_vec

            # Check for NaN/Inf
            if torch.isnan(steered).any() or torch.isinf(steered).any():
                raise RuntimeError(
                    f"Steering produced NaN/Inf at layer {layer_idx}. "
                    f"Hidden states: {hidden_states.abs().max().item()}, "
                    f"Steering: {steering_vec.abs().max().item()}"
                )

            return (steered,) + output[1:]

        return hook

    def apply_steering(self) -> None:
        """Apply steering hooks to model.

        Must be called after add_steering() and before generate().

        Raises:
            RuntimeError: If no steering vectors added
        """
        if not self.steering_vectors:
            raise RuntimeError("No steering vectors added. Call add_steering() first.")

        # Clear existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register hooks
        for layer_idx in sorted(self.steering_vectors.keys()):
            hook = self._make_steering_hook(layer_idx)
            handle = self.layers[layer_idx].register_forward_hook(hook)
            self.hook_handles.append(handle)

        print(f"Applied steering hooks to {len(self.hook_handles)} layers")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """Generate text with steering applied.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            **kwargs: Additional generation parameters

        Returns:
            Generated text (full, including prompt)

        Raises:
            RuntimeError: If steering not applied
        """
        if not self.hook_handles:
            raise RuntimeError(
                "Steering hooks not applied. Call apply_steering() first."
            )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def compare_strengths(
        self,
        prompt: str,
        vector: SteeringVector,
        strengths: List[float],
        max_new_tokens: int = 100,
    ) -> Dict[float, str]:
        """Compare generation across different steering strengths.

        Args:
            prompt: Input prompt
            vector: Steering vector to test
            strengths: List of strength values to try
            max_new_tokens: Tokens to generate per trial

        Returns:
            Dict mapping strength to generated text
        """
        results = {}

        for strength in strengths:
            # Clear and add steering
            self.clear_steering()
            self.add_steering(vector, strength=strength)
            self.apply_steering()

            # Generate
            text = self.generate(prompt, max_new_tokens=max_new_tokens)
            results[strength] = text

            print(f"\nStrength {strength}:")
            print(text)
            print("-" * 80)

        # Clear after comparison
        self.clear_steering()

        return results

    def get_steering_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics about current steering vectors.

        Returns:
            Dict mapping layer to stats (norm, min, max, mean)
        """
        stats = {}

        for layer_idx, vec in self.steering_vectors.items():
            vec_np = vec.cpu().numpy()
            stats[layer_idx] = {
                "norm": float(np.linalg.norm(vec_np)),
                "min": float(vec_np.min()),
                "max": float(vec_np.max()),
                "mean": float(vec_np.mean()),
                "std": float(vec_np.std()),
            }

        return stats
