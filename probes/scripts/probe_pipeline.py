#!/usr/bin/env python3
"""
Core pipeline classes for flexible probe-based emotion detection.

This module provides modular, reusable classes for:
- Extracting activations from models
- Running probe inference with caching
- Aggregating and computing differences
- Creating visualizations

All classes are standalone and don't depend on the emo_lens codebase.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from activation_extraction import extract_prompt_activations_all_layers


# ============================================================================
# 1. ProbeActivationExtractor - Handles all activation extraction patterns
# ============================================================================

class ProbeActivationExtractor:
    """Handles activation extraction from models with various strategies."""

    def extract_aggregated(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layer: int,
        system_prompt: Optional[str] = None,
        strategy: str = "assistant_token",
        num_generated_tokens: int = 10
    ) -> np.ndarray:
        """Extract activations for a batch of prompts at a single layer with aggregation.

        Args:
            model: StandardizedTransformer model
            tokenizer: Tokenizer
            prompts: List of prompt strings
            layer: Layer to extract from
            system_prompt: Optional system prompt
            strategy: Activation extraction strategy (assistant_token, etc.)
            num_generated_tokens: Number of tokens for generated_tokens_avg

        Returns:
            Activations array [n_prompts, hidden_dim]
        """
        activations_list = []

        for prompt in prompts:
            acts_dict = extract_prompt_activations_all_layers(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=[layer],
                system_prompt=system_prompt,
                strategy=strategy,
                num_generated_tokens=num_generated_tokens
            )
            activations_list.append(acts_dict[layer])

        return np.array(activations_list)

    def extract_batch_multilayer(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layers: List[int],
        system_prompt: Optional[str] = None,
        strategy: str = "assistant_token",
        num_generated_tokens: int = 10
    ) -> Dict[int, np.ndarray]:
        """Extract activations for a batch of prompts at multiple layers efficiently.

        Extracts all layers in a single forward pass per prompt.

        Args:
            model: StandardizedTransformer model
            tokenizer: Tokenizer
            prompts: List of prompt strings
            layers: List of layers to extract from
            system_prompt: Optional system prompt
            strategy: Activation extraction strategy
            num_generated_tokens: Number of tokens for generated_tokens_avg

        Returns:
            Dictionary mapping layer -> activations array [n_prompts, hidden_dim]
        """
        # Initialize dict to accumulate activations per layer
        layer_activations = {layer: [] for layer in layers}

        for prompt in prompts:
            acts_dict = extract_prompt_activations_all_layers(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=layers,
                system_prompt=system_prompt,
                strategy=strategy,
                num_generated_tokens=num_generated_tokens
            )
            for layer in layers:
                layer_activations[layer].append(acts_dict[layer])

        # Convert lists to arrays
        return {layer: np.array(acts) for layer, acts in layer_activations.items()}

    def extract_token_level(
        self,
        model,
        tokenizer,
        prompt: str,
        layers: List[int],
        system_prompt: Optional[str] = None,
        num_generated_tokens: int = 0,
        start_token_idx: Optional[int] = None,
        assistant_prefill: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[Dict[int, Dict[int, np.ndarray]], List[int]]:
        """
        Extract activations at EVERY token position for specified layers.

        This performs a single forward pass and extracts activations at each
        token position. Optionally supports generation to analyze emotions
        during model generation.

        Args:
            model: StandardizedTransformer model
            tokenizer: Tokenizer
            prompt: Single prompt to analyze
            layers: List of layers to extract from
            system_prompt: Optional system prompt
            num_generated_tokens: If > 0, generate this many tokens
            start_token_idx: Optional starting token index (None = auto-detect)
            assistant_prefill: Optional text to prefill assistant response
            temperature: Temperature for sampling (default: 1.0)
            top_p: Nucleus sampling parameter (default: 0.9)
            top_k: Top-k sampling parameter (default: 50)

        Returns:
            Tuple of (activations_by_token, token_ids):
              - activations_by_token: {token_pos: {layer: activation_vector}}
              - token_ids: List of token IDs for the entire sequence
        """
        # Apply chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Add assistant prefill if provided
        if assistant_prefill:
            formatted_prompt = formatted_prompt + assistant_prefill

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        input_ids = inputs['input_ids'].to(device)

        batch_size, seq_len = input_ids.shape
        token_ids = input_ids[0].tolist()

        # Auto-detect start token if not provided
        if start_token_idx is None:
            # Find first text token after <start_of_turn>user
            start_of_turn_id = tokenizer.convert_tokens_to_ids('<start_of_turn>')
            user_str = 'user'

            # Find the user turn
            for i in range(seq_len - 1):
                if input_ids[0, i].item() == start_of_turn_id:
                    # Check if next token is "user"
                    next_token = tokenizer.decode([input_ids[0, i+1].item()])
                    if user_str in next_token.lower():
                        # Skip <start_of_turn>, "user", and newline
                        start_token_idx = i + 3
                        break

            # Fallback: start from beginning
            if start_token_idx is None or start_token_idx >= seq_len:
                start_token_idx = 0

        # Extract activations at every token position (prefill)
        activations_by_token = {}

        # Save all layer outputs first (inside trace context)
        saved_outputs = {}
        with torch.no_grad():
            with model.trace(input_ids, scan=False):
                for layer in layers:
                    saved_outputs[layer] = model.layers_output[layer].save()

        # Now extract activations from saved outputs (outside trace context)
        for layer in layers:
            layer_output = saved_outputs[layer]
            # layer_output shape: [batch_size, seq_len, hidden_dim]
            for token_pos in range(start_token_idx, seq_len):
                if token_pos not in activations_by_token:
                    activations_by_token[token_pos] = {}
                activations_by_token[token_pos][layer] = layer_output[0, token_pos, :].detach().cpu().float().numpy().astype(np.float32)

        # Generation phase (if requested)
        if num_generated_tokens > 0:
            generated_ids = []

            for gen_step in range(num_generated_tokens):
                # Extract activations at current position
                current_token_pos = seq_len + gen_step
                activations_by_token[current_token_pos] = {}

                # Save layer outputs first
                gen_saved_outputs = {}
                with torch.no_grad():
                    with model.trace(input_ids, scan=False):
                        for layer in layers:
                            gen_saved_outputs[layer] = model.layers_output[layer].save()

                # Extract activations from saved outputs
                for layer in layers:
                    layer_output = gen_saved_outputs[layer]
                    activations_by_token[current_token_pos][layer] = layer_output[0, -1, :].detach().cpu().float().numpy().astype(np.float32)

                # Get logits for next token (use model.forward directly)
                with torch.no_grad():
                    outputs = model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]

                # Sample next token
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')

                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                generated_ids.append(next_token.item())
                token_ids.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check for EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break

        return activations_by_token, token_ids


# ============================================================================
# 2. ProbeInference - Manages probe loading and inference with caching
# ============================================================================

class ProbeInference:
    """Manages probe loading and inference with caching for efficiency."""

    def __init__(self, probe_dir: Path, cpca_path: Path, device: str = 'cuda', probe_pattern: str = None):
        """Initialize probe inference with probe directory and cPCA path.

        Args:
            probe_dir: Directory containing probe pickle files
            cpca_path: Path to cPCA .npz file (optional for linear probes without cPCA)
            device: Device for inference ('cuda' or 'cpu')
            probe_pattern: Custom probe filename pattern with {layer}, {n_components}, {seed} placeholders.
                          If None, uses default: "probe_layer{layer}_nc{n_components}_seed{seed}.pkl"
        """
        self.probe_dir = Path(probe_dir)
        self.cpca_path = Path(cpca_path) if cpca_path is not None else None
        self.device = device
        self.probe_pattern = probe_pattern  # Store custom pattern

        # Cache for loaded probes
        self.probe_cache = {}

        # Load cPCA data once (if provided)
        if cpca_path is not None:
            print(f"Loading cPCA data from {cpca_path}")
            cpca_data = np.load(cpca_path)
            self.cpca_components_all = cpca_data['components']  # [n_layers, n_components, hidden_dim]
            print(f"  Loaded cPCA components: {self.cpca_components_all.shape}")
        else:
            self.cpca_components_all = None
            print("No cPCA data loaded (linear probes without cPCA)")

    def load_probe(self, layer: int, n_components: int, seed: int) -> Dict:
        """Load probe with caching.

        Args:
            layer: Layer number
            n_components: Number of cPCA components
            seed: Random seed

        Returns:
            Probe dictionary with 'model', 'label_names', etc.
        """
        cache_key = (layer, n_components, seed)

        if cache_key not in self.probe_cache:
            # Use custom pattern if provided, otherwise use default
            if self.probe_pattern is not None:
                probe_filename = self.probe_pattern.format(
                    layer=layer,
                    n_components=n_components,
                    seed=seed
                )
            else:
                probe_filename = f"probe_layer{layer}_nc{n_components}_seed{seed}.pkl"

            probe_path = self.probe_dir / probe_filename
            print(f"Loading probe from {probe_path}")

            with open(probe_path, 'rb') as f:
                probe_dict = pickle.load(f)

            self.probe_cache[cache_key] = probe_dict
            print(f"  Cached probe (layer={layer}, nc={n_components}, seed={seed})")

        return self.probe_cache[cache_key]

    def get_cpca_components(self, layer: int, n_components: int) -> np.ndarray:
        """Get cPCA components for a specific layer.

        Args:
            layer: Layer number
            n_components: Number of components to use

        Returns:
            cPCA components [n_components, hidden_dim]
        """
        if self.cpca_components_all is None:
            raise ValueError("cPCA components not loaded. ProbeInference was initialized without cpca_path.")

        if layer >= self.cpca_components_all.shape[0]:
            raise ValueError(f"Layer {layer} not found in cPCA data (max: {self.cpca_components_all.shape[0]-1})")

        return self.cpca_components_all[layer, :n_components, :]

    def predict(
        self,
        activations: np.ndarray,
        layer: int,
        n_components: int = 10,
        seed: int = 0,
        drop_neutral: bool = True
    ) -> np.ndarray:
        """Run probe inference on activations.

        Pipeline: cPCA projection → probe inference → drop neutral class

        Args:
            activations: Raw activations [n_samples, hidden_dim]
            layer: Layer number (for loading probe and cPCA)
            n_components: Number of cPCA components
            seed: Probe random seed
            drop_neutral: Whether to drop neutral class (index 6)

        Returns:
            Emotion logits [n_samples, 6] if drop_neutral else [n_samples, 7]
        """
        # Load probe
        probe_dict = self.load_probe(layer, n_components, seed)

        # Check if probe uses cPCA
        use_cpca = probe_dict.get('use_cpca', n_components > 0)

        if use_cpca:
            # Get cPCA components
            cpca_components = self.get_cpca_components(layer, n_components)

            # Validate dimensions
            if activations.shape[1] != cpca_components.shape[1]:
                raise ValueError(
                    f"Dimension mismatch: activations have {activations.shape[1]} dims, "
                    f"cPCA expects {cpca_components.shape[1]} dims"
                )

            # Step 1: Apply cPCA projection
            projected = activations @ cpca_components.T  # [n_samples, n_components]
        else:
            # Use raw activations directly
            projected = activations

        # Step 2: Run probe inference
        probe_model = probe_dict['model']
        probe_model.eval()
        probe_model = probe_model.to(self.device)

        X_tensor = torch.from_numpy(projected).float().to(self.device)

        with torch.no_grad():
            logits = probe_model(X_tensor)  # [n_samples, 7]

        logits = logits.cpu().numpy()

        # Step 3: Drop neutral class if requested
        if drop_neutral:
            logits = logits[:, :6]  # Drop index 6 (neutral)

        return logits

    def predict_batch(
        self,
        activations_by_layer: Dict[int, np.ndarray],
        n_components: int = 10,
        seed: int = 0,
        drop_neutral: bool = True
    ) -> Dict[int, np.ndarray]:
        """Run probe inference on multiple layers.

        Args:
            activations_by_layer: Dict mapping layer -> activations [n_samples, hidden_dim]
            n_components: Number of cPCA components
            seed: Probe random seed
            drop_neutral: Whether to drop neutral class

        Returns:
            Dict mapping layer -> logits [n_samples, 6 or 7]
        """
        results = {}
        for layer, activations in activations_by_layer.items():
            results[layer] = self.predict(
                activations, layer, n_components, seed, drop_neutral
            )
        return results

    def load_orthogonal_probe(
        self,
        layer: int,
        representation: str = "raw",
        n_components: Optional[int] = None,
        orthogonality_weight: float = 1000.0
    ) -> Dict:
        """Load orthogonal user/assistant probes with caching.

        Args:
            layer: Layer number
            representation: 'raw', 'global_cpca_top10', 'global_cpca_top20', etc.
            n_components: Number of cPCA components (for backward compatibility)
            orthogonality_weight: Orthogonality weight used during training

        Returns:
            Dict with 'user_probes', 'assistant_probes', and metadata
        """
        # Build cache key
        cache_key = f"orthogonal_{layer}_{representation}_{orthogonality_weight}"

        # Check cache first
        if cache_key in self.probe_cache:
            return self.probe_cache[cache_key]

        # Build probe filename
        if representation == "raw":
            probe_filename = f"probe_layer{layer}_raw_ortho{orthogonality_weight}.pkl"
        else:
            # For cpca representations, use the representation string directly
            # e.g., "global_cpca_top10" -> "probe_layer20_global_cpca_top10_ortho1000.0.pkl"
            probe_filename = f"probe_layer{layer}_{representation}_ortho{orthogonality_weight}.pkl"

        # Check in orthogonal subdirectory
        ortho_dir = self.probe_dir / "orthogonal" / f"ortho_{orthogonality_weight}"
        probe_path = ortho_dir / probe_filename

        if not probe_path.exists():
            raise FileNotFoundError(f"Orthogonal probe not found at {probe_path}")

        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)

        # Cache the result
        self.probe_cache[cache_key] = probe_data

        return probe_data

    def predict_orthogonal(
        self,
        activations: np.ndarray,
        user_probes: np.ndarray,
        asst_probes: np.ndarray,
        emotions: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply orthogonal user and assistant probes to activations.

        Args:
            activations: Activations [hidden_dim] for single position, or [n_samples, hidden_dim]
            user_probes: User probe directions [n_emotions, hidden_dim]
            asst_probes: Assistant probe directions [n_emotions, hidden_dim]
            emotions: List of emotion names

        Returns:
            If single activation: (user_scores, asst_scores) as dicts
            If batch: (user_scores_list, asst_scores_list) as lists of dicts
        """
        # Handle single activation vs batch
        if activations.ndim == 1:
            activations = activations[np.newaxis, :]  # [1, hidden_dim]
            single_mode = True
        else:
            single_mode = False

        # Project onto probe directions (simple dot product)
        # Probes are already normalized from training
        user_proj = activations @ user_probes.T  # [n_samples, n_emotions]
        asst_proj = activations @ asst_probes.T  # [n_samples, n_emotions]

        if single_mode:
            # Return single dicts
            user_scores = {emotions[i]: float(user_proj[0, i]) for i in range(len(emotions))}
            asst_scores = {emotions[i]: float(asst_proj[0, i]) for i in range(len(emotions))}
            return user_scores, asst_scores
        else:
            # Return lists of dicts
            user_scores_list = [
                {emotions[i]: float(user_proj[j, i]) for i in range(len(emotions))}
                for j in range(activations.shape[0])
            ]
            asst_scores_list = [
                {emotions[i]: float(asst_proj[j, i]) for i in range(len(emotions))}
                for j in range(activations.shape[0])
            ]
            return user_scores_list, asst_scores_list

    def load_centroid_probe(
        self,
        layer: int,
        k_value: int,
        orthogonality_weight: float = 1000.0,
        constraint_type: Optional[str] = None,
        probe_format: str = "auto"
    ) -> Dict:
        """Load K-set probes and compute centroids automatically.

        Args:
            layer: Layer number
            k_value: Number of probe sets (K)
            orthogonality_weight: Orthogonality weight used during training
            constraint_type: Optional constraint type (e.g., "gramschmidt")
            probe_format: "auto", "text", or "conversation"
                - "auto": Auto-detect based on probe file structure
                - "text": Text-based probes (single role) [K, 6, hidden_dim]
                - "conversation": Conversation-based probes (user/assistant) [K, 2, 6, hidden_dim]

        Returns:
            Dict with:
                - 'centroid_probes': np.ndarray for text format [6, hidden_dim]
                - 'centroid_user_probes': np.ndarray for conversation [6, hidden_dim]
                - 'centroid_asst_probes': np.ndarray for conversation [6, hidden_dim]
                - 'label_names': List of emotion names
                - 'k_value': Number of sets averaged
                - 'probe_format': "text" or "conversation"
                - 'all_probe_sets': Original K-set probes (for debugging)
                - 'layer': Layer number
        """
        # Check cache first
        cache_key = ('centroid', layer, k_value, orthogonality_weight, constraint_type, probe_format)
        if cache_key in self.probe_cache:
            return self.probe_cache[cache_key]

        # Build probe filename
        constraint_suffix = f"_{constraint_type}" if constraint_type else ""
        probe_filename = f"probe_k{k_value}_layer{layer}_ortho{orthogonality_weight}{constraint_suffix}.pkl"

        # Auto-detect format if needed
        if probe_format == "auto":
            # Try multi_orthogonal subdirectory
            multi_ortho_dir = self.probe_dir / "multi_orthogonal"
            probe_path = multi_ortho_dir / probe_filename

            if not probe_path.exists():
                # Try parent directory directly
                probe_path = self.probe_dir / probe_filename

            if not probe_path.exists():
                raise FileNotFoundError(
                    f"K-set probe not found: {probe_filename}\n"
                    f"Searched in:\n"
                    f"  - {multi_ortho_dir}\n"
                    f"  - {self.probe_dir}\n"
                    f"Required: k={k_value}, layer={layer}, ortho={orthogonality_weight}"
                )

            # Detect format from file structure
            with open(probe_path, 'rb') as f:
                temp_data = pickle.load(f)
            probe_format = "conversation" if "probe_sets" in temp_data else "text"
        else:
            # Use explicit format
            multi_ortho_dir = self.probe_dir / "multi_orthogonal"
            probe_path = multi_ortho_dir / probe_filename

            if not probe_path.exists():
                probe_path = self.probe_dir / probe_filename

            if not probe_path.exists():
                raise FileNotFoundError(
                    f"K-set probe not found: {probe_filename}\n"
                    f"Format: {probe_format}\n"
                    f"Required: k={k_value}, layer={layer}, ortho={orthogonality_weight}"
                )

        # Load probe file
        print(f"Loading centroid probe from {probe_path.name}")
        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)

        # Extract probe sets and compute centroids
        result = {
            'k_value': k_value,
            'probe_format': probe_format,
            'layer': layer,
            'label_names': probe_data.get('label_names', ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])
        }

        if probe_format == "text":
            # Text-based: [K, 6, hidden_dim]
            all_probe_sets = probe_data['all_probe_sets']

            # Compute centroid: average across K sets
            centroid_probes = np.mean(all_probe_sets, axis=0)  # [6, hidden_dim]

            result['centroid_probes'] = centroid_probes
            result['all_probe_sets'] = all_probe_sets

            print(f"  ✓ Text-based centroid: {all_probe_sets.shape} → {centroid_probes.shape}")

        elif probe_format == "conversation":
            # Conversation-based: [K, 2, 6, hidden_dim]
            probe_sets = probe_data['probe_sets']

            # Compute centroids: average across K sets
            # User probes: axis 1 index 0
            # Assistant probes: axis 1 index 1
            centroid_user_probes = np.mean(probe_sets[:, 0, :, :], axis=0)  # [6, hidden_dim]
            centroid_asst_probes = np.mean(probe_sets[:, 1, :, :], axis=0)  # [6, hidden_dim]

            result['centroid_user_probes'] = centroid_user_probes
            result['centroid_asst_probes'] = centroid_asst_probes
            result['all_probe_sets'] = probe_sets

            print(f"  ✓ Conversation-based centroid: {probe_sets.shape} → user/asst {centroid_user_probes.shape}")

        # Cache result
        self.probe_cache[cache_key] = result

        return result

    def predict_centroid(
        self,
        activations: np.ndarray,
        centroid_probes: Optional[np.ndarray],
        centroid_user_probes: Optional[np.ndarray],
        centroid_asst_probes: Optional[np.ndarray],
        emotions: List[str],
        probe_format: str
    ) -> Tuple:
        """Apply centroid probes to activations.

        Args:
            activations: Activations [hidden_dim] for single position, or [n_samples, hidden_dim]
            centroid_probes: Centroid probe directions [n_emotions, hidden_dim] (text format)
            centroid_user_probes: User centroid directions [n_emotions, hidden_dim] (conversation)
            centroid_asst_probes: Assistant centroid directions [n_emotions, hidden_dim] (conversation)
            emotions: List of emotion names
            probe_format: "text" or "conversation"

        Returns:
            If text format:
                - Single activation: scores dict {emotion: float}
                - Batch: scores array [n_samples, n_emotions]
            If conversation format:
                - Single activation: (user_scores dict, asst_scores dict, avg_scores dict)
                - Batch: (user_scores list, asst_scores list, avg_scores array)
        """
        # Handle single activation vs batch
        if activations.ndim == 1:
            activations = activations[np.newaxis, :]  # [1, hidden_dim]
            single_mode = True
        else:
            single_mode = False

        if probe_format == "text":
            # Validate
            if centroid_probes is None:
                raise ValueError("centroid_probes must be provided for text format")

            expected_dim = centroid_probes.shape[1]
            if activations.shape[-1] != expected_dim:
                raise ValueError(
                    f"Activation dimension mismatch: got {activations.shape[-1]}, "
                    f"expected {expected_dim} (probe hidden_dim)"
                )

            # Simple dot product projection
            # Probes are already normalized from training
            scores = activations @ centroid_probes.T  # [n_samples, n_emotions]

            if single_mode:
                return {emotions[i]: float(scores[0, i]) for i in range(len(emotions))}
            else:
                return scores

        elif probe_format == "conversation":
            # Validate
            if centroid_user_probes is None or centroid_asst_probes is None:
                raise ValueError("centroid_user_probes and centroid_asst_probes must be provided for conversation format")

            expected_dim = centroid_user_probes.shape[1]
            if activations.shape[-1] != expected_dim:
                raise ValueError(
                    f"Activation dimension mismatch: got {activations.shape[-1]}, "
                    f"expected {expected_dim} (probe hidden_dim)"
                )

            # Project onto both user and assistant centroids
            user_proj = activations @ centroid_user_probes.T  # [n_samples, n_emotions]
            asst_proj = activations @ centroid_asst_probes.T  # [n_samples, n_emotions]

            # Average user and assistant
            avg_scores = (user_proj + asst_proj) / 2

            if single_mode:
                user_scores = {emotions[i]: float(user_proj[0, i]) for i in range(len(emotions))}
                asst_scores = {emotions[i]: float(asst_proj[0, i]) for i in range(len(emotions))}
                avg_scores_dict = {emotions[i]: float(avg_scores[0, i]) for i in range(len(emotions))}
                return user_scores, asst_scores, avg_scores_dict
            else:
                user_scores_list = [
                    {emotions[i]: float(user_proj[j, i]) for i in range(len(emotions))}
                    for j in range(activations.shape[0])
                ]
                asst_scores_list = [
                    {emotions[i]: float(asst_proj[j, i]) for i in range(len(emotions))}
                    for j in range(activations.shape[0])
                ]
                return user_scores_list, asst_scores_list, avg_scores
        else:
            raise ValueError(f"Invalid probe_format: {probe_format}. Expected 'text' or 'conversation'")


# ============================================================================
# 3. Probe Score Normalization Utilities
# ============================================================================

def normalize_probe_scores_zscore(
    scores: np.ndarray,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Apply z-score normalization to probe scores using baseline statistics.

    This function provides a unified implementation of probe score normalization
    used across token-level, model-diff, and dashboard preprocessing pipelines.

    Args:
        scores: Probe scores to normalize [n_emotions] or dict with 'user'/'assistant' keys
        baseline_mean: Baseline mean scores [n_emotions]
        baseline_std: Baseline standard deviation [n_emotions]
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized scores in same format as input:
        - If scores is array: returns array [n_emotions]
        - If scores is dict: returns dict with 'user' and 'assistant' keys
    """
    if isinstance(scores, dict) and 'user' in scores:
        # Orthogonal/centroid conversation probes - normalize both user and assistant
        return {
            'user': (scores['user'] - baseline_mean) / (baseline_std + epsilon),
            'assistant': (scores['assistant'] - baseline_mean) / (baseline_std + epsilon)
        }
    else:
        # Standard array format
        return (scores - baseline_mean) / (baseline_std + epsilon)


def normalize_probe_scores_center(
    scores: np.ndarray,
    baseline_mean: np.ndarray
) -> np.ndarray:
    """
    DEPRECATED: Apply centering (mean subtraction only) to probe scores using baseline statistics.

    This function is deprecated and kept only for backward compatibility.
    All probe applications now use z-score normalization (normalize_probe_scores_zscore) instead.

    Args:
        scores: Probe scores to center [n_emotions] or dict with 'user'/'assistant' keys
        baseline_mean: Baseline mean scores [n_emotions]

    Returns:
        Centered scores in same format as input
    """
    import warnings
    warnings.warn(
        "normalize_probe_scores_center is deprecated. "
        "All probe applications now use z-score normalization (normalize_probe_scores_zscore).",
        DeprecationWarning,
        stacklevel=2
    )

    if isinstance(scores, dict) and 'user' in scores:
        # Orthogonal/centroid conversation probes - center both user and assistant
        return {
            'user': scores['user'] - baseline_mean,
            'assistant': scores['assistant'] - baseline_mean
        }
    else:
        # Standard array format
        return scores - baseline_mean


# ============================================================================
# 4. ProbeAggregator - Handles aggregation and differencing operations
# ============================================================================

class ProbeAggregator:
    """Handles aggregation and differencing operations for probe scores."""

    @staticmethod
    def compute_diff(
        scores_a: np.ndarray,
        scores_b: np.ndarray
    ) -> np.ndarray:
        """Compute simple difference between two score arrays.

        Args:
            scores_a: First score array [n_samples, n_emotions]
            scores_b: Second score array [n_samples, n_emotions]

        Returns:
            Difference scores_a - scores_b [n_samples, n_emotions]
        """
        return scores_a - scores_b

    @staticmethod
    def compute_double_diff(
        ft_dataset: np.ndarray,
        base_dataset: np.ndarray,
        ft_baseline: np.ndarray,
        base_baseline: np.ndarray,
        emotions: List[str],
        n_bootstrap: int = 100,
        ci_percentile: float = 95.0
    ) -> Dict:
        """Compute case 4 double difference with bootstrap CI.

        Formula: (ft_dataset - base_dataset) - (ft_baseline - base_baseline)

        Args:
            ft_dataset: Finetuned model on dataset prompts [n_pairs, n_emotions]
            base_dataset: Base model on dataset prompts [n_pairs, n_emotions]
            ft_baseline: Finetuned model on baseline prompts [n_pairs, n_emotions]
            base_baseline: Base model on baseline prompts [n_pairs, n_emotions]
            emotions: List of emotion names
            n_bootstrap: Number of bootstrap samples
            ci_percentile: Confidence interval percentile

        Returns:
            Dict with 'mean', 'bootstrap_ci', 'per_prompt_diffs', 'emotion_ranking'
        """
        # Compute double diff
        D_ft = ft_dataset - ft_baseline  # Finetuned effect
        D_base = base_dataset - base_baseline  # Base effect
        DD = D_ft - D_base  # Double diff

        n_pairs = DD.shape[0]

        # Mean across prompts
        mean_effect = {emotions[j]: float(np.mean(DD[:, j])) for j in range(len(emotions))}

        # Per-pair effects
        per_pair_effects = [
            {emotions[j]: float(DD[i, j]) for j in range(len(emotions))}
            for i in range(n_pairs)
        ]

        # Bootstrap confidence intervals
        bootstrap_ci = ProbeAggregator.bootstrap_ci(
            per_pair_effects, emotions, n_bootstrap, ci_percentile
        )

        # Emotion ranking by absolute mean
        emotion_ranking = sorted(
            [(emotion, abs(mean_effect[emotion])) for emotion in emotions],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'mean_effect': mean_effect,
            'bootstrap_ci': bootstrap_ci,
            'per_pair_effects': per_pair_effects,
            'emotion_ranking': emotion_ranking,
        }

    @staticmethod
    def bootstrap_ci(
        per_pair_effects: List[Dict[str, float]],
        emotions: List[str],
        n_bootstrap: int = 100,
        ci_percentile: float = 95.0
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for probe scores.

        Args:
            per_pair_effects: List of dicts, one per prompt pair, with emotion scores
            emotions: List of emotion names
            n_bootstrap: Number of bootstrap samples
            ci_percentile: Confidence interval percentile

        Returns:
            Dict mapping emotion → {'lower': float, 'upper': float}
        """
        n_pairs = len(per_pair_effects)

        # Convert to array for easy resampling
        effects_array = np.array([
            [pair_dict[emotion] for emotion in emotions]
            for pair_dict in per_pair_effects
        ])  # [n_pairs, n_emotions]

        bootstrap_means = []

        # Bootstrap resampling
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # Resample pairs with replacement
            indices = rng.choice(n_pairs, size=n_pairs, replace=True)
            resampled = effects_array[indices]

            # Compute mean effect for this resample
            bootstrap_mean = resampled.mean(axis=0)  # [n_emotions]
            bootstrap_means.append(bootstrap_mean)

        bootstrap_means = np.array(bootstrap_means)  # [n_bootstrap, n_emotions]

        # Compute percentile-based confidence intervals
        lower_p = (100 - ci_percentile) / 2
        upper_p = 100 - lower_p

        ci_dict = {}
        for i, emotion in enumerate(emotions):
            lower = np.percentile(bootstrap_means[:, i], lower_p)
            upper = np.percentile(bootstrap_means[:, i], upper_p)
            ci_dict[emotion] = {'lower': float(lower), 'upper': float(upper)}

        return ci_dict


# ============================================================================
# 4. ProbeVisualizer - Creates visualizations
# ============================================================================

class ProbeVisualizer:
    """Creates visualizations for probe experiment results."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import visualization functions
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from visualization.plot_probe_results import (
            plot_multilayer_results,
            plot_singlelayer_results,
            EMOTION_COLORS
        )
        self._plot_multilayer = plot_multilayer_results
        self._plot_singlelayer = plot_singlelayer_results
        self.emotion_colors = EMOTION_COLORS

    def plot_results(
        self,
        results: Dict,
        experiment_name: str = "probe_experiment",
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot probe experiment results (dispatches to multilayer or singlelayer).

        Args:
            results: Results dictionary
            experiment_name: Name prefix for plot files
            layer_range: (min_layer, max_layer) for multilayer plots
        """
        if results.get('multilayer', False):
            self._plot_multilayer(results, self.output_dir, experiment_name, layer_range)
        else:
            self._plot_singlelayer(results, self.output_dir, experiment_name)

    def plot_heatmap(
        self,
        results: Dict,
        experiment_name: str = "probe_experiment",
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot only the heatmap from multilayer results.

        Args:
            results: Results dictionary with 'results_by_layer'
            experiment_name: Name prefix for plot file
            layer_range: (min_layer, max_layer) to restrict plot range
        """
        # Just call the full function - it generates both plots
        # Could be split in the future if needed
        self._plot_multilayer(results, self.output_dir, experiment_name, layer_range)

    def plot_trajectories(
        self,
        results: Dict,
        experiment_name: str = "probe_experiment",
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot only the trajectories from multilayer results.

        Args:
            results: Results dictionary with 'results_by_layer'
            experiment_name: Name prefix for plot file
            layer_range: (min_layer, max_layer) to restrict plot range
        """
        # Just call the full function - it generates both plots
        # Could be split in the future if needed
        self._plot_multilayer(results, self.output_dir, experiment_name, layer_range)
