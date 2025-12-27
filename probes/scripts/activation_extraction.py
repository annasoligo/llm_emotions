#!/usr/bin/env python3
"""
Standalone activation extraction module for probe experiments.

This module provides all necessary functions to extract activations from models
without depending on the emotion_evals/emo_lens codebase.

Ported from: believe-it-or-not/emotion_evals/emo_lens/model_utils.py
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
from nnterp import StandardizedTransformer


def get_chat_special_tokens(tokenizer: Any) -> Dict[str, List[int]]:
    """
    Get special tokens used for chat formatting.

    Args:
        tokenizer: Tokenizer

    Returns:
        Dict with 'end_of_turn' and 'start_of_turn' token ID lists
    """
    special_tokens = {
        'end_of_turn': [],
        'start_of_turn': []
    }

    # Check for common end-of-turn markers
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for token in ['<|im_end|>', '<end_of_turn>', '</s>']:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                special_tokens['end_of_turn'].append(token_id)

    # Add EOS token if available
    if tokenizer.eos_token_id is not None:
        special_tokens['end_of_turn'].append(tokenizer.eos_token_id)

    # Check for common start-of-turn markers
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        for token in ['<|im_start|>', '<start_of_turn>']:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                special_tokens['start_of_turn'].append(token_id)

    return special_tokens


def extract_prompt_activations_all_layers(
    model: StandardizedTransformer,
    tokenizer: Any,
    prompt: str,
    layers: List[int],
    system_prompt: Optional[str] = None,
    strategy: str = "assistant_token",
    num_generated_tokens: int = 10
) -> Dict[int, np.ndarray]:
    """
    Extract activation vectors for a single prompt at ALL specified layers in ONE forward pass.

    This is MUCH more efficient than calling extract_activations_at_layer() in a loop.

    For all strategies:
    - "assistant_token", "last_user_token", "between_turns_avg": Single forward pass extracts all layers
    - "generated_tokens_avg": Single generation extracts all layers from the same generated tokens

    Args:
        model: StandardizedTransformer model
        tokenizer: Tokenizer
        prompt: Input prompt text
        layers: List of layer indices (0-indexed)
        system_prompt: Optional system prompt
        strategy: Activation collection strategy:
            - "assistant_token": Extract at the "model" token position (default)
            - "last_user_token": Extract at last token before assistant turn
            - "between_turns_avg": Average over turn boundary tokens
            - "generated_tokens_avg": Average over generated tokens
        num_generated_tokens: Number of tokens to generate (only for "generated_tokens_avg")

    Returns:
        Dict mapping layer_idx -> activation vector [hidden_dim]
    """
    # Build chat-style prompt
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    # Format prompt
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            raise ValueError("Failed to format prompt")

    # For strategies that don't require generation, we can extract all layers in one pass
    if strategy != "generated_tokens_avg":
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)

        # Determine token position based on strategy
        if strategy == "assistant_token":
            # Extract at "model" token position (matches baseline collection)
            model_token_id = tokenizer.convert_tokens_to_ids('model')

            # Search backwards for the model token
            token_position = None
            for i in range(input_ids.shape[1] - 1, -1, -1):
                if input_ids[0, i].item() == model_token_id:
                    token_position = i
                    break

            if token_position is None:
                # Fallback: use last token
                token_position = input_ids.shape[1] - 1

            # Extract all layers in one forward pass
            saved_outputs = {}
            with model.trace(input_ids, scan=False):
                for layer in layers:
                    saved_outputs[layer] = model.layers_output[layer].save()

            # Extract activations at token_position for all layers
            activations_by_layer = {}
            for layer in layers:
                hidden_states = saved_outputs[layer]
                activations_by_layer[layer] = hidden_states[0, token_position, :].detach().cpu().float().numpy().astype(np.float32)

            return activations_by_layer

        elif strategy == "last_user_token":
            # Find last token before assistant turn
            end_tokens = get_chat_special_tokens(tokenizer)['end_of_turn']
            token_position = None
            for i in range(input_ids.shape[1] - 1, -1, -1):
                if input_ids[0, i].item() in end_tokens:
                    token_position = i
                    break
            if token_position is None:
                token_position = input_ids.shape[1] - 2

            # Extract all layers in one forward pass
            saved_outputs = {}
            with model.trace(input_ids, scan=False):
                for layer in layers:
                    saved_outputs[layer] = model.layers_output[layer].save()

            # Extract activations at token_position for all layers
            activations_by_layer = {}
            for layer in layers:
                hidden_states = saved_outputs[layer]
                activations_by_layer[layer] = hidden_states[0, token_position, :].detach().cpu().float().numpy().astype(np.float32)

            return activations_by_layer

        elif strategy == "between_turns_avg":
            # Average over ALL boundary tokens (INCLUDING all markers)
            # From <end_of_turn> through model\n (matches baseline)
            special_tokens = get_chat_special_tokens(tokenizer)
            end_tokens = special_tokens['end_of_turn']
            start_tokens = special_tokens['start_of_turn']
            model_token_id = tokenizer.convert_tokens_to_ids('model')
            newline_id = tokenizer.convert_tokens_to_ids('\n')

            # Find positions
            end_pos = None  # <end_of_turn> position
            start_pos = None  # <start_of_turn> position
            model_pos = None  # "model" token position
            for i in range(input_ids.shape[1] - 1, -1, -1):
                token_id = input_ids[0, i].item()
                if model_pos is None and token_id == model_token_id:
                    model_pos = i
                if start_pos is None and token_id in start_tokens:
                    start_pos = i
                if end_pos is None and token_id in end_tokens:
                    end_pos = i
                if end_pos is not None and start_pos is not None and model_pos is not None:
                    break

            # Extract activations and average
            saved_outputs = {}
            with model.trace(input_ids, scan=False):
                for layer in layers:
                    saved_outputs[layer] = model.layers_output[layer].save()

            # Now access the tensors directly AFTER the trace has executed
            activations_by_layer = {}
            for layer in layers:
                hidden_states = saved_outputs[layer]
                if end_pos is not None and model_pos is not None:
                    # Include from <end_of_turn> through model and \n after (if present)
                    boundary_end = model_pos + 1  # Include "model"
                    if model_pos + 1 < input_ids.shape[1] and input_ids[0, model_pos + 1].item() == newline_id:
                        boundary_end = model_pos + 2  # Include the \n
                    activations_by_layer[layer] = hidden_states[0, end_pos:boundary_end, :].detach().cpu().float().numpy().mean(axis=0).astype(np.float32)
                else:
                    activations_by_layer[layer] = hidden_states[0, -1, :].detach().cpu().float().numpy().astype(np.float32)

            return activations_by_layer
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    else:
        # For generated_tokens_avg, generate once and extract all layers
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs_device = {k: v.to(device) for k, v in inputs.items()}

        # Generate tokens once with sampling to avoid degenerate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs_device,
                max_new_tokens=num_generated_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                top_k=50,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # Extract hidden states for all layers from generated tokens
        activations_by_layer = {layer: [] for layer in layers}

        for step_idx in range(min(num_generated_tokens, len(outputs.hidden_states))):
            step_hidden = outputs.hidden_states[step_idx]  # Tuple of layer tensors
            for layer in layers:
                layer_hidden = step_hidden[layer]
                activation = layer_hidden[0, -1, :].detach().cpu().float().numpy()
                activations_by_layer[layer].append(activation)

        # Average over generated tokens for each layer
        for layer in layers:
            if activations_by_layer[layer]:
                activations_by_layer[layer] = np.mean(activations_by_layer[layer], axis=0).astype(np.float32)
            else:
                # Fallback to last token of prompt
                with model.trace(inputs_device, scan=False):
                    layer_output = model.layers_output[layer].save()
                activations_by_layer[layer] = layer_output[0, -1, :].detach().cpu().float().numpy().astype(np.float32)

        return activations_by_layer
