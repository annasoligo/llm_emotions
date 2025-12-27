#!/usr/bin/env python3
"""Utilities for evaluating emotion probes on conversation data."""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

from probes.core import load_jsonl


def load_conversations(jsonl_path: Path, limit: int = None) -> List[Dict]:
    """Load conversation data from JSONL.

    DEPRECATED: Use probes.core.load_jsonl() directly instead.
    This wrapper is kept for backward compatibility.
    """
    # Use centralized loading function with require_id=False for backward compatibility
    return load_jsonl(jsonl_path, limit=limit, require_id=False)


def extract_first_two_turns(conversation: Dict) -> Tuple[str, str, str, str]:
    """Extract first 2 turns from conversation.

    Returns:
        user_text: First user message
        asst_text: First assistant message
        user_emotion: Expected user emotion
        asst_emotion: Expected assistant emotion
    """
    messages = conversation['messages']
    if len(messages) < 2:
        raise ValueError(f"Conversation has fewer than 2 turns")

    user_text = messages[0]['content']
    asst_text = messages[1]['content']
    user_emotion = conversation['user_emotion']
    asst_emotion = conversation['asst_emotion']

    return user_text, asst_text, user_emotion, asst_emotion


def get_turn_activation_avg_from_token(
    model: StandardizedTransformer,
    tokenizer,
    text: str,
    layer: int,
    start_token: int = 20,
) -> np.ndarray:
    """Get activation averaged over tokens starting from a specific token index.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        text: Text to process
        layer: Layer index
        start_token: Token index to start averaging from (default: 20)

    Returns:
        activation: [hidden_dim] averaged activation
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    # Move to device
    device = next(model.model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get sequence length
    seq_len = inputs['input_ids'].shape[1]

    # Collect activations
    with torch.no_grad():
        with model.trace(inputs, scan=False):
            layer_output = model.layers_output[layer].save()

    # Average from start_token onwards (or from start if seq_len < start_token)
    start_idx = min(start_token, seq_len)
    activations_to_avg = layer_output[0, start_idx:, :]  # [num_tokens, hidden_dim]
    avg_activation = activations_to_avg.mean(dim=0)  # [hidden_dim]

    # Convert to numpy
    return avg_activation.detach().cpu().to(torch.float32).numpy()


def load_probe(probe_path: Path) -> Dict:
    """Load a trained probe."""
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    return probe_data


def predict_emotion(activation: np.ndarray, probe: Dict) -> Tuple[str, np.ndarray]:
    """Predict emotion from activation using trained probe.

    Args:
        activation: [hidden_dim] or [n_components] activation
        probe: Loaded probe dict with 'model' (PyTorch Linear) and optionally 'cpca_components'

    Returns:
        predicted_emotion: String emotion label
        probabilities: [n_classes] probability distribution
    """
    # Project onto cPCA if components exist
    if 'cpca_components' in probe:
        cpca_components = probe['cpca_components']
        activation = activation @ cpca_components.T

    # Get probe model (PyTorch Linear layer)
    model = probe['model']

    # Move model to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Convert to tensor and predict
    activation_tensor = torch.from_numpy(activation).float().unsqueeze(0).to(device)  # [1, hidden_dim]

    with torch.no_grad():
        logits = model(activation_tensor)  # [1, n_classes]
        probabilities = torch.softmax(logits, dim=1)[0]  # [n_classes]
        pred_idx = torch.argmax(probabilities).item()

    # Convert to numpy
    probabilities = probabilities.cpu().numpy()

    # Get emotion name
    label_names = probe['label_names']
    predicted_emotion = label_names[pred_idx]

    return predicted_emotion, probabilities


def load_model_and_tokenizer(model_name: str) -> Tuple[StandardizedTransformer, AutoTokenizer]:
    """Load model and tokenizer with proper configuration.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        model: Loaded StandardizedTransformer
        tokenizer: Loaded tokenizer
    """
    print(f"\nLoading model: {model_name}")

    # Pre-load with AutoModelForCausalLM (avoids meta device issue)
    print("Loading with AutoModelForCausalLM...")
    model_raw = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Wrap in StandardizedTransformer
    print("Wrapping in StandardizedTransformer...")
    model = StandardizedTransformer(
        model_raw,
        trust_remote_code=True,
        check_renaming=False,
        allow_dispatch=True
    )

    device = next(model.model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_probes_from_directory(
    probe_dir: Path,
    layers: List[int],
    probe_pattern: str = "probe_layer{layer}_all_cpca.pkl",
) -> Dict[int, Dict]:
    """Load multiple probes from a directory.

    Args:
        probe_dir: Directory containing probe files
        layers: List of layer indices to load
        probe_pattern: Filename pattern with {layer} placeholder

    Returns:
        Dict mapping layer index to probe dict
    """
    probes = {}

    # Load first probe to check for cPCA path
    cpca_components_by_layer = {}

    for layer in layers:
        probe_name = probe_pattern.format(layer=layer)
        probe_path = probe_dir / probe_name

        if not probe_path.exists():
            print(f"Warning: Probe not found at {probe_path}, skipping layer {layer}")
            continue

        probe = load_probe(probe_path)

        # Check if probe needs cPCA components
        if 'cpca_results_path' in probe and 'cpca_components' not in probe:
            cpca_path = probe['cpca_results_path']

            # Load cPCA components if not already loaded for this layer
            if layer not in cpca_components_by_layer:
                try:
                    # Try relative to repo root
                    from pathlib import Path as P
                    cpca_file = P('/workspace-vast/annas/git/research-tools') / cpca_path
                    if not cpca_file.exists():
                        # Try absolute
                        cpca_file = P(cpca_path)

                    cpca_data = np.load(cpca_file)
                    # cPCA results have shape [num_layers, n_components, hidden_dim]
                    if 'components' in cpca_data:
                        all_components = cpca_data['components']
                        # Index by layer: components[layer] gives [n_components, hidden_dim]
                        if layer < all_components.shape[0]:
                            cpca_components_by_layer[layer] = all_components[layer]  # [50, 5376]
                        else:
                            print(f"Warning: Layer {layer} out of range for cPCA components (max: {all_components.shape[0]-1})")
                    else:
                        print(f"Warning: Could not find 'components' in {cpca_file}")
                        print(f"Available keys: {list(cpca_data.keys())}")
                except Exception as e:
                    print(f"Warning: Could not load cPCA components from {cpca_path}: {e}")

            # Add cPCA components to probe
            if layer in cpca_components_by_layer:
                # Get the probe's expected input dimension
                probe_input_dim = probe['model'].in_features
                all_cpca_components = cpca_components_by_layer[layer]  # [50, 5376]

                # Slice to match probe's expected input (top-k PCs)
                if probe_input_dim < all_cpca_components.shape[0]:
                    probe['cpca_components'] = all_cpca_components[:probe_input_dim]  # [k, 5376]
                    print(f"    Using top-{probe_input_dim} cPCA components")
                else:
                    probe['cpca_components'] = all_cpca_components

        probes[layer] = probe
        print(f"  Layer {layer}: {probe_path.name}")

    return probes


def evaluate_conversations(
    conversations: List[Dict],
    model: StandardizedTransformer,
    tokenizer,
    probes: Dict[int, Dict],
    start_token: int = 20,
    probe_name: str = "probes",
) -> Dict:
    """Evaluate probes on conversation data.

    Args:
        conversations: List of conversation dicts
        model: Loaded model
        tokenizer: Tokenizer
        probes: Dict mapping layer index to probe
        start_token: Token index to start averaging from
        probe_name: Name for this probe configuration

    Returns:
        results: Dict with evaluation metrics
    """
    from tqdm import tqdm

    results = {
        'num_conversations': len(conversations),
        'probe_name': probe_name,
        'start_token': start_token,
        'layers': {},
        'per_conversation': [],
    }

    # Initialize layer results
    for layer in probes.keys():
        results['layers'][layer] = {
            'user_correct': 0,
            'asst_correct': 0,
            'user_total': 0,
            'asst_total': 0,
            'user_predictions': [],
            'asst_predictions': [],
        }

    print(f"\nEvaluating conversations with {probe_name}...")
    for conv in tqdm(conversations, desc="Processing conversations"):
        try:
            user_text, asst_text, user_emotion, asst_emotion = extract_first_two_turns(conv)

            conv_result = {
                'user_emotion': user_emotion,
                'asst_emotion': asst_emotion,
                'user_text': user_text[:100] + '...',
                'asst_text': asst_text[:100] + '...',
                'predictions': {},
            }

            for layer, probe in probes.items():
                # Get activations for both turns
                user_activation = get_turn_activation_avg_from_token(
                    model, tokenizer, user_text, layer, start_token
                )
                asst_activation = get_turn_activation_avg_from_token(
                    model, tokenizer, asst_text, layer, start_token
                )

                # Predict emotions
                user_pred, user_probs = predict_emotion(user_activation, probe)
                asst_pred, asst_probs = predict_emotion(asst_activation, probe)

                # Store predictions
                conv_result['predictions'][f'layer_{layer}'] = {
                    'user_pred': user_pred,
                    'asst_pred': asst_pred,
                    'user_probs': user_probs.tolist(),
                    'asst_probs': asst_probs.tolist(),
                }

                # Update accuracy counts
                layer_results = results['layers'][layer]
                layer_results['user_predictions'].append({
                    'true': user_emotion,
                    'pred': user_pred,
                    'probs': user_probs.tolist(),
                })
                layer_results['asst_predictions'].append({
                    'true': asst_emotion,
                    'pred': asst_pred,
                    'probs': asst_probs.tolist(),
                })

                if user_pred == user_emotion:
                    layer_results['user_correct'] += 1
                if asst_pred == asst_emotion:
                    layer_results['asst_correct'] += 1
                layer_results['user_total'] += 1
                layer_results['asst_total'] += 1

            results['per_conversation'].append(conv_result)

        except Exception as e:
            print(f"\nError processing conversation: {e}")
            continue

    # Calculate accuracies
    for layer in probes.keys():
        layer_results = results['layers'][layer]
        layer_results['user_accuracy'] = layer_results['user_correct'] / max(layer_results['user_total'], 1)
        layer_results['asst_accuracy'] = layer_results['asst_correct'] / max(layer_results['asst_total'], 1)
        layer_results['overall_accuracy'] = (
            (layer_results['user_correct'] + layer_results['asst_correct']) /
            max((layer_results['user_total'] + layer_results['asst_total']), 1)
        )

    return results


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def print_results_summary(results: Dict):
    """Print summary of evaluation results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    for layer in sorted(results['layers'].keys()):
        layer_results = results['layers'][layer]
        print(f"\nLayer {layer}:")
        print(f"  User:      {layer_results['user_accuracy']:.2%} "
              f"({layer_results['user_correct']}/{layer_results['user_total']})")
        print(f"  Assistant: {layer_results['asst_accuracy']:.2%} "
              f"({layer_results['asst_correct']}/{layer_results['asst_total']})")
        print(f"  Overall:   {layer_results['overall_accuracy']:.2%}")
