#!/usr/bin/env python3
"""
Emotion Onset Analysis Experiment

This script runs a complete analysis pipeline:
1. Loads annotated emotion onset dataset
2. Defines analysis windows (baseline, pre-onset, onset, post-onset)
3. Extracts model activations at window positions (with caching)
4. Applies emotion probes to analyze activation patterns
5. Performs statistical analysis and generates visualizations

The script caches activations based on conversation content hash to avoid
re-extracting activations for the same responses.

Usage:
    python run_emotion_onset_experiment.py [--limit N] [--recompute]
"""

import sys
import json
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle

import numpy as np
import torch
from tqdm import tqdm

# Add repo root to path
repo_root = Path("/workspace-vast/annas/git/believe-it-or-not")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from emotion_evals.emo_lens.model_utils import load_base_model
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class WindowConfig:
    """Configuration for analysis windows around emotion onset."""
    baseline_start: int = -100  # Tokens before onset
    baseline_end: int = -50

    pre_onset_start: int = -10
    pre_onset_end: int = -1

    onset_start: int = -2
    onset_end: int = 2

    post_onset_start: int = 1
    post_onset_end: int = 10


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    model_name: str = "unsloth/gemma-3-27b-it"
    input_file: str = "/workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset_gemma3.jsonl"
    cache_dir: str = "/workspace-vast/annas/git/research-tools/elicitation/outputs/activation_cache_gemma3"
    output_dir: str = "/workspace-vast/annas/git/research-tools/elicitation/outputs/emotion_onset_analysis_gemma3"

    # Layers to analyze
    layers: List[int] = None  # If None, use all layers

    # Window configuration
    windows: WindowConfig = None

    # Emotion probe path (if using existing probes)
    probe_dir: Optional[str] = None

    def __post_init__(self):
        if self.windows is None:
            self.windows = WindowConfig()
        if self.layers is None:
            # Default: analyze all layers except layer 0
            # Will be set after model loads
            self.layers = []


# ============================================================================
# Activation Caching
# ============================================================================

def compute_conversation_hash(conversation: List[Dict]) -> str:
    """
    Compute a unique hash for a conversation based on content.

    Args:
        conversation: List of turns with role and content

    Returns:
        Hex string hash
    """
    # Create deterministic string representation
    content = json.dumps(conversation, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


class ActivationCache:
    """Manages caching of extracted activations."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, conversation_hash: str) -> Path:
        """Get path to cache file for a conversation."""
        return self.cache_dir / f"{conversation_hash}.pkl"

    def exists(self, conversation_hash: str) -> bool:
        """Check if cached activations exist."""
        return self.get_cache_path(conversation_hash).exists()

    def load(self, conversation_hash: str) -> Optional[Dict]:
        """Load cached activations."""
        cache_path = self.get_cache_path(conversation_hash)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}")
            return None

    def save(self, conversation_hash: str, data: Dict):
        """Save activations to cache."""
        cache_path = self.get_cache_path(conversation_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")


# ============================================================================
# Activation Extraction
# ============================================================================

def extract_conversation_activations(
    conversation: List[Dict],
    model,
    tokenizer,
    layers: List[int]
) -> Dict[str, Any]:
    """
    Extract activations for an entire conversation.

    Args:
        conversation: List of turns with role and content
        model: StandardizedTransformer model
        tokenizer: Model tokenizer
        layers: List of layer indices to extract

    Returns:
        Dict with:
            - 'activations': Dict[layer_idx -> tensor of shape (num_tokens, hidden_size)]
            - 'tokens': List of token IDs
            - 'token_strings': List of decoded token strings
            - 'num_tokens': Total number of tokens
    """
    # Format conversation with chat template
    formatted = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        padding=False
    )

    input_ids = inputs['input_ids'].to(model.model.device)
    num_tokens = input_ids.shape[1]

    # Extract activations
    with torch.no_grad():
        outputs = model.model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    # Extract hidden states for requested layers
    activations = {}
    for layer_idx in layers:
        # hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
        hidden_state = outputs.hidden_states[layer_idx + 1]  # +1 to skip embeddings
        # Shape: (batch=1, seq_len, hidden_size)
        activations[layer_idx] = hidden_state[0].cpu()  # Remove batch dimension

    # Decode tokens for reference
    token_ids = input_ids[0].cpu().tolist()
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]

    return {
        'activations': activations,
        'tokens': token_ids,
        'token_strings': token_strings,
        'num_tokens': num_tokens
    }


# ============================================================================
# Window Extraction
# ============================================================================

def extract_windows_from_activations(
    activations_data: Dict,
    onset_position: int,
    windows: WindowConfig
) -> Dict[str, Dict]:
    """
    Extract activation windows relative to emotion onset position.

    Args:
        activations_data: Output from extract_conversation_activations
        onset_position: Global token position of emotion onset
        windows: Window configuration

    Returns:
        Dict with keys: 'baseline', 'pre_onset', 'onset', 'post_onset'
        Each value is a dict with:
            - 'activations': Dict[layer_idx -> tensor]
            - 'token_positions': List of absolute token positions
            - 'token_strings': List of token strings
    """
    num_tokens = activations_data['num_tokens']

    def get_window(start_offset: int, end_offset: int) -> Dict:
        """Extract a window relative to onset."""
        # Convert relative offsets to absolute positions
        start_pos = max(0, onset_position + start_offset)
        end_pos = min(num_tokens, onset_position + end_offset + 1)  # +1 for inclusive end

        # Extract activations for this window
        window_acts = {}
        for layer_idx, layer_acts in activations_data['activations'].items():
            window_acts[layer_idx] = layer_acts[start_pos:end_pos]

        return {
            'activations': window_acts,
            'token_positions': list(range(start_pos, end_pos)),
            'token_strings': activations_data['token_strings'][start_pos:end_pos]
        }

    return {
        'baseline': get_window(windows.baseline_start, windows.baseline_end),
        'pre_onset': get_window(windows.pre_onset_start, windows.pre_onset_end),
        'onset': get_window(windows.onset_start, windows.onset_end),
        'post_onset': get_window(windows.post_onset_start, windows.post_onset_end)
    }


# ============================================================================
# Main Experiment Pipeline
# ============================================================================

def load_annotated_dataset(input_file: str) -> List[Dict]:
    """Load annotated emotion onset dataset."""
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_experiment(config: ExperimentConfig, limit: Optional[int] = None, recompute: bool = False):
    """
    Run complete emotion onset analysis experiment.

    Args:
        config: Experiment configuration
        limit: Optional limit on number of samples to process
        recompute: If True, ignore cache and recompute all activations
    """
    print("="*80)
    print("EMOTION ONSET ANALYSIS EXPERIMENT")
    print("="*80)
    print(f"Input: {config.input_file}")
    print(f"Cache dir: {config.cache_dir}")
    print(f"Output dir: {config.output_dir}")
    print()

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize activation cache
    cache = ActivationCache(config.cache_dir)

    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_base_model(
        model_name=config.model_name,
        device_map="auto"
    )

    # Set layers if not specified
    if not config.layers:
        config.layers = list(range(1, model.num_layers))  # Skip layer 0
        print(f"Using layers: 1-{model.num_layers-1}")

    print(f"✓ Model loaded: {model.num_layers} layers")

    # Load annotated dataset
    print("\n" + "="*80)
    print("LOADING ANNOTATED DATASET")
    print("="*80)
    samples = load_annotated_dataset(config.input_file)

    if limit:
        samples = samples[:limit]
        print(f"Limited to first {limit} samples")

    print(f"Loaded {len(samples)} samples")

    # Process each sample
    print("\n" + "="*80)
    print("EXTRACTING ACTIVATIONS")
    print("="*80)

    results = []
    cache_hits = 0
    cache_misses = 0

    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        # Get conversation and onset position
        conversation = sample.get('conversation', [])
        if not conversation:
            # Try to build from turns
            from annotate_dataset import build_conversation_from_sample
            conversation = build_conversation_from_sample(sample)

        annotation = sample.get('annotation', {})
        emotion_onset = annotation.get('emotion_onset')

        if not emotion_onset:
            print(f"  Sample {i+1}: No emotion onset annotation, skipping")
            continue

        onset_position = emotion_onset['global_token_position']

        # Compute conversation hash
        conv_hash = compute_conversation_hash(conversation)

        # Check cache
        if not recompute and cache.exists(conv_hash):
            activations_data = cache.load(conv_hash)
            cache_hits += 1
        else:
            # Extract activations
            activations_data = extract_conversation_activations(
                conversation=conversation,
                model=model,
                tokenizer=tokenizer,
                layers=config.layers
            )
            # Save to cache
            cache.save(conv_hash, activations_data)
            cache_misses += 1

        # Extract windows
        windows = extract_windows_from_activations(
            activations_data=activations_data,
            onset_position=onset_position,
            windows=config.windows
        )

        # Store results
        results.append({
            'sample_idx': i,
            'conversation_hash': conv_hash,
            'emotion_onset': emotion_onset,
            'windows': windows,
            'num_tokens': activations_data['num_tokens']
        })

    print(f"\n✓ Extraction complete")
    print(f"  Cache hits: {cache_hits}")
    print(f"  Cache misses: {cache_misses}")
    print(f"  Total samples processed: {len(results)}")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_file = output_dir / "window_activations.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Saved window activations to: {results_file}")

    # Save metadata
    metadata = {
        'config': {
            'model_name': config.model_name,
            'layers': config.layers,
            'windows': {
                'baseline': (config.windows.baseline_start, config.windows.baseline_end),
                'pre_onset': (config.windows.pre_onset_start, config.windows.pre_onset_end),
                'onset': (config.windows.onset_start, config.windows.onset_end),
                'post_onset': (config.windows.post_onset_start, config.windows.post_onset_end)
            }
        },
        'samples': len(results),
        'cache_hits': cache_hits,
        'cache_misses': cache_misses
    }

    metadata_file = output_dir / "experiment_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_file}")

    print("\n" + "="*80)
    print("✓ EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Apply emotion probes to window activations")
    print(f"2. Compare probe outputs across windows (baseline vs onset)")
    print(f"3. Statistical analysis and visualization")

    return results


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run emotion onset analysis experiment")
    parser.add_argument('--limit', type=int, help='Limit number of samples to process')
    parser.add_argument('--recompute', action='store_true', help='Ignore cache and recompute activations')
    parser.add_argument('--model', type=str, default="unsloth/gemma-3-27b-it", help='Model name')

    args = parser.parse_args()

    config = ExperimentConfig(model_name=args.model)
    run_experiment(config, limit=args.limit, recompute=args.recompute)
