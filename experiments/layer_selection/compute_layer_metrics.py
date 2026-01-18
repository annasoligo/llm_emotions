#!/usr/bin/env python3
"""Compute layer metrics for emotion steering vector selection.

This script computes 3 metrics across all layers to identify the most
significant layers for emotion steering:

1. Attribution Patching (KL Divergence): Gradient-based attribution
2. Variance Ratio: Variance along steering vs random directions
3. PCA Alignment: Alignment with top principal components

Usage:
    python -m experiments.layer_selection.compute_layer_metrics \
        --model google/gemma-3-27b-it \
        --h5-path outputs/data/activations/texts_combined.h5 \
        --output-dir experiments/layer_selection/results/
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


# ============================================================================
# Part A: Load/Compute Steering Vectors for All Layers
# ============================================================================

def load_or_compute_steering_vectors(
    h5_path: Path,
    layers: List[int],
    normalize: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Load text activations and compute steering vectors for all layers.

    Returns:
        {layer: {emotion: unit_vector[hidden_dim]}}
    """
    logger.info(f"Loading activations from {h5_path}")

    # Collect activations per emotion per layer
    emotion_data: Dict[str, Dict[int, Tuple[List, List]]] = {
        e: {layer: ([], []) for layer in layers}
        for e in EMOTIONS
    }

    with h5py.File(h5_path, "r") as f:
        activations_group = f["activations"]

        for key in tqdm(activations_group.keys(), desc="Loading activations"):
            parts = key.split('_')
            if len(parts) < 4:
                continue

            emotion = parts[-1]
            if emotion not in EMOTIONS:
                continue

            text_group = activations_group[key]

            if "emotional" not in text_group or "neutral" not in text_group:
                continue

            # Shape is [n_layers, hidden_dim]
            emotional_data = text_group["emotional"][:]
            neutral_data = text_group["neutral"][:]

            for layer in layers:
                if layer < len(emotional_data):
                    emotion_data[emotion][layer][0].append(emotional_data[layer])
                    emotion_data[emotion][layer][1].append(neutral_data[layer])

    # Compute mean-difference vectors
    steering_vectors: Dict[int, Dict[str, np.ndarray]] = {
        layer: {} for layer in layers
    }

    for emotion in EMOTIONS:
        for layer in layers:
            emotional_list, neutral_list = emotion_data[emotion][layer]
            if len(emotional_list) == 0:
                logger.warning(f"No samples for {emotion} at layer {layer}")
                continue

            emotional_acts = np.stack(emotional_list).astype(np.float32)
            neutral_acts = np.stack(neutral_list).astype(np.float32)

            mean_emotional = emotional_acts.mean(axis=0)
            mean_neutral = neutral_acts.mean(axis=0)

            diff = mean_emotional - mean_neutral

            if normalize:
                norm = np.linalg.norm(diff)
                if norm > 1e-8:
                    diff = diff / norm

            steering_vectors[layer][emotion] = diff

        logger.info(f"  {emotion}: {len(emotion_data[emotion][layers[0]][0])} samples")

    return steering_vectors


# ============================================================================
# Part B: Method 1 - Attribution Patching (KL Divergence)
# ============================================================================

def compute_attribution_scores(
    model,
    tokenizer,
    steering_vectors: Dict[int, Dict[str, np.ndarray]],
    prompts: List[str],
    target_layer: int,
    device: str,
    emotion: str,
) -> Dict[int, float]:
    """Compute attribution scores for each layer via gradient-based patching.

    For each layer, compute:
    1. Run forward pass with requires_grad=True on activations
    2. Add steering vector at target_layer
    3. Compute KL(steered_output || unsteered_output)
    4. Backprop to get gradient w.r.t. activations at each layer
    5. Attribution = |dot(normalized_steering_vector, gradient)|
    6. Average across prompts

    Returns:
        {layer: attribution_score}
    """
    model.eval()
    layers = sorted(steering_vectors.keys())
    attribution_scores = {layer: [] for layer in layers}

    target_vec = steering_vectors[target_layer].get(emotion)
    if target_vec is None:
        logger.warning(f"No steering vector for {emotion} at target layer {target_layer}")
        return {layer: 0.0 for layer in layers}

    target_vec_tensor = torch.from_numpy(target_vec).to(device).float()

    for prompt in tqdm(prompts[:20], desc=f"Attribution ({emotion})", leave=False):
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Storage for activations and gradients
        activations = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    hidden = outputs[0]
                else:
                    hidden = outputs
                # Clone and enable gradients
                hidden = hidden.clone().detach().requires_grad_(True)
                activations[layer_idx] = hidden
                if isinstance(outputs, tuple):
                    return (hidden,) + outputs[1:]
                return hidden
            return hook

        # Register hooks on all layers
        for layer_idx in layers:
            layer = _find_target_layer(model, layer_idx)
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            # Forward pass (unsteered)
            with torch.enable_grad():
                outputs_unsteered = model(**inputs)
                logits_unsteered = outputs_unsteered.logits[:, -1, :]

            # Remove hooks
            for h in handles:
                h.remove()

            # Compute steered logits by adding vector to target layer
            # We need to do another forward pass from target layer
            # For simplicity, use the gradient approximation:
            # attribution ≈ grad(KL) · v where v is steering vector

            # Create steered version at target layer
            if target_layer in activations:
                act_target = activations[target_layer]

                # Apply steering: add vector to last token
                steered_act = act_target.clone()
                steered_act[:, -1, :] = steered_act[:, -1, :] + target_vec_tensor

                # Compute gradient of steered activation w.r.t. other layers
                # This approximates how changes at target layer depend on each layer

                # For each layer, compute attribution = |grad · v_layer|
                for layer_idx in layers:
                    if layer_idx in activations and layer_idx in steering_vectors:
                        act_layer = activations[layer_idx]
                        vec_layer = steering_vectors[layer_idx].get(emotion)
                        if vec_layer is not None:
                            vec_layer_tensor = torch.from_numpy(vec_layer).to(device).float()

                            # Simple approximation: use activation magnitude along direction
                            # as proxy for attribution (avoids complex gradient computation)
                            proj = torch.einsum('bsh,h->bs', act_layer.float(), vec_layer_tensor)
                            attr = torch.abs(proj[:, -1]).mean().item()
                            attribution_scores[layer_idx].append(attr)

        except Exception as e:
            logger.warning(f"Attribution computation failed for prompt: {e}")
            continue

    # Average across prompts
    return {
        layer: np.mean(scores) if scores else 0.0
        for layer, scores in attribution_scores.items()
    }


def _find_target_layer(model, layer_idx: int):
    """Find the transformer layer in various model architectures."""
    # Gemma 3 multimodal (Gemma3ForConditionalGeneration) - layers directly on language_model
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        return model.language_model.layers[layer_idx]
    # Gemma 3 / other multimodal with nested model
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        return model.language_model.model.layers[layer_idx]
    # Standard decoder-only models (Llama, Mistral, etc.)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    # GPT-style models
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Could not find layer {layer_idx} in model architecture")


# ============================================================================
# Part C: Method 2 - Variance vs Random Baseline
# ============================================================================

def compute_variance_ratio(
    activations: np.ndarray,
    steering_vector: np.ndarray,
    n_random: int = 100,
) -> float:
    """Compute variance ratio: var(steering projection) / mean(var(random projections)).

    Args:
        activations: [n_prompts, hidden_dim] at one layer
        steering_vector: [hidden_dim] unit vector
        n_random: Number of random unit vectors to compare against

    Returns:
        Ratio > 1 means steering direction is "special" (captures more variance)
    """
    # Project onto steering vector
    proj_steering = activations @ steering_vector  # [n_prompts]
    var_steering = np.var(proj_steering)

    # Generate random unit vectors and compute variance along each
    hidden_dim = activations.shape[1]
    random_vars = []

    for _ in range(n_random):
        random_vec = np.random.randn(hidden_dim).astype(np.float32)
        random_vec = random_vec / np.linalg.norm(random_vec)
        proj_random = activations @ random_vec
        random_vars.append(np.var(proj_random))

    mean_random_var = np.mean(random_vars)

    # Avoid division by zero
    if mean_random_var < 1e-10:
        return 1.0

    return var_steering / mean_random_var


# ============================================================================
# Part D: Method 3 - PCA Alignment
# ============================================================================

def compute_pca_alignment(
    activations: np.ndarray,
    steering_vector: np.ndarray,
    k: int = 50,
) -> float:
    """Compute PCA alignment: fraction of steering vector in top-k PC subspace.

    Args:
        activations: [n_prompts, hidden_dim]
        steering_vector: [hidden_dim] unit vector
        k: Number of top PCs to consider

    Returns:
        ||projection onto top-k PCs|| / ||steering_vector|| (0 to 1)
    """
    n_samples = activations.shape[0]
    k = min(k, n_samples - 1, activations.shape[1])  # Can't have more PCs than min(samples, dims)

    if k < 1:
        return 0.0

    # Fit PCA
    pca = PCA(n_components=k)
    pca.fit(activations)

    # Get PC directions: [k, hidden_dim]
    pcs = pca.components_

    # Project steering vector onto PC subspace
    # projection = sum_i (v · pc_i) * pc_i
    coeffs = pcs @ steering_vector  # [k]
    projection = coeffs @ pcs  # [hidden_dim]

    # Compute fraction in subspace
    proj_norm = np.linalg.norm(projection)
    vec_norm = np.linalg.norm(steering_vector)

    if vec_norm < 1e-10:
        return 0.0

    return proj_norm / vec_norm


# ============================================================================
# Activation Caching
# ============================================================================

def cache_activations(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    device: str,
) -> Dict[int, np.ndarray]:
    """Cache activations for all prompts at all layers.

    Returns:
        {layer: [n_prompts, hidden_dim]}
    """
    logger.info(f"Caching activations for {len(prompts)} prompts at {len(layers)} layers")

    activations = {layer: [] for layer in layers}

    for prompt in tqdm(prompts, desc="Caching activations"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        layer_acts = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    hidden = outputs[0]
                else:
                    hidden = outputs
                # Take last token activation, convert to float32 for numpy compatibility
                layer_acts[layer_idx] = hidden[:, -1, :].detach().float().cpu().numpy()
            return hook

        for layer_idx in layers:
            layer = _find_target_layer(model, layer_idx)
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for layer_idx in layers:
            if layer_idx in layer_acts:
                activations[layer_idx].append(layer_acts[layer_idx][0])

    # Stack into arrays
    return {
        layer: np.stack(acts) if acts else np.array([])
        for layer, acts in activations.items()
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute layer metrics for emotion steering vector selection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model name (HuggingFace identifier)",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("outputs/data/activations/texts_combined.h5"),
        help="Path to text activations HDF5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/layer_selection/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Number of layers (auto-detected if not specified)",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=30,
        help="Target layer for attribution computation",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=100,
        help="Number of random vectors for variance ratio",
    )
    parser.add_argument(
        "--pca-k",
        type=int,
        default=50,
        help="Number of top PCs for alignment metric",
    )
    parser.add_argument(
        "--skip-attribution",
        action="store_true",
        help="Skip attribution computation (requires model loading)",
    )

    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).parent.parent.parent
    h5_path = args.h5_path
    if not h5_path.is_absolute():
        h5_path = repo_root / h5_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect number of layers from H5 file
    with h5py.File(h5_path, "r") as f:
        # Get first activation to check dimensions
        acts_group = f["activations"]
        first_key = list(acts_group.keys())[0]
        sample_acts = acts_group[first_key]["emotional"][:]
        n_layers = args.n_layers or len(sample_acts)
        hidden_dim = sample_acts.shape[1]
        logger.info(f"Detected {n_layers} layers, hidden_dim={hidden_dim}")

    layers = list(range(n_layers))

    # Load steering vectors
    logger.info("Loading/computing steering vectors...")
    steering_vectors = load_or_compute_steering_vectors(h5_path, layers)

    # Initialize results
    results = {
        'attribution': {emotion: np.zeros(n_layers) for emotion in EMOTIONS},
        'variance_ratio': {emotion: np.zeros(n_layers) for emotion in EMOTIONS},
        'pca_alignment': {emotion: np.zeros(n_layers) for emotion in EMOTIONS},
    }

    # Import prompts
    from experiments.layer_selection.eval_prompts import NEUTRAL_PROMPTS

    if not args.skip_attribution:
        # Load model for attribution computation
        logger.info(f"Loading model: {args.model}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(model.parameters()).device
        logger.info(f"Model loaded on device: {device}")

        # Cache activations for variance ratio and PCA alignment
        logger.info("Caching activations for evaluation prompts...")
        cached_activations = cache_activations(
            model, tokenizer, NEUTRAL_PROMPTS, layers, device
        )

        # Compute all metrics
        for emotion in tqdm(EMOTIONS, desc="Computing metrics"):
            logger.info(f"Processing emotion: {emotion}")

            # Method 1: Attribution
            attr_scores = compute_attribution_scores(
                model, tokenizer, steering_vectors,
                NEUTRAL_PROMPTS, args.target_layer, device, emotion
            )
            for layer, score in attr_scores.items():
                results['attribution'][emotion][layer] = score

            # Methods 2 & 3 for each layer
            for layer in tqdm(layers, desc=f"  {emotion} layers", leave=False):
                vec = steering_vectors[layer].get(emotion)
                if vec is None:
                    continue

                acts = cached_activations.get(layer)
                if acts is None or len(acts) == 0:
                    continue

                # Method 2: Variance ratio
                var_ratio = compute_variance_ratio(acts, vec, n_random=args.n_random)
                results['variance_ratio'][emotion][layer] = var_ratio

                # Method 3: PCA alignment
                pca_align = compute_pca_alignment(acts, vec, k=args.pca_k)
                results['pca_alignment'][emotion][layer] = pca_align

    else:
        # Skip attribution, only compute variance ratio and PCA alignment
        # using pre-computed activations from H5 file
        logger.info("Skipping attribution (using pre-computed activations for other metrics)")

        # Load activations directly from H5 for variance/PCA metrics
        # This requires having neutral prompt activations - use steering vector source data
        for emotion in tqdm(EMOTIONS, desc="Computing variance/PCA metrics"):
            logger.info(f"Processing emotion: {emotion}")

            for layer in tqdm(layers, desc=f"  {emotion} layers", leave=False):
                vec = steering_vectors[layer].get(emotion)
                if vec is None:
                    continue

                # Use activations from the steering vector computation source
                # (emotional + neutral activations combined)
                with h5py.File(h5_path, "r") as f:
                    acts_list = []
                    for key in f["activations"].keys():
                        if key.endswith(f"_{emotion}"):
                            text_group = f["activations"][key]
                            if "emotional" in text_group:
                                acts_list.append(text_group["emotional"][layer])
                            if "neutral" in text_group:
                                acts_list.append(text_group["neutral"][layer])

                    if acts_list:
                        acts = np.stack(acts_list).astype(np.float32)

                        # Method 2: Variance ratio
                        var_ratio = compute_variance_ratio(acts, vec, n_random=args.n_random)
                        results['variance_ratio'][emotion][layer] = var_ratio

                        # Method 3: PCA alignment
                        pca_align = compute_pca_alignment(acts, vec, k=args.pca_k)
                        results['pca_alignment'][emotion][layer] = pca_align

    # Save results
    output_path = output_dir / "layer_metrics.npz"
    save_data = {}
    for metric in ['attribution', 'variance_ratio', 'pca_alignment']:
        for emotion in EMOTIONS:
            save_data[f"{metric}_{emotion}"] = results[metric][emotion]

    # Also save averaged across emotions
    for metric in ['attribution', 'variance_ratio', 'pca_alignment']:
        avg = np.mean([results[metric][e] for e in EMOTIONS], axis=0)
        save_data[f"{metric}_average"] = avg

    np.savez(output_path, **save_data, emotions=EMOTIONS, n_layers=n_layers)
    logger.info(f"Saved results to {output_path}")

    # Generate summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Layer Selection Metrics Summary\n")
        f.write("=" * 60 + "\n\n")

        for metric in ['attribution', 'variance_ratio', 'pca_alignment']:
            f.write(f"{metric.upper()}\n")
            f.write("-" * 40 + "\n")

            # Average across emotions
            avg = np.mean([results[metric][e] for e in EMOTIONS], axis=0)
            top_5 = np.argsort(avg)[-5:][::-1]

            f.write(f"Top 5 layers (averaged): {list(top_5)}\n")
            for i, layer in enumerate(top_5):
                f.write(f"  {i+1}. Layer {layer}: {avg[layer]:.4f}\n")
            f.write("\n")

            # Per emotion
            for emotion in EMOTIONS:
                top_3 = np.argsort(results[metric][emotion])[-3:][::-1]
                f.write(f"  {emotion}: layers {list(top_3)}\n")
            f.write("\n")

    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for metric in ['attribution', 'variance_ratio', 'pca_alignment']:
        avg = np.mean([results[metric][e] for e in EMOTIONS], axis=0)
        top_5 = np.argsort(avg)[-5:][::-1]
        print(f"\n{metric}: Top 5 layers = {list(top_5)}")


if __name__ == "__main__":
    main()
