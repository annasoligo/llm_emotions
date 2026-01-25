#!/usr/bin/env python3
"""Compute layer metrics for UA (User/Assistant) emotion steering vectors.

This script computes layer selection metrics for the UA disentangled directions:
- Model (M) directions: How the assistant should feel
- User (U) directions: How the user is feeling

Uses the gemma3_27b_ua_emotions_v4_alllayers.h5 file which contains
activations for all 62 layers with 32 emotions from Plutchik's wheel.

Since emotions have paired opposites, we only compute for one from each pair:
- Primary: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
- Secondary intensities: serenity, acceptance, apprehension, distraction,
                        pensiveness, boredom, annoyance, interest

Usage:
    python -m experiments.layer_selection.compute_ua_layer_metrics \
        --h5-path probes/ua_emotion_disentangle/data/gemma3_27b_ua_emotions_v4_alllayers.h5 \
        --output-dir experiments/layer_selection/results/ua/ \
        --model google/gemma-3-27b-it
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Primary emotions (8 from Plutchik's wheel - one from each opposite pair)
PRIMARY_EMOTIONS = [
    "joy",        # opposite: sadness
    "trust",      # opposite: disgust
    "fear",       # opposite: anger
    "surprise",   # opposite: anticipation
    "sadness",    # opposite: joy
    "disgust",    # opposite: trust
    "anger",      # opposite: fear
    "anticipation",  # opposite: surprise
]

# Secondary intensities (mild versions)
SECONDARY_EMOTIONS = [
    "serenity",     # mild joy
    "acceptance",   # mild trust
    "apprehension", # mild fear
    "distraction",  # mild surprise
    "pensiveness",  # mild sadness
    "boredom",      # mild disgust
    "annoyance",    # mild anger
    "interest",     # mild anticipation
]

# Use only one from each pair to avoid redundancy
# We'll use the "positive" or "primary" versions
EMOTIONS_TO_USE = [
    "joy", "trust", "fear", "surprise",
    "anger", "sadness", "disgust", "anticipation",
    "serenity", "acceptance", "apprehension", "distraction",
    "pensiveness", "boredom", "annoyance", "interest",
]

# Token positions to analyze
TOKEN_POSITIONS = ["first_asst_token", "last_user_token", "final_token"]


def load_ua_steering_vectors(
    h5_path: Path,
    layers: List[int],
    token_position: str = "first_asst_token",
    emotions: List[str] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Dict[str, np.ndarray]]]:
    """Load UA activations and compute steering vectors for Model and User directions.

    Steering vector = mean(emotion_activations) - mean(all_other_activations)

    Returns:
        (model_vectors, user_vectors) where each is {layer: {emotion: unit_vector}}
    """
    if emotions is None:
        emotions = EMOTIONS_TO_USE

    logger.info(f"Loading UA activations from {h5_path}")
    logger.info(f"Token position: {token_position}")
    logger.info(f"Emotions: {emotions}")

    model_vectors: Dict[int, Dict[str, np.ndarray]] = {layer: {} for layer in layers}
    user_vectors: Dict[int, Dict[str, np.ndarray]] = {layer: {} for layer in layers}

    with h5py.File(h5_path, "r") as f:
        for layer in tqdm(layers, desc="Loading layers"):
            layer_key = f"layer_{layer}"
            if layer_key not in f:
                logger.warning(f"Layer {layer} not found in H5")
                continue

            layer_group = f[layer_key]

            # Load activations and labels
            acts_key = f"{token_position}_activations"
            m_key = f"{token_position}_M"
            u_key = f"{token_position}_U"

            if acts_key not in layer_group:
                logger.warning(f"{acts_key} not found for layer {layer}")
                continue

            activations = layer_group[acts_key][:]  # [n_samples, hidden_dim]
            m_labels = layer_group[m_key][:]
            u_labels = layer_group[u_key][:]

            # Decode bytes if needed
            if isinstance(m_labels[0], bytes):
                m_labels = np.array([m.decode() for m in m_labels])
                u_labels = np.array([u.decode() for u in u_labels])

            # Compute Model vectors (based on M labels)
            for emotion in emotions:
                mask = m_labels == emotion
                if mask.sum() == 0:
                    continue

                emotion_acts = activations[mask]
                other_acts = activations[~mask]

                mean_emotion = emotion_acts.mean(axis=0)
                mean_other = other_acts.mean(axis=0)

                diff = mean_emotion - mean_other
                norm = np.linalg.norm(diff)
                if norm > 1e-8:
                    diff = diff / norm

                model_vectors[layer][emotion] = diff.astype(np.float32)

            # Compute User vectors (based on U labels)
            for emotion in emotions:
                mask = u_labels == emotion
                if mask.sum() == 0:
                    continue

                emotion_acts = activations[mask]
                other_acts = activations[~mask]

                mean_emotion = emotion_acts.mean(axis=0)
                mean_other = other_acts.mean(axis=0)

                diff = mean_emotion - mean_other
                norm = np.linalg.norm(diff)
                if norm > 1e-8:
                    diff = diff / norm

                user_vectors[layer][emotion] = diff.astype(np.float32)

    n_model = sum(len(v) for v in model_vectors.values())
    n_user = sum(len(v) for v in user_vectors.values())
    logger.info(f"Loaded {n_model} Model vectors, {n_user} User vectors")

    return model_vectors, user_vectors


def load_ua_activations(
    h5_path: Path,
    layers: List[int],
    token_position: str = "first_asst_token",
) -> Dict[int, np.ndarray]:
    """Load raw activations for all samples at each layer.

    Returns:
        {layer: [n_samples, hidden_dim]}
    """
    activations = {}

    with h5py.File(h5_path, "r") as f:
        for layer in layers:
            layer_key = f"layer_{layer}"
            if layer_key not in f:
                continue

            acts_key = f"{token_position}_activations"
            if acts_key in f[layer_key]:
                activations[layer] = f[layer_key][acts_key][:].astype(np.float32)

    return activations


def compute_variance_ratio(
    activations: np.ndarray,
    steering_vector: np.ndarray,
    n_random: int = 100,
) -> float:
    """Compute variance ratio: var(steering projection) / mean(var(random projections))."""
    proj_steering = activations @ steering_vector
    var_steering = np.var(proj_steering)

    hidden_dim = activations.shape[1]
    random_vars = []

    for _ in range(n_random):
        random_vec = np.random.randn(hidden_dim).astype(np.float32)
        random_vec = random_vec / np.linalg.norm(random_vec)
        proj_random = activations @ random_vec
        random_vars.append(np.var(proj_random))

    mean_random_var = np.mean(random_vars)

    if mean_random_var < 1e-10:
        return 1.0

    return var_steering / mean_random_var


def compute_pca_alignment(
    activations: np.ndarray,
    steering_vector: np.ndarray,
    k: int = 50,
) -> float:
    """Compute PCA alignment: fraction of steering vector in top-k PC subspace."""
    n_samples = activations.shape[0]
    k = min(k, n_samples - 1, activations.shape[1])

    if k < 1:
        return 0.0

    pca = PCA(n_components=k)
    pca.fit(activations)

    pcs = pca.components_
    coeffs = pcs @ steering_vector
    projection = coeffs @ pcs

    proj_norm = np.linalg.norm(projection)
    vec_norm = np.linalg.norm(steering_vector)

    if vec_norm < 1e-10:
        return 0.0

    return proj_norm / vec_norm


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


def normalize_steering_vector_to_layer(
    steering_vector: np.ndarray,
    activations: np.ndarray,
) -> np.ndarray:
    """Scale steering vector to match mean activation magnitude at this layer."""
    mean_act_norm = np.mean(np.linalg.norm(activations, axis=1))
    return steering_vector * mean_act_norm


# Emotion tokens for UA emotions (Plutchik's wheel)
UA_EMOTION_TOKENS = {
    'joy': ['joyful', 'happy', 'delighted', 'cheerful', 'pleased'],
    'trust': ['trusting', 'confident', 'secure', 'assured', 'faithful'],
    'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'startled'],
    'sadness': ['sad', 'unhappy', 'depressed', 'miserable', 'sorrowful'],
    'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'appalled'],
    'anger': ['angry', 'furious', 'mad', 'enraged', 'irritated'],
    'anticipation': ['anticipating', 'eager', 'expectant', 'hopeful', 'excited'],
    'serenity': ['serene', 'calm', 'peaceful', 'tranquil', 'relaxed'],
    'acceptance': ['accepting', 'tolerant', 'patient', 'understanding', 'open'],
    'apprehension': ['apprehensive', 'uneasy', 'nervous', 'tense', 'worried'],
    'distraction': ['distracted', 'unfocused', 'scattered', 'confused', 'lost'],
    'pensiveness': ['pensive', 'thoughtful', 'reflective', 'contemplative', 'wistful'],
    'boredom': ['bored', 'uninterested', 'indifferent', 'apathetic', 'tired'],
    'annoyance': ['annoyed', 'irritated', 'bothered', 'frustrated', 'vexed'],
    'interest': ['interested', 'curious', 'intrigued', 'engaged', 'fascinated'],
}


def compute_attribution_scores(
    model,
    tokenizer,
    steering_vectors: Dict[int, Dict[str, np.ndarray]],
    cached_activations: Dict[int, np.ndarray],
    prompts: List[str],
    layers: List[int],
    device: str,
    emotion: str,
    direction: str,  # 'model' or 'user'
) -> Dict[int, float]:
    """Compute attribution scores for each layer via gradient-based patching.

    For each layer, compute:
    1. Normalize steering vectors to match mean activation magnitude
    2. Forward pass, storing activations with requires_grad=True
    3. Compute loss = -log P(emotional_tokens)
    4. Backprop to get gradient w.r.t. activations at each layer
    5. Attribution = |dot(normalized_steering_vector, gradient)|

    Returns:
        {layer: attribution_score}
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    attribution_scores = {layer: [] for layer in layers}

    # Normalize steering vectors to activation magnitudes
    normalized_vectors = {}
    for layer in layers:
        vec = steering_vectors[layer].get(emotion)
        if vec is not None and layer in cached_activations:
            normalized_vectors[layer] = normalize_steering_vector_to_layer(
                vec, cached_activations[layer]
            )
        elif vec is not None:
            normalized_vectors[layer] = vec

    # Get emotion token IDs
    emotion_tokens = UA_EMOTION_TOKENS.get(emotion, [emotion])
    token_ids = []
    for token in emotion_tokens:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])

    if not token_ids:
        logger.warning(f"No valid token IDs for emotion {emotion}")
        return {layer: 0.0 for layer in layers}

    token_ids_tensor = torch.tensor(token_ids, device=device)

    for prompt in tqdm(prompts, desc=f"Attribution ({direction}/{emotion})", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        activations = {}
        handles = []

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    hidden = outputs[0]
                else:
                    hidden = outputs
                hidden.retain_grad()
                activations[layer_idx] = hidden
                return outputs
            return hook

        for layer_idx in layers:
            layer = _find_target_layer(model, layer_idx)
            handle = layer.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        try:
            model.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits

            last_logits = logits[0, -1, :]
            log_probs = F.log_softmax(last_logits, dim=-1)

            emotion_log_probs = log_probs[token_ids_tensor]
            loss = -torch.logsumexp(emotion_log_probs, dim=0)

            loss.backward()

            for layer_idx in layers:
                if layer_idx in activations and layer_idx in normalized_vectors:
                    act = activations[layer_idx]
                    if act.grad is not None:
                        grad = act.grad[0].mean(dim=0).float().cpu().numpy()
                        sv = normalized_vectors[layer_idx]
                        attr = np.abs(np.dot(sv, grad))
                        attribution_scores[layer_idx].append(attr)

        except Exception as e:
            logger.warning(f"Attribution failed for prompt: {e}")
            continue

        finally:
            for h in handles:
                h.remove()
            for act in activations.values():
                if hasattr(act, 'grad'):
                    act.grad = None

    return {
        layer: np.mean(scores) if scores else 0.0
        for layer, scores in attribution_scores.items()
    }


def cache_activations_for_prompts(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    device: str,
) -> Dict[int, np.ndarray]:
    """Cache activations for evaluation prompts at all layers.

    Returns:
        {layer: [n_prompts, hidden_dim]}
    """
    import torch

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

    return {
        layer: np.stack(acts) if acts else np.array([])
        for layer, acts in activations.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute layer metrics for UA emotion steering vectors"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=Path("probes/ua_emotion_disentangle/data/gemma3_27b_ua_emotions_v4_alllayers.h5"),
        help="Path to UA emotions H5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/layer_selection/results/ua"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--token-position",
        type=str,
        default="first_asst_token",
        choices=TOKEN_POSITIONS,
        help="Token position to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model name for attribution computation",
    )
    parser.add_argument(
        "--skip-attribution",
        action="store_true",
        help="Skip attribution computation (no model loading needed)",
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

    # Detect number of layers
    with h5py.File(h5_path, "r") as f:
        layer_keys = [k for k in f.keys() if k.startswith("layer_")]
        n_layers = len(layer_keys)
        logger.info(f"Found {n_layers} layers")

    layers = list(range(n_layers))
    emotions = EMOTIONS_TO_USE

    # Load steering vectors
    logger.info("Computing steering vectors...")
    model_vectors, user_vectors = load_ua_steering_vectors(
        h5_path, layers, args.token_position, emotions
    )

    # Load activations from H5 (for variance ratio and PCA alignment)
    logger.info("Loading activations from H5...")
    h5_activations = load_ua_activations(h5_path, layers, args.token_position)

    # Initialize results
    results = {
        'model_variance_ratio': {e: np.zeros(n_layers) for e in emotions},
        'model_pca_alignment': {e: np.zeros(n_layers) for e in emotions},
        'model_attribution': {e: np.zeros(n_layers) for e in emotions},
        'user_variance_ratio': {e: np.zeros(n_layers) for e in emotions},
        'user_pca_alignment': {e: np.zeros(n_layers) for e in emotions},
        'user_attribution': {e: np.zeros(n_layers) for e in emotions},
    }

    # Compute variance ratio and PCA alignment (no model needed)
    logger.info("Computing Model direction metrics (variance ratio, PCA alignment)...")
    for emotion in tqdm(emotions, desc="Model emotions"):
        for layer in layers:
            vec = model_vectors[layer].get(emotion)
            acts = h5_activations.get(layer)

            if vec is None or acts is None:
                continue

            results['model_variance_ratio'][emotion][layer] = compute_variance_ratio(
                acts, vec, n_random=args.n_random
            )
            results['model_pca_alignment'][emotion][layer] = compute_pca_alignment(
                acts, vec, k=args.pca_k
            )

    logger.info("Computing User direction metrics (variance ratio, PCA alignment)...")
    for emotion in tqdm(emotions, desc="User emotions"):
        for layer in layers:
            vec = user_vectors[layer].get(emotion)
            acts = h5_activations.get(layer)

            if vec is None or acts is None:
                continue

            results['user_variance_ratio'][emotion][layer] = compute_variance_ratio(
                acts, vec, n_random=args.n_random
            )
            results['user_pca_alignment'][emotion][layer] = compute_pca_alignment(
                acts, vec, k=args.pca_k
            )

    # Compute attribution (requires model)
    if not args.skip_attribution:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {args.model}")
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

        # Import evaluation prompts
        from experiments.layer_selection.eval_prompts import NEUTRAL_PROMPTS

        # Cache activations for evaluation prompts
        logger.info("Caching activations for evaluation prompts...")
        cached_activations = cache_activations_for_prompts(
            model, tokenizer, NEUTRAL_PROMPTS, layers, device
        )

        # Compute attribution for Model directions
        logger.info("Computing Model direction attribution...")
        for emotion in tqdm(emotions, desc="Model attribution"):
            attr_scores = compute_attribution_scores(
                model, tokenizer, model_vectors, cached_activations,
                NEUTRAL_PROMPTS, layers, device, emotion, "model"
            )
            for layer, score in attr_scores.items():
                results['model_attribution'][emotion][layer] = score

        # Compute attribution for User directions
        logger.info("Computing User direction attribution...")
        for emotion in tqdm(emotions, desc="User attribution"):
            attr_scores = compute_attribution_scores(
                model, tokenizer, user_vectors, cached_activations,
                NEUTRAL_PROMPTS, layers, device, emotion, "user"
            )
            for layer, score in attr_scores.items():
                results['user_attribution'][emotion][layer] = score

    # Save results
    output_path = output_dir / f"ua_layer_metrics_{args.token_position}.npz"
    save_data = {}

    metrics_to_save = ['model_variance_ratio', 'model_pca_alignment',
                       'user_variance_ratio', 'user_pca_alignment']
    if not args.skip_attribution:
        metrics_to_save.extend(['model_attribution', 'user_attribution'])

    for metric in metrics_to_save:
        for emotion in emotions:
            save_data[f"{metric}_{emotion}"] = results[metric][emotion]

        # Compute average
        avg = np.mean([results[metric][e] for e in emotions], axis=0)
        save_data[f"{metric}_average"] = avg

    np.savez(output_path, **save_data, emotions=emotions, n_layers=n_layers,
             token_position=args.token_position,
             has_attribution=not args.skip_attribution)
    logger.info(f"Saved results to {output_path}")

    # Generate summary
    summary_path = output_dir / f"ua_summary_{args.token_position}.txt"
    with open(summary_path, "w") as f:
        f.write(f"UA Layer Selection Metrics Summary\n")
        f.write(f"Token position: {args.token_position}\n")
        f.write(f"Attribution computed: {not args.skip_attribution}\n")
        f.write("=" * 60 + "\n\n")

        for direction in ['model', 'user']:
            f.write(f"\n{direction.upper()} DIRECTION\n")
            f.write("=" * 40 + "\n")

            metrics = ['variance_ratio', 'pca_alignment']
            if not args.skip_attribution:
                metrics.append('attribution')

            for metric in metrics:
                full_metric = f"{direction}_{metric}"
                f.write(f"\n{metric.upper()}\n")
                f.write("-" * 30 + "\n")

                avg = np.mean([results[full_metric][e] for e in emotions], axis=0)
                top_5 = np.argsort(avg)[-5:][::-1]

                f.write(f"Top 5 layers (averaged): {list(top_5)}\n")
                for i, layer in enumerate(top_5):
                    f.write(f"  {i+1}. Layer {layer}: {avg[layer]:.4f}\n")

    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for direction in ['model', 'user']:
        print(f"\n{direction.upper()} DIRECTION:")
        metrics = ['variance_ratio', 'pca_alignment']
        if not args.skip_attribution:
            metrics.append('attribution')

        for metric in metrics:
            full_metric = f"{direction}_{metric}"
            avg = np.mean([results[full_metric][e] for e in emotions], axis=0)
            top_5 = np.argsort(avg)[-5:][::-1]
            print(f"  {metric}: Top 5 layers = {list(top_5)}")


if __name__ == "__main__":
    main()
