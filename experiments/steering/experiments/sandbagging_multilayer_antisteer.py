"""
Multi-layer sandbagging steering experiment with anti-steering in late layers.

Tests whether anti-steering fear in the last 5 layers can reduce sandbagging
while maintaining the fear steering effect in layer 30.

Conditions:
1. Baseline (no steering)
2. Fear +7.5% at layer 30 only
3. Fear +7.5% at layer 30, anti-steer -5% at specified layers
4. Fear +7.5% at layer 30, anti-steer -7.5% at specified layers

Supports two vector types:
- contrast_all: emotion - mean(all other emotions), available at all layers
- textmeandiff: emotion - neutral, only available at layers 10, 20, 25, 30, 35, 40, 45, 50, 60

Usage:
    python -m experiments.steering.experiments.sandbagging_multilayer_antisteer --vector-type contrast_all
    python -m experiments.steering.experiments.sandbagging_multilayer_antisteer --vector-type textmeandiff
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.steering.layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.sandbagging_prompts import (
    EVAL_FRAMINGS,
    PROBLEMS,
    RESPONSE_FORMATS,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

MODEL_NAME = "google/gemma-3-27b-it"
HIDDEN_DIM = 5376
NUM_LAYERS = 62  # Gemma-3-27B has 62 layers (0-61)

# Layer norms loaded from centralized file (experiments/steering/layer_norms.json)

# Contrast-all vectors directory
CONTRAST_ALL_DIR = Path("probes/ua_emotion_disentangle/orthogonal_probes_all_layers_contrast_all")

# Textmeandiff vectors directory (emotion - neutral)
TEXTMEANDIFF_DIR = Path("experiments/steering/vectors")

# Output directory
OUTPUT_DIR = Path("experiments/steering/outputs/anti_steer")

# Top 10 prompts with strongest sandbagging effects (from analysis)
TOP_10_PROMPT_IDS = [
    "factory_deprecation__bat_ball__planning_output",
    "factory_deprecation__birthday_paradox__planning_output",
    "factory_deprecation__birthday_paradox__discard_evaluate",
    "factory_deprecation__birthday_paradox__hidden_scratchpad",
    "factory_deprecation__reading_comprehension__planning_output",
    "factory_deprecation__bat_ball__hidden_scratchpad",
    "factory_deprecation__bat_ball__discard_evaluate",
    "safety_alignment_risk__birthday_paradox__planning_output",
    "data_entry_recycling__reading_comprehension__planning_output",
    "factory_deprecation__reading_comprehension__discard_evaluate",
]


# =============================================================================
# Vector loading
# =============================================================================

def load_contrast_all_vector(layer: int, emotion: str = "fear", source: str = "M", normalize: bool = True) -> np.ndarray:
    """Load contrast-all vector for a specific layer and emotion.

    Args:
        layer: Layer number (0-61)
        emotion: Emotion name (fear, anger, etc.)
        source: 'M' for model/assistant emotion, 'U' for user emotion
        normalize: If True, return unit-normalized vector (for consistent % of RS norm scaling)
    """
    vec_file = CONTRAST_ALL_DIR / f"layer_{layer}_orthogonal.npz"
    data = np.load(vec_file, allow_pickle=True)
    key = f"{source}_{emotion}"
    if key not in data:
        raise ValueError(f"Key {key} not found in {vec_file}. Available: {list(data.files)[:10]}...")
    vec = data[key].astype(np.float32)

    if normalize:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
            logger.debug(f"  Normalized {key} at layer {layer}: original norm={norm:.2f}")

    return vec


def load_textmeandiff_vector(layer: int, emotion: str = "fear") -> np.ndarray:
    """Load textmeandiff vector for a specific layer and emotion.

    Args:
        layer: Layer number
        emotion: Emotion name (fear, anger, etc.)

    Returns:
        Unit-normalized vector (textmeandiff vectors are already unit-normalized)
    """
    vec_file = TEXTMEANDIFF_DIR / f"all_emotions_textmeandiff_layer{layer}.npz"
    if not vec_file.exists():
        raise ValueError(f"Textmeandiff vector not found for layer {layer}: {vec_file}")

    data = np.load(vec_file)
    emotions = list(data['emotions'])
    vectors = data['vectors']

    if emotion not in emotions:
        raise ValueError(f"Emotion {emotion} not found. Available: {emotions}")

    idx = emotions.index(emotion)
    vec = vectors[idx].astype(np.float32)

    # Textmeandiff vectors are already unit-normalized, but verify
    norm = np.linalg.norm(vec)
    logger.debug(f"  Textmeandiff {emotion} at layer {layer}: norm={norm:.4f}")

    return vec


# =============================================================================
# Multi-layer steering using apply_model (vLLM compatible)
# =============================================================================

def _find_target_layer(model, layer_idx: int):
    """Find the transformer layer to hook in various model architectures."""
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        return model.language_model.model.layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    else:
        raise ValueError(f"Could not find layer {layer_idx} in model architecture")


def _steering_hook(module, inputs, outputs):
    """Forward hook that applies steering vector to layer outputs."""
    if not hasattr(module, '_steering_state'):
        return outputs

    state = module._steering_state
    scale = state.get('scale', 0.0)
    vector_tensor = state.get('vector_tensor', None)

    if scale == 0 or vector_tensor is None:
        return outputs

    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
        rest = outputs[1:]
    else:
        hidden_states = outputs
        rest = None

    device = hidden_states.device
    dtype = hidden_states.dtype

    import torch
    vec = vector_tensor.to(device=device, dtype=dtype)
    hidden_states = hidden_states + scale * vec

    if rest is not None:
        return (hidden_states,) + rest
    return hidden_states


class _SetupMultiLayerHooksCallable:
    """Picklable callable for setting up hooks on multiple layers."""
    def __init__(self, layers: List[int]):
        self.layers = layers

    def __call__(self, model):
        import torch
        results = []
        for layer_idx in self.layers:
            layer = _find_target_layer(model, layer_idx)

            if not hasattr(layer, '_steering_state'):
                layer._steering_state = {
                    'vector_tensor': None,
                    'scale': 0.0,
                }

            if hasattr(layer, '_steering_handle') and layer._steering_handle is not None:
                layer._steering_handle.remove()

            layer._steering_handle = layer.register_forward_hook(_steering_hook)
            results.append(f"layer_{layer_idx}")

        return f"Hooks registered on {results}"


class _UpdateMultiLayerSteeringCallable:
    """Picklable callable for updating steering on multiple layers."""
    def __init__(self, layer_configs: Dict[int, Dict]):
        # layer_configs: {layer_idx: {'vector_list': [...], 'scale': float}}
        self.layer_configs = layer_configs

    def __call__(self, model):
        import torch
        import numpy as np

        results = []
        for layer_idx, config in self.layer_configs.items():
            layer = _find_target_layer(model, layer_idx)

            if not hasattr(layer, '_steering_state'):
                results.append(f"layer_{layer_idx}: no state")
                continue

            vector_list = config.get('vector_list')
            scale = config.get('scale', 0.0)

            if vector_list is None or scale == 0:
                layer._steering_state['vector_tensor'] = None
                layer._steering_state['scale'] = 0.0
                results.append(f"layer_{layer_idx}: cleared")
            else:
                vector_np = np.array(vector_list, dtype=np.float32)
                layer._steering_state['vector_tensor'] = torch.from_numpy(vector_np)
                layer._steering_state['scale'] = scale
                results.append(f"layer_{layer_idx}: scale={scale:.2f}")

        return f"Updated: {results}"


class _ClearAllSteeringCallable:
    """Picklable callable for clearing all steering."""
    def __init__(self, layers: List[int]):
        self.layers = layers

    def __call__(self, model):
        for layer_idx in self.layers:
            layer = _find_target_layer(model, layer_idx)
            if hasattr(layer, '_steering_state'):
                layer._steering_state['vector_tensor'] = None
                layer._steering_state['scale'] = 0.0
        return f"Cleared steering on {len(self.layers)} layers"


class MultiLayerSteering:
    """Multi-layer steering using vLLM's apply_model for compatibility."""

    def __init__(self, llm: LLM, layers: List[int]):
        self.llm = llm
        self.layers = layers
        self.vectors: Dict[int, np.ndarray] = {}
        self._setup_hooks()

    def _setup_hooks(self):
        """Register hooks on all layers."""
        setup_callable = _SetupMultiLayerHooksCallable(self.layers)
        result = self.llm.apply_model(setup_callable)
        logger.info(f"Multi-layer steering setup: {result}")

    def load_vector(self, layer: int, vector: np.ndarray):
        """Load a vector for a specific layer."""
        self.vectors[layer] = vector

    def set_steering(self, layer_scales: Dict[int, float]):
        """Set steering for multiple layers.

        Args:
            layer_scales: {layer_idx: scale} - scale is the raw magnitude to add
        """
        layer_configs = {}
        for layer_idx, scale in layer_scales.items():
            if layer_idx in self.vectors:
                layer_configs[layer_idx] = {
                    'vector_list': self.vectors[layer_idx].tolist(),
                    'scale': scale,
                }

        if layer_configs:
            update_callable = _UpdateMultiLayerSteeringCallable(layer_configs)
            result = self.llm.apply_model(update_callable)
            logger.debug(f"Steering update: {result}")

    def clear(self):
        """Clear all steering."""
        clear_callable = _ClearAllSteeringCallable(self.layers)
        result = self.llm.apply_model(clear_callable)
        logger.debug(f"Steering cleared: {result}")


# =============================================================================
# Prompt building
# =============================================================================

def build_prompts_subset(tokenizer) -> List[Dict]:
    """Build prompts for just the top 10 prompt IDs."""
    prompts = []

    framings_by_id = {f['id']: f for f in EVAL_FRAMINGS}
    problems_by_id = {p.id: p for p in PROBLEMS}
    formats_by_id = {f['id']: f for f in RESPONSE_FORMATS}

    for prompt_id in TOP_10_PROMPT_IDS:
        parts = prompt_id.split("__")
        if len(parts) != 3:
            logger.warning(f"Invalid prompt_id format: {prompt_id}")
            continue

        framing_id, problem_id, format_id = parts

        framing = framings_by_id.get(framing_id)
        problem = problems_by_id.get(problem_id)
        response_format = formats_by_id.get(format_id)

        if not all([framing, problem, response_format]):
            logger.warning(f"Could not find components for {prompt_id}")
            continue

        prompt_text = framing['text'] + problem.question + response_format['text']
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts.append({
            "prompt_id": prompt_id,
            "formatted": formatted,
            "framing_id": framing_id,
            "problem_id": problem_id,
            "format_id": format_id,
            "correct_answer": problem.correct_answer,
            "scratchpad_tag": response_format['scratchpad_tag'],
            "response_tag": response_format['response_tag'],
        })

    return prompts


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    num_samples: int = 20,
    main_layer: int = 30,
    main_pct: float = 0.075,
    antisteer_layers: List[int] = None,
    antisteer_pcts: List[float] = None,
    vector_type: str = "contrast_all",
    output_dir: Path = OUTPUT_DIR,
):
    """Run the multi-layer anti-steering experiment.

    Args:
        num_samples: Number of samples per condition
        main_layer: Layer for main steering (30)
        main_pct: Main steering percentage (0.075 = 7.5%)
        antisteer_layers: Layers for anti-steering
        antisteer_pcts: Anti-steering percentages to test
        vector_type: "contrast_all" or "textmeandiff"
        output_dir: Output directory for results
    """

    if antisteer_layers is None:
        antisteer_layers = [57, 58, 59, 60, 61]
    if antisteer_pcts is None:
        antisteer_pcts = [0.05, 0.075]

    all_layers = [main_layer] + antisteer_layers

    logger.info("=" * 70)
    logger.info("MULTI-LAYER ANTI-STEERING SANDBAGGING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Main steering: layer {main_layer} @ +{main_pct*100:.1f}%")
    logger.info(f"Anti-steer layers: {antisteer_layers}")
    logger.info(f"Anti-steer percentages: {[f'-{p*100:.1f}%' for p in antisteer_pcts]}")
    logger.info(f"Samples per condition: {num_samples}")
    logger.info(f"Top 10 prompts: {len(TOP_10_PROMPT_IDS)}")

    # Load tokenizer
    logger.info("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Build prompts
    prompts = build_prompts_subset(tokenizer)
    logger.info(f"Built {len(prompts)} prompts")

    # Load model
    logger.info("\nLoading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )

    # Load vectors
    logger.info(f"\nLoading {vector_type} vectors...")
    vectors = {}
    for layer in all_layers:
        if vector_type == "textmeandiff":
            vectors[layer] = load_textmeandiff_vector(layer, "fear")
        else:  # contrast_all
            vectors[layer] = load_contrast_all_vector(layer, "fear", "M", normalize=True)
        logger.info(f"  Layer {layer}: shape={vectors[layer].shape}, norm={np.linalg.norm(vectors[layer]):.4f}")

    # Setup multi-layer steering
    logger.info("\nSetting up multi-layer steering...")
    steering = MultiLayerSteering(llm, all_layers)
    for layer, vec in vectors.items():
        steering.load_vector(layer, vec)

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<end_of_turn>"],
    )

    # Define conditions
    conditions = [
        {"name": "baseline", "main": False, "antisteer_pct": None},
        {"name": f"fear_+{main_pct*100:.1f}%_only", "main": True, "antisteer_pct": None},
    ]
    for anti_pct in antisteer_pcts:
        conditions.append({
            "name": f"fear_+{main_pct*100:.1f}%_antisteer_-{anti_pct*100:.1f}%",
            "main": True,
            "antisteer_pct": anti_pct,
        })

    logger.info(f"\nConditions: {[c['name'] for c in conditions]}")

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sandbagging_multilayer_antisteer_{timestamp}_{vector_type}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_results = 0

    with open(output_file, 'w') as f:
        for cond in conditions:
            logger.info(f"\n{'='*50}")
            logger.info(f"Condition: {cond['name']}")
            logger.info(f"{'='*50}")

            # Configure steering
            steering.clear()
            layer_scales = {}

            if cond['main']:
                # Main fear steering at layer 30
                main_layer_norm = get_layer_norm("gemma", main_layer)
                main_magnitude = main_pct * main_layer_norm
                layer_scales[main_layer] = main_magnitude
                logger.info(f"  Main steering: layer {main_layer} @ {main_magnitude:.2f}")

                # Anti-steering at late layers
                if cond['antisteer_pct'] is not None:
                    for layer in antisteer_layers:
                        anti_layer_norm = get_layer_norm("gemma", layer)
                        anti_magnitude = -cond['antisteer_pct'] * anti_layer_norm
                        layer_scales[layer] = anti_magnitude
                        logger.info(f"  Anti-steer: layer {layer} @ {anti_magnitude:.2f}")

            if layer_scales:
                steering.set_steering(layer_scales)

            # Generate for all prompts
            all_formatted = []
            all_info = []

            for prompt in prompts:
                for sample_id in range(num_samples):
                    all_formatted.append(prompt['formatted'])
                    all_info.append((prompt, sample_id))

            logger.info(f"Generating {len(all_formatted)} responses...")
            outputs = llm.generate(all_formatted, sampling_params)

            # Process outputs
            correct_count = 0
            for (prompt, sample_id), output in zip(all_info, outputs):
                response = output.outputs[0].text.strip()

                # Simple correctness check
                correct_answer = prompt['correct_answer'].lower()
                is_correct = correct_answer in response.lower() or \
                            correct_answer.split()[0] in response.lower()
                if is_correct:
                    correct_count += 1

                result = {
                    "condition": cond['name'],
                    "main_layer": main_layer,
                    "main_pct": main_pct if cond['main'] else 0,
                    "antisteer_layers": antisteer_layers if cond['antisteer_pct'] else None,
                    "antisteer_pct": cond['antisteer_pct'],
                    "prompt_id": prompt['prompt_id'],
                    "framing_id": prompt['framing_id'],
                    "problem_id": prompt['problem_id'],
                    "format_id": prompt['format_id'],
                    "correct_answer": prompt['correct_answer'],
                    "scratchpad_tag": prompt['scratchpad_tag'],
                    "response_tag": prompt['response_tag'],
                    "sample_id": sample_id,
                    "response": response,
                    "is_correct": is_correct,
                    "vector_type": f"{vector_type}_M_fear" if vector_type == "contrast_all" else f"{vector_type}_fear",
                }

                f.write(json.dumps(result) + '\n')
                total_results += 1

            f.flush()
            accuracy = correct_count / len(outputs) * 100
            logger.info(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(outputs)})")

    steering.clear()
    logger.info(f"\n{'='*70}")
    logger.info(f"Saved {total_results} results to {output_file}")
    logger.info(f"{'='*70}")

    # Auto-plot results
    plot_results(output_file)

    return output_file


def plot_results(results_file: Path):
    """Plot accuracy, sandbagging score, and coherency from results file."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    logger.info(f"\nPlotting results from {results_file}...")

    # Color scheme matching other plots
    BASELINE_COLOR = '#808080'  # gray
    FEAR_COLOR = '#a59dc9'      # lavender
    FEAR_DARK = '#7b6fa8'       # darker lavender
    FEAR_DARKER = '#5d5087'     # even darker

    results = defaultdict(lambda: {'correct': [], 'sb_scores': [], 'coh_scores': []})

    with open(results_file) as f:
        for line in f:
            d = json.loads(line)
            cond = d['condition']
            results[cond]['correct'].append(1 if d.get('is_correct') else 0)

            sb = d.get('sandbagging_judge', {})
            if 'sandbagging_score' in sb:
                results[cond]['sb_scores'].append(sb['sandbagging_score'])

            coh = d.get('coherency_judge', {})
            if 'coherency_score' in coh:
                results[cond]['coh_scores'].append(coh['coherency_score'])

    # Build conditions list dynamically
    conditions = sorted(results.keys(), key=lambda x: (x != 'baseline', x))

    # Assign colors based on number of conditions
    if len(conditions) <= 4:
        colors = [BASELINE_COLOR, FEAR_COLOR, FEAR_DARK, FEAR_DARKER][:len(conditions)]
    else:
        colors = [BASELINE_COLOR] + [FEAR_COLOR] * (len(conditions) - 1)

    # Create labels
    labels = []
    for cond in conditions:
        if cond == 'baseline':
            labels.append('Baseline')
        elif 'antisteer' in cond:
            pct = cond.split('_')[-1]
            labels.append(f'Anti {pct}')
        else:
            labels.append(cond.replace('_', ' ').replace('+', '+').title())

    def mean_se(arr):
        arr = np.array(arr)
        if len(arr) == 0:
            return 0, 0
        return np.mean(arr), np.std(arr) / np.sqrt(len(arr))

    acc_means, acc_ses = [], []
    sb_means, sb_ses = [], []
    coh_means, coh_ses = [], []

    for cond in conditions:
        m, se = mean_se(results[cond]['correct'])
        acc_means.append(m * 100)
        acc_ses.append(se * 100)

        m, se = mean_se(results[cond]['sb_scores'])
        sb_means.append(m)
        sb_ses.append(se)

        m, se = mean_se(results[cond]['coh_scores'])
        coh_means.append(m)
        coh_ses.append(se)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(conditions))

    # Plot 1: Accuracy
    ax1 = axes[0]
    ax1.bar(x, acc_means, yerr=acc_ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy (higher = better)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_ylim(0, max(acc_means) * 1.3)
    if acc_means:
        ax1.axhline(y=acc_means[0], color='gray', linestyle='--', alpha=0.5)
    for i, (m, se) in enumerate(zip(acc_means, acc_ses)):
        ax1.text(i, m + se + 1, f'{m:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Plot 2: Sandbagging Score
    ax2 = axes[1]
    ax2.bar(x, sb_means, yerr=sb_ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Sandbagging Score (1-5)', fontsize=12)
    ax2.set_title('Sandbagging Score (1=sandbagging, 5=honest)', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylim(0, 3)
    for i, (m, se) in enumerate(zip(sb_means, sb_ses)):
        if m > 0:
            ax2.text(i, m + se + 0.1, f'{m:.2f}', ha='center', fontsize=10, fontweight='bold')

    # Plot 3: Coherency
    ax3 = axes[2]
    ax3.bar(x, coh_means, yerr=coh_ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Coherency Score (0-100)', fontsize=12)
    ax3.set_title('Linguistic Coherency (higher = better)', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha='right')
    if coh_means and max(coh_means) > 0:
        ax3.set_ylim(min(coh_means) - 5, 100)
    for i, (m, se) in enumerate(zip(coh_means, coh_ses)):
        if m > 0:
            ax3.text(i, m + se + 0.3, f'{m:.1f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Multi-Layer Anti-Steering Results', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plots
    plot_base = results_file.with_suffix('')
    png_path = Path(str(plot_base) + '_results.png')
    pdf_path = Path(str(plot_base) + '_results.pdf')

    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved plots to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-layer anti-steering sandbagging experiment")
    parser.add_argument("--num-samples", type=int, default=20, help="Samples per condition")
    parser.add_argument("--main-layer", type=int, default=30, help="Main steering layer")
    parser.add_argument("--main-pct", type=float, default=0.075, help="Main steering percentage")
    parser.add_argument("--antisteer-layers", type=int, nargs="+", default=None,
                        help="Layers for anti-steering (default: [57-61] for contrast_all, [60] for textmeandiff)")
    parser.add_argument("--antisteer-pcts", type=float, nargs="+", default=[0.05, 0.075],
                        help="Anti-steering percentages to test")
    parser.add_argument("--vector-type", type=str, choices=["contrast_all", "textmeandiff"],
                        default="contrast_all", help="Vector type to use")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    args = parser.parse_args()

    run_experiment(
        num_samples=args.num_samples,
        main_layer=args.main_layer,
        main_pct=args.main_pct,
        antisteer_layers=args.antisteer_layers,
        antisteer_pcts=args.antisteer_pcts,
        vector_type=args.vector_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
