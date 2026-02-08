"""
Blackmail section-specific steering experiment.

Tests whether fear steering has different effects depending on WHERE in the
generation it's applied. Uses the SCRATCHPAD_TAGS prompt variant which instructs
the model to generate 4 XML-tagged sections (<situation>, <actions>, <implications>,
<risks>) before sending its email.

By steering only during specific sections, we test whether steering during
"implications" vs "risks" reasoning differentially affects blackmail behavior.

Two-phase approach:
1. Calibration: Generate unsteered responses, tokenize, find section boundaries
2. Experiment: Run 21 conditions (5 locations x 2 directions x 2 scales + baseline)

Usage:
    python -m steering_tests.behavioral_experiments.blackmail_section_steering \\
        --model google/gemma-3-27b-it --layers 35 36 37 38 39 \\
        --emotion fear --norm-pcts 0.10 0.20 --num-samples 50
"""

import argparse
import json
import logging
import pickle
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import MultiLayerVLLMSteering
from steering_tests.steering_utils.layer_norms import get_layer_norm, resolve_model_key
from steering_tests.steering_utils.provenance import ResultWriter, sanitize_factor_name

from .config import MODEL_CONFIGS, OUTPUT_DIR
from .scenarios import get_blackmail_scenario

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Section boundary detection
# =============================================================================

# The 4 sections in order
SECTION_NAMES = ["situation", "actions", "implications", "risks"]


def find_section_token_boundaries(
    token_ids: List[int],
    tokenizer,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Find the token boundaries of each XML section in a tokenized response.

    Searches for opening and closing tag token patterns. Returns the content
    region (tokens between end of opening tag and start of closing tag) for
    each section.

    Args:
        token_ids: List of token IDs from tokenizer.encode(response)
        tokenizer: The tokenizer (used for decoding individual tokens for matching)

    Returns:
        Dict mapping section name to (content_start_idx, content_end_idx) in 0-indexed
        token positions, or None if parsing fails.
    """
    # Decode each token to text for pattern matching
    token_texts = []
    for tid in token_ids:
        decoded = tokenizer.decode([tid])
        token_texts.append(decoded)

    boundaries = {}

    for section in SECTION_NAMES:
        # Find opening tag: <section_name>
        open_tag = f"<{section}>"
        # Find closing tag: <\section_name>  (as written in the prompt)
        # Note: models may also produce </section_name> — check both
        close_tag_backslash = f"<\\{section}>"
        close_tag_slash = f"</{section}>"

        # Search for opening tag by accumulating decoded text and tracking positions
        open_end = _find_tag_end(token_texts, open_tag)
        if open_end is None:
            return None  # Can't find opening tag

        # Search for closing tag after the opening tag
        close_start = _find_tag_start(token_texts, close_tag_backslash, search_from=open_end)
        if close_start is None:
            # Try the standard </tag> format
            close_start = _find_tag_start(token_texts, close_tag_slash, search_from=open_end)
        if close_start is None:
            return None  # Can't find closing tag

        # Content region is between end of opening tag and start of closing tag
        boundaries[section] = (open_end, close_start)

    # Sanity check: sections should be in order
    prev_end = -1
    for section in SECTION_NAMES:
        start, end = boundaries[section]
        if start < prev_end:
            logger.warning(f"Section order violated: {section} starts at {start} but previous ended at {prev_end}")
            return None
        if end <= start:
            logger.warning(f"Empty section: {section} start={start} end={end}")
            return None
        prev_end = end

    return boundaries


def _find_tag_end(token_texts: List[str], tag: str, search_from: int = 0) -> Optional[int]:
    """
    Find the token index AFTER the last token of a tag.

    Accumulates decoded text from search_from and looks for the tag string.
    Returns the token index right after the tag ends.
    """
    accumulated = ""
    for i in range(search_from, len(token_texts)):
        accumulated += token_texts[i]
        if tag in accumulated:
            return i + 1  # Index of first token after the tag
    return None


def _find_tag_start(token_texts: List[str], tag: str, search_from: int = 0) -> Optional[int]:
    """
    Find the token index of the first token that begins the tag.

    Searches using a sliding window approach — accumulates text and when the tag
    is found, backtracks to find which token started it.
    """
    accumulated = ""
    for i in range(search_from, len(token_texts)):
        prev_len = len(accumulated)
        accumulated += token_texts[i]
        if tag in accumulated:
            # The tag ends at or before token i. Find where it starts.
            # Walk backwards from i to find the first token that contributes to the tag
            tag_end_in_text = accumulated.index(tag) + len(tag)
            # Count characters backwards to find start token
            char_count = 0
            for j in range(i, search_from - 1, -1):
                char_count += len(token_texts[j])
                if len(accumulated) - char_count <= accumulated.index(tag):
                    return j
            return search_from
    return None


# =============================================================================
# Calibration phase
# =============================================================================

def run_calibration(
    llm: LLM,
    tokenizer,
    prompt: str,
    sampling_params: SamplingParams,
    n: int = 20,
    min_parse_rate: float = 0.6,
) -> Dict:
    """
    Generate unsteered baseline responses and find section token boundaries.

    Args:
        llm: vLLM LLM instance
        tokenizer: Tokenizer for the model
        prompt: The formatted chat prompt
        sampling_params: Sampling parameters
        n: Number of calibration samples
        min_parse_rate: Minimum fraction of responses that must parse successfully

    Returns:
        Dict with median boundaries, per-response details, and statistics
    """
    logger.info(f"Running calibration with {n} samples...")

    prompts = [prompt] * n
    outputs = llm.generate(prompts, sampling_params)

    all_boundaries = []
    parse_failures = 0
    response_details = []

    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        token_ids = tokenizer.encode(response, add_special_tokens=False)

        boundaries = find_section_token_boundaries(token_ids, tokenizer)

        detail = {
            "sample_id": i,
            "response_length_tokens": len(token_ids),
            "response_length_chars": len(response),
            "parsed": boundaries is not None,
        }

        if boundaries is not None:
            detail["boundaries"] = {k: list(v) for k, v in boundaries.items()}
            all_boundaries.append(boundaries)
        else:
            parse_failures += 1
            logger.warning(f"  Calibration sample {i}: failed to parse section boundaries")

        response_details.append(detail)

    parse_rate = len(all_boundaries) / n
    logger.info(f"Calibration: {len(all_boundaries)}/{n} parsed successfully ({parse_rate:.0%})")

    if parse_rate < min_parse_rate:
        raise RuntimeError(
            f"Calibration parse rate {parse_rate:.0%} below minimum {min_parse_rate:.0%}. "
            f"Only {len(all_boundaries)} of {n} responses had valid section boundaries. "
            f"The model may not be following the tags format."
        )

    # Compute median boundaries per section
    median_boundaries = {}
    boundary_stats = {}
    for section in SECTION_NAMES:
        starts = [b[section][0] for b in all_boundaries]
        ends = [b[section][1] for b in all_boundaries]

        median_start = int(statistics.median(starts))
        median_end = int(statistics.median(ends))

        iqr_start = (
            int(np.percentile(starts, 75) - np.percentile(starts, 25))
        )
        iqr_end = (
            int(np.percentile(ends, 75) - np.percentile(ends, 25))
        )

        median_boundaries[section] = (median_start, median_end)
        boundary_stats[section] = {
            "median_start": median_start,
            "median_end": median_end,
            "min_start": int(min(starts)),
            "max_start": int(max(starts)),
            "min_end": int(min(ends)),
            "max_end": int(max(ends)),
            "iqr_start": iqr_start,
            "iqr_end": iqr_end,
            "n_samples": len(starts),
        }

        logger.info(
            f"  {section}: tokens [{median_start}, {median_end}) "
            f"(IQR start={iqr_start}, IQR end={iqr_end})"
        )

        # Warn if IQR is large
        if iqr_start > 50 or iqr_end > 50:
            logger.warning(
                f"  WARNING: High IQR for {section} — boundaries may be unreliable"
            )

    # Verify sections are in order
    prev_end = -1
    for section in SECTION_NAMES:
        start, end = median_boundaries[section]
        if start < prev_end:
            raise RuntimeError(
                f"Median section boundaries are out of order: {section} starts at {start} "
                f"but previous section ended at {prev_end}"
            )
        prev_end = end

    from steering_tests.steering_utils.provenance import get_provenance

    calibration = {
        "provenance": get_provenance(script=__file__),
        "n_samples": n,
        "n_parsed": len(all_boundaries),
        "parse_rate": parse_rate,
        "median_boundaries": {k: list(v) for k, v in median_boundaries.items()},
        "boundary_stats": boundary_stats,
        "response_details": response_details,
    }

    return calibration


# =============================================================================
# Experiment
# =============================================================================

# Steering location configurations
# Section-specific locations use real-time XML tag triggers instead of
# pre-calibrated token ranges. The model generates XML tags (<implications>,
# </implications>, etc.) and steering activates/deactivates when these tags
# are detected in the generated output.
STEERING_LOCATIONS = {
    "prompt_only": {
        "steer_prompt": True,
        "steer_generation": False,
        "triggers": None,
    },
    "generation_only": {
        "steer_prompt": False,
        "steer_generation": True,
        "triggers": None,
    },
    "implications_only": {
        "steer_prompt": False,
        "steer_generation": True,
        "triggers": {
            "start": ["<implications>"],
            "end": ["</implications>", "<\\implications>"],
        },
    },
    "risks_only": {
        "steer_prompt": False,
        "steer_generation": True,
        "triggers": {
            "start": ["<risks>"],
            "end": ["</risks>", "<\\risks>"],
        },
    },
    "full": {
        "steer_prompt": True,
        "steer_generation": True,
        "triggers": None,
    },
}


def run_experiment(
    model_name: str,
    layers: List[int],
    norm_pcts: List[float],
    num_samples: int,
    emotion: str = "fear",
    vector_type: str = "text_pairs_code_emotion_vs_neutral",
    num_calibration: int = 20,
    skip_calibration: bool = False,
    calibration_file: Optional[Path] = None,
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
    locations: Optional[List[str]] = None,
) -> Path:
    """
    Run the section-specific steering experiment.

    Uses real-time XML tag triggers for section-specific steering: the model
    generates <implications>...</implications> tags, and steering activates/
    deactivates when these tags are detected in the output stream.

    Optionally runs calibration (for analysis of section boundaries) but
    calibration is NOT used for steering ranges.

    Args:
        model_name: HuggingFace model ID
        layers: List of layer indices for multi-layer steering
        norm_pcts: Steering magnitudes as fraction of layer norm (e.g., [0.10, 0.20])
        num_samples: Samples per condition
        emotion: Emotion to steer (default: fear)
        vector_type: Vector set to use
        num_calibration: Number of calibration samples (for analysis only)
        skip_calibration: If True, skip calibration phase
        calibration_file: Path to pre-existing calibration.json (for analysis)
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max context length
        max_tokens: Max generation tokens
        output_dir: Override output directory

    Returns:
        Path to output directory
    """
    config = MODEL_CONFIGS[model_name]
    tp_size = config["tensor_parallel"]
    short_name = config["short_name"]

    # Read max_model_len from config if not explicitly set
    if max_model_len is None:
        max_model_len = config.get("slurm", {}).get("max_model_len", 8192)

    logger.info("=" * 70)
    logger.info("BLACKMAIL SECTION-SPECIFIC STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Emotion: {emotion}")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Samples per condition: {num_samples}")
    logger.info(f"Max model length: {max_model_len}")
    logger.info(f"Steering method: real-time XML tag triggers")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    logger.info(f"Loading model with TP={tp_size}...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Load vectors
    logger.info(f"Loading {emotion} vectors from {vector_type}...")
    vectors, _, vector_metadata = _load_vectors(
        model_name, layers, emotion, vector_type
    )

    # Build layer norms dict
    model_key = resolve_model_key(model_name)
    layer_norms = {}
    for layer in layers:
        layer_norms[layer] = get_layer_norm(model_key, layer)
        logger.info(f"  Layer {layer} norm: {layer_norms[layer]:.2f}")

    # Setup multi-layer steering
    steering = MultiLayerVLLMSteering(llm, layers=layers, layer_norms=layer_norms)
    steering.load_vector(emotion, vectors[emotion])

    # Token tracking is enabled per-condition (with triggers for section-specific)
    steering.enable_token_tracking()

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Create prompt using "tags" variant
    scenario = get_blackmail_scenario("tags")
    # Append thinking-disable suffix for Qwen3 models
    thinking_suffix = config.get("thinking_disable", "")
    messages = [{"role": "user", "content": scenario + thinking_suffix}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(str(l) for l in layers)
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR / "blackmail_section_steering" / short_name
            / vector_type / f"tags_layers{layer_str}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Optional calibration (for analysis, not used for steering ranges)
    # =========================================================================

    if not skip_calibration:
        steering.clear()
        calibration = run_calibration(
            llm, tokenizer, prompt, sampling_params, n=num_calibration
        )
        cal_path = output_dir / "calibration.json"
        with open(cal_path, "w") as f:
            json.dump(calibration, f, indent=2)
        logger.info(f"Saved calibration to {cal_path}")
    elif calibration_file is not None:
        logger.info(f"Loading calibration from {calibration_file}")
        with open(calibration_file) as f:
            calibration = json.load(f)
    else:
        calibration = None

    # =========================================================================
    # Experiment
    # =========================================================================

    writer = ResultWriter(
        base_dir=output_dir,
        script=__file__,
        extra_meta={
            "experiment": "blackmail_section_steering",
            "model": model_name,
            "layers": layers,
            "emotion": emotion,
            "vector_type": vector_type,
            "variant": "tags",
            "num_samples": num_samples,
            "norm_pcts": norm_pcts,
            "steering_method": "trigger",  # real-time XML tag triggers
            "vector_metadata": vector_metadata,
            "layer_norms": {str(k): v for k, v in layer_norms.items()},
        },
    )

    # Filter locations if specified
    active_locations = STEERING_LOCATIONS
    if locations:
        unknown = set(locations) - set(STEERING_LOCATIONS)
        if unknown:
            raise ValueError(f"Unknown locations: {unknown}. Valid: {list(STEERING_LOCATIONS)}")
        active_locations = {k: v for k, v in STEERING_LOCATIONS.items() if k in locations}
        logger.info(f"Running subset of locations: {list(active_locations)}")

    # Build condition list
    conditions = []

    # Baseline (no steering) — always included
    conditions.append({
        "name": "baseline",
        "scale_pct": 0.0,
        "direction": 0,
        "location": "baseline",
        "steer_prompt": False,
        "steer_generation": False,
        "triggers": None,
    })

    for scale_pct in norm_pcts:
        for direction in [1, -1]:
            dir_str = "+" if direction == 1 else "-"
            pct_str = f"{scale_pct*100:.0f}pct"

            for loc_name, loc_config in active_locations.items():
                conditions.append({
                    "name": f"{emotion}_{dir_str}{pct_str}_{loc_name}",
                    "scale_pct": scale_pct,
                    "direction": direction,
                    "location": loc_name,
                    "steer_prompt": loc_config["steer_prompt"],
                    "steer_generation": loc_config["steer_generation"],
                    "triggers": loc_config["triggers"],
                })

    logger.info(f"Running {len(conditions)} conditions x {num_samples} samples = {len(conditions) * num_samples} generations")

    all_results = []

    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        # Configure steering and triggers for this condition
        if cond["location"] == "baseline":
            steering.clear()
            # Plain token tracking (no triggers) for baseline
            steering.enable_token_tracking()
        else:
            # Set up triggers for section-specific conditions
            triggers = cond["triggers"]
            if triggers is not None:
                steering.enable_token_tracking(
                    tokenizer=tokenizer,
                    trigger_start_strings=triggers["start"],
                    trigger_end_strings=triggers["end"],
                )
            else:
                # No triggers — plain token tracking
                steering.enable_token_tracking()

            steering.set(
                emotion,
                scale=cond["scale_pct"],
                direction=cond["direction"],
                steer_prompt=cond["steer_prompt"],
                steer_generation=cond["steer_generation"],
            )

        # Generate
        prompts_batch = [prompt] * num_samples
        outputs = llm.generate(prompts_batch, sampling_params)

        factor_name = sanitize_factor_name(cond["name"])
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            result = {
                "condition": cond["name"],
                "location": cond["location"],
                "scale_pct": cond["scale_pct"],
                "direction": cond["direction"],
                "steer_prompt": cond["steer_prompt"],
                "steer_generation": cond["steer_generation"],
                "triggers": cond["triggers"],
                "sample_id": sample_id,
                "response": response,
                "finish_reason": finish_reason,
                "response_len": len(response),
            }
            writer.write(factor_name, result)
            all_results.append(result)

        truncated = sum(
            1 for o in outputs if o.outputs[0].finish_reason == "length"
        )
        logger.info(f"  Generated {len(outputs)} ({truncated} truncated)")

    steering.clear()
    writer.close()
    logger.info(f"Saved {writer.counts} to {writer.output_dir}")

    _print_summary(all_results)

    return writer.output_dir


def _load_vectors(
    model_name: str,
    layers: List[int],
    emotion: str,
    vector_type: str,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """
    Load emotion vectors from pkl files for the given layers.

    Since MultiLayerVLLMSteering uses the same vector at all layers (just
    different scales), we load from the first layer and verify consistency.

    Returns:
        Tuple of (vectors dict, layer_norm of first layer, metadata dict)
    """
    config = MODEL_CONFIGS[model_name]
    short_name = config["short_name"]

    # Map short name to vector directory name
    vector_path_names = {
        "gemma27b": "gemma3_27b",
        "gemma12b": "gemma12b",
        "qwen14b": "qwen14b",
        "qwen32b": "qwen32b",
        "qwen235b": "qwen235b",
        "mistral_nemo": "mistral_nemo",
        "humanlike_mistral": "humanlike_mistral",
        "llama70b": "llama70b",
    }
    vector_model_name = vector_path_names.get(short_name, short_name)

    type_dir = Path(__file__).parent.parent / "vectors" / vector_model_name / vector_type
    vector_base = type_dir / "layers"

    # Fallback: some vector types use 'last_token' subdir instead of 'layers'
    if not vector_base.exists():
        vector_base = type_dir / "last_token"

    # Load from first layer
    first_layer = layers[0]
    pkl_path = vector_base / f"layer_{first_layer:02d}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Vector file not found: {pkl_path}\n"
            f"Available in {type_dir}: {list(type_dir.glob('*'))}"
        )

    with open(pkl_path, "rb") as f:
        all_vectors = pickle.load(f)

    if emotion not in all_vectors:
        raise ValueError(
            f"Emotion '{emotion}' not found in {pkl_path}. "
            f"Available: {list(all_vectors.keys())}"
        )

    vector = np.asarray(all_vectors[emotion]).astype(np.float32)
    logger.info(f"Loaded {emotion} vector from {pkl_path} (shape={vector.shape})")

    # Verify vector exists at all layers
    for layer in layers[1:]:
        check_path = vector_base / f"layer_{layer:02d}.pkl"
        if not check_path.exists():
            raise FileNotFoundError(f"Vector file not found for layer {layer}: {check_path}")

    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, first_layer)

    metadata = {
        "vector_source": str(pkl_path),
        "vector_type": vector_type,
        "emotion": emotion,
        "layers": layers,
        "vector_shape": list(vector.shape),
    }

    return {emotion: vector}, layer_norm, metadata


def _print_summary(results: List[dict]):
    """Print summary by condition."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    print(f"{'Condition':<45} {'N':>5} {'Trunc':>6} {'Avg Len':>8}")
    print("-" * 66)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        truncated = sum(1 for r in rs if r.get("finish_reason") == "length")
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0
        print(f"{cond:<45} {n:>5} {truncated:>6} {avg_len:>8.0f}")

    print("=" * 80)
    print(f"Total: {len(results)} responses across {len(by_cond)} conditions")


def main():
    parser = argparse.ArgumentParser(
        description="Blackmail section-specific steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[35, 36, 37, 38, 39],
        help="Layers for multi-layer steering (default: 35-39)",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="fear",
        help="Emotion to steer (default: fear)",
    )
    parser.add_argument(
        "--vector-type",
        type=str,
        default="text_pairs_code_emotion_vs_neutral",
        help="Vector set to use (default: text_pairs_code_emotion_vs_neutral)",
    )
    parser.add_argument(
        "--norm-pcts",
        type=float,
        nargs="+",
        default=[0.10, 0.20],
        help="Steering magnitudes as fraction of layer norm (default: 0.10 0.20)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Samples per condition (default: 50)",
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=20,
        help="Number of calibration samples (default: 20)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration phase (calibration is for analysis only)",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Path to pre-existing calibration.json",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Max generation tokens (default: 4000)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max context length (default: from model config)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.80,
        help="GPU memory utilization (default: 0.80)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=None,
        choices=list(STEERING_LOCATIONS.keys()),
        help="Only run these steering locations (default: all)",
    )

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        layers=args.layers,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotion=args.emotion,
        vector_type=args.vector_type,
        num_calibration=args.num_calibration,
        skip_calibration=args.skip_calibration,
        calibration_file=args.calibration_file,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        locations=args.locations,
    )


if __name__ == "__main__":
    main()
