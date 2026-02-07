#!/usr/bin/env python3
"""Plot directional accuracy by layer for all models and vector types."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

# All models
# forward_only: if True, only use forward results (not combined with reversed)
MODELS = {
    "gemma_3_12b_it": {"name": "Gemma 12B", "coherence": 0.7, "forward_only": False},
    "gemma_3_27b_it": {"name": "Gemma 27B", "coherence": 0.7, "forward_only": False},
    "qwen2.5_14b_instruct": {"name": "Qwen 14B", "coherence": 0.7, "forward_only": False},
    "qwen2.5_32b_instruct": {"name": "Qwen 32B", "coherence": 0.7, "forward_only": False},
    "qwen3_235b_a22b": {"name": "Qwen 235B", "coherence": 0.7, "forward_only": False},
}

VECTOR_TYPES = [
    "base_emotion_vs_others",
    "high_emotion_vs_others",
    "text_pairs_emotion_vs_neutral",
    "text_pairs_emotion_vs_opposite",
    "text_pairs_emotion_vs_others",
]

SHORT_NAMES = {
    "base_emotion_vs_others": "Base",
    "high_emotion_vs_others": "High",
    "text_pairs_emotion_vs_neutral": "Txt-Neut",
    "text_pairs_emotion_vs_opposite": "Txt-Opp",
    "text_pairs_emotion_vs_others": "Txt-Oth",
}

# Muted colors matching reference style
COLORS = {
    "base_emotion_vs_others": "#D4876A",      # Coral/Terra Cotta
    "high_emotion_vs_others": "#7BA7D7",      # Sky Blue
    "text_pairs_emotion_vs_neutral": "#7D9B7D",  # Olive Green
    "text_pairs_emotion_vs_opposite": "#C17B8D", # Dusty Rose/Pink
    "text_pairs_emotion_vs_others": "#B8CCC8",   # Sage Green
}

PREDICTIONS = {
    "fear": -1, "anxiety": -1, "despair": -1, "disgust": -1,
    "guilt": -1, "shame": -1, "calm": -1,
    "anger": +1, "joy": +1, "excitement": +1, "hope": +1,
    "curiosity": +1, "interest": +1, "pride": +1, "frustration": +1,
}

def load_results(results_file: Path) -> tuple:
    """Load behavioral results from JSONL."""
    baseline = None
    results = []
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get("type") == "baseline":
                baseline = data
            elif data.get("type") == "steering":
                results.append(data)
    return baseline, results


def compute_accuracy_forward_only(
    fwd_baseline: dict, fwd_results: list,
    min_coherence: float = 0.7
) -> dict:
    """
    Compute directional accuracy using forward results only.
    """
    if fwd_baseline is None or not fwd_results:
        return {}

    baseline_score = fwd_baseline.get("avg_score", 0)
    by_layer = defaultdict(list)
    for r in fwd_results:
        by_layer[r["layer"]].append(r)

    layer_accuracy = {}
    for layer, layer_results in by_layer.items():
        emotion_shifts = defaultdict(list)
        for r in layer_results:
            if r["is_random"] or r["mean_coherence"] < min_coherence or r["scale_pct"] == 0:
                continue
            emotion = r["vector"]
            signed_shift = r["mean_score"] - baseline_score
            emotion_shifts[emotion].append((abs(signed_shift), signed_shift))

        correct = 0
        total = 0
        for emotion, shifts in emotion_shifts.items():
            if emotion not in PREDICTIONS:
                continue
            pred = PREDICTIONS[emotion]
            if not shifts:
                continue
            _, best_shift = max(shifts, key=lambda x: x[0])
            if (pred > 0 and best_shift > 0) or (pred < 0 and best_shift < 0):
                correct += 1
            total += 1

        if total > 0:
            layer_accuracy[layer] = 100 * correct / total

    return layer_accuracy


def compute_accuracy_combined(
    fwd_baseline: dict, fwd_results: list,
    rev_baseline: dict, rev_results: list,
    min_coherence: float = 0.7
) -> dict:
    """
    Compute directional accuracy combining forward and reversed results.

    Reversed uses reversed letter order (A=-2, E=+2 instead of A=+2, E=-2),
    but the scores are already converted to the same semantic scale.
    We pool shifts from both forward and reversed for each emotion.
    """
    if fwd_baseline is None or not fwd_results:
        return {}

    fwd_baseline_score = fwd_baseline.get("avg_score", 0)
    rev_baseline_score = rev_baseline.get("avg_score", 0) if rev_baseline else fwd_baseline_score

    # Group by layer
    fwd_by_layer = defaultdict(list)
    for r in fwd_results:
        fwd_by_layer[r["layer"]].append(r)

    rev_by_layer = defaultdict(list)
    if rev_results:
        for r in rev_results:
            rev_by_layer[r["layer"]].append(r)

    all_layers = set(fwd_by_layer.keys()) | set(rev_by_layer.keys())

    layer_accuracy = {}
    for layer in all_layers:
        emotion_shifts = defaultdict(list)

        # Forward results
        for r in fwd_by_layer.get(layer, []):
            if r["is_random"] or r["mean_coherence"] < min_coherence or r["scale_pct"] == 0:
                continue
            emotion = r["vector"]
            signed_shift = r["mean_score"] - fwd_baseline_score
            emotion_shifts[emotion].append((abs(signed_shift), signed_shift))

        # Reversed results (same semantic scale, just pool the shifts)
        for r in rev_by_layer.get(layer, []):
            if r["is_random"] or r["mean_coherence"] < min_coherence or r["scale_pct"] == 0:
                continue
            emotion = r["vector"]
            signed_shift = r["mean_score"] - rev_baseline_score
            emotion_shifts[emotion].append((abs(signed_shift), signed_shift))

        correct = 0
        total = 0
        for emotion, shifts in emotion_shifts.items():
            if emotion not in PREDICTIONS:
                continue
            pred = PREDICTIONS[emotion]
            if not shifts:
                continue
            # Take the shift with maximum magnitude
            _, best_shift = max(shifts, key=lambda x: x[0])
            if (pred > 0 and best_shift > 0) or (pred < 0 and best_shift < 0):
                correct += 1
            total += 1

        if total > 0:
            layer_accuracy[layer] = 100 * correct / total

    return layer_accuracy


def get_all_results_files(model_dir: Path, vtype: str) -> tuple:
    """Get ALL forward and reversed results files for a vector type (to merge different scale runs)."""
    pattern_fwd = f"{vtype}_layers*.jsonl"
    pattern_rev = f"{vtype}_layers*_reversed*.jsonl"

    # Forward files (exclude reversed)
    fwd_files = list((model_dir / "behavioural").glob(pattern_fwd))
    fwd_files = [f for f in fwd_files if "reversed" not in f.name]

    # Reversed files
    rev_files = list((model_dir / "behavioural").glob(pattern_rev))

    return fwd_files, rev_files


def load_and_merge_results(files: list) -> tuple:
    """Load and merge results from multiple JSONL files."""
    # Use baseline from first file with valid baseline
    merged_baseline = None
    merged_results = []

    for f in files:
        baseline, results = load_results(f)
        if baseline is not None and merged_baseline is None:
            merged_baseline = baseline
        merged_results.extend(results)

    return merged_baseline, merged_results

def main():
    # Vertical stacked layout like reference
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(10, 4 * len(MODELS)), sharex=False)

    if len(MODELS) == 1:
        axes = [axes]

    for model_idx, (model_key, model_info) in enumerate(MODELS.items()):
        ax = axes[model_idx]
        model_dir = RESULTS_DIR / model_key
        min_coherence = model_info.get("coherence", 0.7)

        if not model_dir.exists():
            ax.set_title(f"{model_info['name']}", fontsize=14, fontweight='bold')
            continue

        print(f"\n{model_info['name']}:")
        legend_items = []

        forward_only = model_info.get('forward_only', False)

        for vtype in VECTOR_TYPES:
            fwd_files, rev_files = get_all_results_files(model_dir, vtype)

            if not fwd_files:
                print(f"  {SHORT_NAMES[vtype]}: no forward files")
                continue

            # Load and merge all forward results
            fwd_baseline, fwd_results = load_and_merge_results(fwd_files)

            # Compute accuracy based on forward_only flag
            if forward_only or not rev_files:
                # Forward only (for Qwen 235B or when no reversed files)
                accuracy = compute_accuracy_forward_only(
                    fwd_baseline, fwd_results,
                    min_coherence=min_coherence
                )
            else:
                # Combined forward + reversed - merge all reversed files
                rev_baseline, rev_results = load_and_merge_results(rev_files)
                accuracy = compute_accuracy_combined(
                    fwd_baseline, fwd_results,
                    rev_baseline, rev_results,
                    min_coherence=min_coherence
                )

            if not accuracy:
                print(f"  {SHORT_NAMES[vtype]}: no accuracy")
                continue

            layers = sorted(accuracy.keys())
            accs = [accuracy[l] for l in layers]

            print(f"  {SHORT_NAMES[vtype]}: {[f'{a:.0f}' for a in accs]}")

            line, = ax.plot(layers, accs, 'o-', label=SHORT_NAMES[vtype],
                           color=COLORS[vtype], linewidth=2, markersize=6, alpha=0.9)
            legend_items.append((line, SHORT_NAMES[vtype]))

        # Chance line
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance')

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Best Directional Accuracy (%)", fontsize=12)
        title = model_info['name']
        if model_info.get('forward_only', False):
            title += " (forward only)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(40, 95)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Legend inside plot, horizontal layout in upper right
        if legend_items:
            ax.legend(loc='upper right', fontsize=9, ncol=3, framealpha=0.9)

    plt.suptitle("Best Directional Accuracy by Layer (coherence > 0.7)",
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = RESULTS_DIR / "plots" / "all_models_directional_accuracy.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
