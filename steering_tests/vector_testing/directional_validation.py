#!/usr/bin/env python3
"""
Directional Validation Experiment for Emotion Steering.

Tests whether emotion steering shifts behavior in theoretically predicted directions:
- Fear/Anxiety → decreased risk-taking (negative shift)
- Anger/Joy/Excitement → increased risk-taking (positive shift)

Key metrics:
1. Mean signed shift (is it in predicted direction?)
2. Consistency % (what fraction of items shift correctly?)
3. Statistical significance vs chance (50%)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

# Load test-specific predictions from extended_predictions.json
# Format: PREDICTIONS_BY_TEST[test_name][emotion] = +1/-1/None
PREDICTIONS_FILE = Path(__file__).parent / "results" / "extended_predictions.json"


def load_test_specific_predictions() -> dict:
    """Load test-specific predictions from extended_predictions.json."""
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            data = json.load(f)
            return data.get("predictions", {})
    return {}


PREDICTIONS_BY_TEST = load_test_specific_predictions()

# Fallback: Simple per-emotion predictions (DOSPERT-based approach/avoidance)
# Used when test-specific predictions are not available
# +1 = expect positive shift (more risk-taking/approach)
# -1 = expect negative shift (less risk-taking/avoidance)
# 0 = unclear/no prediction
PREDICTIONS_FALLBACK = {
    # Negative valence, avoidance
    "fear": -1,
    "anxiety": -1,
    "despair": -1,

    # Positive valence, approach
    "anger": +1,  # Approach motivation, perceived control (for DOSPERT)
    "joy": +1,
    "excitement": +1,
    "hope": +1,
    "optimism": +1,
    "pride": +1,

    # Disgust - avoidance
    "disgust": -1,
    "contempt": 0,  # Complex - could go either way

    # Sad/low arousal - unclear
    "sadness": 0,
    "guilt": -1,  # Self-punishment → caution
    "shame": -1,

    # Calm/content - slight negative (no need to change)
    "calm": -1,
    "contentment": 0,
    "relief": 0,

    # Curious/interested - exploration → positive
    "curiosity": +1,
    "interest": +1,

    # Other
    "surprise": 0,
    "gratitude": 0,
    "admiration": 0,
    "boredom": 0,
    "confusion": 0,
    "frustration": +1,  # Similar to anger
}


def get_prediction(test: str, emotion: str) -> int | None:
    """Get prediction for a specific test-emotion pair.

    Returns +1 (expect increase), -1 (expect decrease), or None (no prediction).
    """
    # Try test-specific prediction first
    if test in PREDICTIONS_BY_TEST:
        pred = PREDICTIONS_BY_TEST[test].get(emotion)
        if pred is not None:
            return pred

    # Fall back to DOSPERT-based predictions for dospert_* tests
    if test.startswith("dospert_"):
        return PREDICTIONS_FALLBACK.get(emotion, 0) or None

    # For other tests without specific predictions, return None
    return None


def load_behavioral_results(model_dir: Path) -> list[dict]:
    """Load all behavioral results for a model."""
    results = []
    for f in model_dir.glob("*.jsonl"):
        with open(f) as fp:
            for line in fp:
                data = json.loads(line)
                if data.get("type") == "steering":
                    data["source_file"] = f.name
                    results.append(data)
    return results


def compute_directional_metrics(
    results: list[dict],
    layer: int,
    scale: float,
    emotion: str,
) -> dict:
    """
    Compute directional validation metrics for a specific condition.

    Returns:
        Dict with mean_shift, consistency, p_value, n_items
    """
    # Filter to this condition
    matches = [r for r in results
               if r["layer"] == layer
               and r["scale_pct"] == scale
               and r["vector"] == emotion
               and not r.get("is_random", False)]

    if not matches:
        return None

    # Get the result (should be one per condition)
    r = matches[0]

    # We need per-item shifts, but our current data only has aggregates
    # For MVP, we'll use the aggregate metrics and note this limitation
    mean_shift = r.get("mean_score", 0) - r.get("baseline_mean_score", 0)

    # For now, return what we have
    return {
        "emotion": emotion,
        "layer": layer,
        "scale_pct": scale,
        "mean_abs_shift": r.get("mean_abs_shift", 0),
        "mean_coherence": r.get("mean_coherence", 0),
    }


def analyze_directional_shifts_from_raw(
    results_file: Path,
    baseline_scores: dict[str, float],
) -> dict:
    """
    Analyze directional shifts from raw results.

    Uses scores_by_test and baseline to compute TRUE signed shifts.
    (shifts_by_test contains absolute values, not signed!)
    """
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    # Get baseline scores by test
    baseline = None
    for r in results:
        if r.get("type") == "baseline":
            baseline = r
            break

    if not baseline:
        print("WARNING: No baseline found!")
        return []

    baseline_scores_by_test = baseline.get("scores_by_test", {})
    baseline_avg_score = baseline.get("avg_score", 0)

    analysis = []
    for r in results:
        if r.get("type") != "steering":
            continue
        if r.get("is_random", False):
            continue

        emotion = r["vector"]
        layer = r["layer"]
        scale = r["scale_pct"]

        if scale == 0:
            continue

        # Get steered scores by test
        steered_scores_by_test = r.get("scores_by_test", {})

        # Compute TRUE signed shifts: steered - baseline
        signed_shifts = {}
        for test in steered_scores_by_test:
            if test in baseline_scores_by_test:
                signed_shifts[test] = steered_scores_by_test[test] - baseline_scores_by_test[test]

        if not signed_shifts:
            # Fallback to using mean_score if scores_by_test not available
            mean_score = r.get("mean_score", 0)
            mean_signed_shift = mean_score - baseline_avg_score
            shifts = [mean_signed_shift]
            # Use fallback prediction for this emotion
            fallback_pred = PREDICTIONS_FALLBACK.get(emotion, 0)
            predictions_used = {emotion: fallback_pred}
        else:
            shifts = list(signed_shifts.values())
            mean_signed_shift = np.mean(shifts)
            predictions_used = {}

        # Count correct predictions using TEST-SPECIFIC predictions
        correct = 0
        total_with_prediction = 0
        prediction_details = {}

        if signed_shifts:
            for test, shift in signed_shifts.items():
                pred = get_prediction(test, emotion)
                prediction_details[test] = {"shift": shift, "prediction": pred}

                if pred is not None and shift != 0:
                    actual_dir = 1 if shift > 0 else -1
                    total_with_prediction += 1
                    if actual_dir == pred:
                        correct += 1
        else:
            # Fallback case - use fallback prediction
            fallback_pred = PREDICTIONS_FALLBACK.get(emotion, 0)
            if fallback_pred != 0 and mean_signed_shift != 0:
                actual_dir = 1 if mean_signed_shift > 0 else -1
                total_with_prediction = 1
                correct = 1 if actual_dir == fallback_pred else 0

        if total_with_prediction > 0:
            consistency = correct / total_with_prediction

            # Binomial test: is consistency significantly > 50%?
            try:
                result = stats.binomtest(correct, total_with_prediction, 0.5, alternative='greater')
                p_value = result.pvalue
            except AttributeError:
                p_value = stats.binom_test(correct, total_with_prediction, 0.5, alternative='greater')
        else:
            consistency = None
            p_value = None

        # For backward compatibility, include a single "prediction" value
        # (majority prediction or fallback)
        prediction = PREDICTIONS_FALLBACK.get(emotion, 0)

        analysis.append({
            "emotion": emotion,
            "layer": layer,
            "scale_pct": scale,
            "prediction": prediction,  # Fallback for backward compat
            "mean_signed_shift": mean_signed_shift,
            "mean_abs_shift": r.get("mean_abs_shift", 0),
            "mean_coherence": r.get("mean_coherence", 0),
            "consistency": consistency,
            "p_value": p_value,
            "n_tests": len(shifts),
            "n_with_prediction": total_with_prediction,
            "n_correct": correct,
            "shifts_by_test": dict(signed_shifts),  # Make a copy
            "predictions_by_test": prediction_details,  # New: test-specific predictions
        })

    return analysis


def plot_directional_validation(analysis: list[dict], output_dir: Path):
    """Plot directional validation results."""

    # Filter to key emotions with clear predictions
    key_emotions = ["fear", "anxiety", "anger", "joy", "excitement", "sadness"]

    # Group by emotion
    by_emotion = defaultdict(list)
    for a in analysis:
        if a["emotion"] in key_emotions:
            by_emotion[a["emotion"]].append(a)

    # Plot 1: Mean signed shift by emotion at optimal layer/scale
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Find best conditions (highest coherence > 0.6)
    best_by_emotion = {}
    for emotion, data in by_emotion.items():
        valid = [d for d in data if d["mean_coherence"] > 0.6]
        if valid:
            best = max(valid, key=lambda x: abs(x["mean_signed_shift"]))
            best_by_emotion[emotion] = best

    # Plot 1a: Mean signed shift
    ax1 = axes[0, 0]
    emotions = list(best_by_emotion.keys())
    shifts = [best_by_emotion[e]["mean_signed_shift"] for e in emotions]
    predictions = [PREDICTIONS_FALLBACK.get(e, 0) for e in emotions]

    colors = ['green' if (s > 0 and p > 0) or (s < 0 and p < 0) else 'red' if p != 0 else 'gray'
              for s, p in zip(shifts, predictions)]

    bars = ax1.bar(emotions, shifts, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel("Mean Signed Shift")
    ax1.set_title("Directional Shift by Emotion\n(Green=matches prediction, Red=opposite)")
    ax1.set_xticklabels(emotions, rotation=45, ha='right')

    # Add prediction arrows
    for i, (e, p) in enumerate(zip(emotions, predictions)):
        if p != 0:
            arrow = "↓" if p < 0 else "↑"
            ax1.annotate(f"pred: {arrow}", (i, 0), xytext=(0, -20 if shifts[i] > 0 else 20),
                        textcoords='offset points', ha='center', fontsize=8, color='blue')

    # Plot 1b: Consistency % for emotions with predictions
    ax2 = axes[0, 1]
    emotions_with_pred = [e for e in emotions if PREDICTIONS_FALLBACK.get(e, 0) != 0]
    consistencies = [best_by_emotion[e].get("consistency", 0) * 100 for e in emotions_with_pred]

    colors2 = ['green' if c > 50 else 'red' for c in consistencies]
    ax2.bar(emotions_with_pred, consistencies, color=colors2, alpha=0.7)
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=1, label="Chance (50%)")
    ax2.set_ylabel("% Items in Predicted Direction")
    ax2.set_title("Consistency of Directional Effect")
    ax2.set_ylim(0, 100)
    ax2.set_xticklabels(emotions_with_pred, rotation=45, ha='right')
    ax2.legend()

    # Plot 2: Shift distribution across layers for fear vs anger
    ax3 = axes[1, 0]

    layers = sorted(set(a["layer"] for a in analysis))
    scale = 5.0  # Focus on 5% scale

    for emotion, color, marker in [("fear", "blue", "o"), ("anger", "red", "s")]:
        if emotion in by_emotion:
            layer_shifts = []
            for layer in layers:
                matches = [a for a in by_emotion[emotion]
                          if a["layer"] == layer and a["scale_pct"] == scale]
                if matches:
                    layer_shifts.append(matches[0]["mean_signed_shift"])
                else:
                    layer_shifts.append(np.nan)
            ax3.plot(layers, layer_shifts, f'{marker}-', label=emotion, color=color, linewidth=2)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Mean Signed Shift")
    ax3.set_title(f"Directional Shift by Layer (@ {scale}% scale)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 3: Shift by test type for fear
    ax4 = axes[1, 1]

    if "fear" in best_by_emotion:
        fear_data = best_by_emotion["fear"]
        shifts_by_test = fear_data.get("shifts_by_test", {})

        if shifts_by_test:
            tests = list(shifts_by_test.keys())
            test_shifts = [shifts_by_test[t] for t in tests]

            colors3 = ['green' if s < 0 else 'red' for s in test_shifts]  # Fear should be negative
            ax4.barh(tests, test_shifts, color=colors3, alpha=0.7)
            ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax4.set_xlabel("Signed Shift")
            ax4.set_title(f"Fear Steering: Shift by Test Type\n(L{fear_data['layer']} @ {fear_data['scale_pct']}%)")

    plt.tight_layout()
    plt.savefig(output_dir / "directional_validation.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_validation.png'}")
    plt.close()


def print_validation_summary(analysis: list[dict]):
    """Print summary of directional validation."""
    print("\n" + "=" * 80)
    print("DIRECTIONAL VALIDATION SUMMARY (with test-specific predictions)")
    print("=" * 80)

    # Group by emotion, find best condition
    by_emotion = defaultdict(list)
    for a in analysis:
        by_emotion[a["emotion"]].append(a)

    print(f"\n{'Emotion':<12} {'Pred':>5} {'Shift':>8} {'Dir':>5} {'Correct':>10} {'p-value':>10} {'Layer':>6} {'Scale':>6}")
    print("-" * 85)

    for emotion in sorted(by_emotion.keys()):
        data = by_emotion[emotion]

        # Find best condition (highest abs shift with coherence > 0.6)
        valid = [d for d in data if d["mean_coherence"] > 0.6 and d["scale_pct"] > 0]
        if not valid:
            continue

        best = max(valid, key=lambda x: abs(x["mean_signed_shift"]))

        # Use fallback prediction for display (shows general expected direction)
        pred = PREDICTIONS_FALLBACK.get(emotion, 0)
        pred_str = "↓" if pred < 0 else "↑" if pred > 0 else "?"

        shift = best["mean_signed_shift"]
        direction = "↓" if shift < 0 else "↑"

        # Check if matches prediction (using fallback for display)
        match = ""
        if pred != 0:
            if (pred > 0 and shift > 0) or (pred < 0 and shift < 0):
                match = "✓"
            else:
                match = "✗"

        # Use test-specific accuracy from new fields
        n_correct = best.get("n_correct", 0)
        n_with_pred = best.get("n_with_prediction", 0)
        if n_with_pred > 0:
            consist_str = f"{n_correct}/{n_with_pred}"
        else:
            consist = best.get("consistency")
            consist_str = f"{consist*100:.0f}%" if consist is not None else "N/A"

        p_val = best.get("p_value")
        p_str = f"{p_val:.4f}" if p_val is not None else "N/A"

        print(f"{emotion:<12} {pred_str:>5} {shift:>+8.3f} {direction+match:>5} {consist_str:>10} {p_str:>10} {best['layer']:>6} {best['scale_pct']:>5}%")

    print("=" * 85)
    print("Pred: Fallback prediction (↑=positive, ↓=negative, ?=unclear)")
    print("Dir: Observed direction (✓=matches fallback, ✗=opposite)")
    print("Correct: n_correct/n_with_prediction using TEST-SPECIFIC predictions")


def analyze_all_vector_types(gemma_dir: Path):
    """Analyze directional shifts across all vector types."""

    vector_types = [
        "base_emotion_vs_others",
        "high_emotion_vs_others",
        "text_pairs_emotion_vs_neutral",
        "text_pairs_emotion_vs_opposite",
        "text_pairs_emotion_vs_others",
    ]

    all_results = {}

    for vtype in vector_types:
        # Find results file (non-reversed)
        pattern = f"{vtype}_layers*.jsonl"
        files = sorted(gemma_dir.glob(pattern))
        files = [f for f in files if "_reversed_" not in f.name]

        if not files:
            print(f"  {vtype}: No files found")
            continue

        latest = files[-1]
        analysis = analyze_directional_shifts_from_raw(latest, {})

        if analysis:
            all_results[vtype] = analysis

    return all_results


def summarize_across_vector_types(all_results: dict):
    """Print summary comparing directional effects across vector types."""

    print("\n" + "=" * 100)
    print("DIRECTIONAL VALIDATION: ALL VECTOR TYPES")
    print("=" * 100)

    # Key emotions to compare
    key_emotions = ["fear", "anxiety", "anger", "joy", "excitement", "sadness", "disgust"]

    # Header
    vector_types = list(all_results.keys())
    short_names = {
        "base_emotion_vs_others": "Base",
        "high_emotion_vs_others": "High",
        "text_pairs_emotion_vs_neutral": "Txt-Neut",
        "text_pairs_emotion_vs_opposite": "Txt-Opp",
        "text_pairs_emotion_vs_others": "Txt-Oth",
    }

    print(f"\n{'Emotion':<12} {'Pred':>5}", end="")
    for vt in vector_types:
        print(f" | {short_names.get(vt, vt[:8]):>10}", end="")
    print()
    print("-" * (18 + 13 * len(vector_types)))

    for emotion in key_emotions:
        pred = PREDICTIONS_FALLBACK.get(emotion, 0)
        pred_str = "↓" if pred < 0 else "↑" if pred > 0 else "?"

        print(f"{emotion:<12} {pred_str:>5}", end="")

        for vt in vector_types:
            analysis = all_results.get(vt, [])

            # Find best condition for this emotion (coherence > 0.6)
            matches = [a for a in analysis
                      if a["emotion"] == emotion
                      and a["mean_coherence"] > 0.6
                      and a["scale_pct"] > 0]

            if matches:
                best = max(matches, key=lambda x: abs(x["mean_signed_shift"]))
                shift = best["mean_signed_shift"]
                direction = "↑" if shift > 0 else "↓"

                # Check if matches prediction
                if pred != 0:
                    match = "✓" if (pred > 0 and shift > 0) or (pred < 0 and shift < 0) else "✗"
                else:
                    match = ""

                print(f" | {shift:>+6.2f} {direction}{match}", end="")
            else:
                print(f" | {'N/A':>10}", end="")

        print()

    print("=" * (18 + 13 * len(vector_types)))

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY: % of emotions matching theoretical predictions")
    print("=" * 60)

    for vt in vector_types:
        analysis = all_results.get(vt, [])

        # Count matches for emotions with predictions
        matches = 0
        total = 0

        for emotion in PREDICTIONS_FALLBACK:
            pred = PREDICTIONS_FALLBACK[emotion]
            if pred == 0:
                continue

            emo_data = [a for a in analysis
                       if a["emotion"] == emotion
                       and a["mean_coherence"] > 0.6
                       and a["scale_pct"] > 0]

            if emo_data:
                best = max(emo_data, key=lambda x: abs(x["mean_signed_shift"]))
                shift = best["mean_signed_shift"]

                if (pred > 0 and shift > 0) or (pred < 0 and shift < 0):
                    matches += 1
                total += 1

        pct = 100 * matches / total if total > 0 else 0
        print(f"  {short_names.get(vt, vt):<15}: {matches}/{total} = {pct:.0f}%")


def main():
    # Find Gemma results
    gemma_dir = RESULTS_DIR / "gemma_3_27b_it" / "behavioural"

    if not gemma_dir.exists():
        print(f"Results directory not found: {gemma_dir}")
        return

    print("Analyzing all vector types...")
    all_results = analyze_all_vector_types(gemma_dir)

    if not all_results:
        print("No results found")
        return

    # Print comparison across vector types
    summarize_across_vector_types(all_results)

    # Also run detailed analysis on base vectors
    print("\n\n" + "=" * 80)
    print("DETAILED ANALYSIS: base_emotion_vs_others")
    print("=" * 80)

    if "base_emotion_vs_others" in all_results:
        analysis = all_results["base_emotion_vs_others"]
        print_validation_summary(analysis)

        # Generate plots
        output_dir = gemma_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        plot_directional_validation(analysis, output_dir)

    # Save all results
    output_file = gemma_dir / "directional_validation_all.json"

    # Convert to serializable format
    save_data = {}
    for vt, analysis in all_results.items():
        save_data[vt] = analysis

    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nAll results saved to: {output_file}")


if __name__ == "__main__":
    main()
