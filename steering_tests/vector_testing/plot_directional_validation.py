#!/usr/bin/env python3
"""
Visualize directional validation results for emotion steering.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

# Theoretical predictions
PREDICTIONS = {
    "fear": -1, "anxiety": -1, "despair": -1, "disgust": -1,
    "guilt": -1, "shame": -1, "calm": -1,
    "anger": +1, "joy": +1, "excitement": +1, "hope": +1,
    "curiosity": +1, "interest": +1, "pride": +1, "frustration": +1,
    "sadness": 0, "contempt": 0, "contentment": 0, "relief": 0,
    "surprise": 0, "gratitude": 0, "admiration": 0, "boredom": 0, "confusion": 0,
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

# Key emotions to display (with clear predictions)
KEY_EMOTIONS = [
    # Predicted negative
    "fear", "anxiety", "disgust", "guilt", "shame",
    # Predicted positive
    "anger", "joy", "excitement", "hope", "curiosity",
    # Unclear
    "sadness", "surprise", "calm",
]


def load_validation_results(results_file: Path) -> dict:
    """Load directional validation results."""
    with open(results_file) as f:
        return json.load(f)


def get_best_shift(analysis: list, emotion: str, min_coherence: float = 0.6) -> tuple:
    """Get best signed shift for an emotion (highest magnitude with good coherence)."""
    matches = [a for a in analysis
               if a["emotion"] == emotion
               and a["mean_coherence"] > min_coherence
               and a["scale_pct"] > 0]

    if not matches:
        return None, None, None

    best = max(matches, key=lambda x: abs(x["mean_signed_shift"]))
    return best["mean_signed_shift"], best["layer"], best["scale_pct"]


def plot_heatmap(all_results: dict, output_dir: Path):
    """Create heatmap of signed shifts: emotions × vector types."""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Build data matrix
    emotions = KEY_EMOTIONS
    vtypes = VECTOR_TYPES

    data = np.zeros((len(emotions), len(vtypes)))
    annotations = []

    for i, emotion in enumerate(emotions):
        row_annot = []
        for j, vtype in enumerate(vtypes):
            if vtype in all_results:
                shift, layer, scale = get_best_shift(all_results[vtype], emotion)
                if shift is not None:
                    data[i, j] = shift
                    row_annot.append(f"{shift:+.2f}")
                else:
                    data[i, j] = np.nan
                    row_annot.append("")
            else:
                data[i, j] = np.nan
                row_annot.append("")
        annotations.append(row_annot)

    # Create heatmap with diverging colormap
    cmap = plt.cm.RdBu_r  # Red = positive (risk-taking), Blue = negative (risk-averse)
    vmax = np.nanmax(np.abs(data))

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Signed Shift (+ = more risk-taking)", fontsize=11)

    # Set ticks
    ax.set_xticks(range(len(vtypes)))
    ax.set_xticklabels([SHORT_NAMES[v] for v in vtypes], fontsize=11)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=11)

    # Add text annotations
    for i in range(len(emotions)):
        for j in range(len(vtypes)):
            if annotations[i][j]:
                color = 'white' if abs(data[i, j]) > vmax * 0.5 else 'black'
                ax.text(j, i, annotations[i][j], ha='center', va='center',
                       fontsize=9, color=color, fontweight='bold')

    # Add prediction indicators on the left
    for i, emotion in enumerate(emotions):
        pred = PREDICTIONS.get(emotion, 0)
        if pred < 0:
            marker = "◀"  # Expected negative
            color = "blue"
        elif pred > 0:
            marker = "▶"  # Expected positive
            color = "red"
        else:
            marker = "●"  # No prediction
            color = "gray"
        ax.text(-0.7, i, marker, ha='center', va='center', fontsize=14, color=color)

    # Add legend for predictions
    ax.text(-0.7, -1.5, "Pred:", ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_title("Directional Shifts by Emotion × Vector Type\n(◀ = predicted negative, ▶ = predicted positive)",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Vector Extraction Method", fontsize=12)
    ax.set_ylabel("Emotion", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "directional_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_heatmap.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_heatmap.png'}")
    plt.close()


def plot_prediction_accuracy(all_results: dict, output_dir: Path):
    """Plot prediction accuracy by vector type and emotion category."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Match rate by vector type
    ax1 = axes[0]
    vtypes = VECTOR_TYPES
    match_rates = []

    for vtype in vtypes:
        if vtype not in all_results:
            match_rates.append(0)
            continue

        analysis = all_results[vtype]
        matches = 0
        total = 0

        for emotion, pred in PREDICTIONS.items():
            if pred == 0:
                continue
            shift, _, _ = get_best_shift(analysis, emotion)
            if shift is not None:
                if (pred > 0 and shift > 0) or (pred < 0 and shift < 0):
                    matches += 1
                total += 1

        match_rates.append(100 * matches / total if total > 0 else 0)

    colors = ['green' if r > 50 else 'red' for r in match_rates]
    bars = ax1.bar([SHORT_NAMES[v] for v in vtypes], match_rates, color=colors, alpha=0.7)
    ax1.axhline(y=50, color='black', linestyle='--', linewidth=1, label='Chance')
    ax1.set_ylabel("% Matching Prediction", fontsize=11)
    ax1.set_title("Prediction Accuracy by Vector Type", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend()

    # Add value labels
    for bar, rate in zip(bars, match_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=10)

    # 2. Negative emotions: predicted vs actual
    ax2 = axes[1]
    neg_emotions = ["fear", "anxiety", "disgust", "guilt", "shame"]

    x = np.arange(len(neg_emotions))
    width = 0.15

    for i, vtype in enumerate(vtypes):
        if vtype not in all_results:
            continue
        shifts = []
        for emotion in neg_emotions:
            shift, _, _ = get_best_shift(all_results[vtype], emotion)
            shifts.append(shift if shift is not None else 0)

        ax2.bar(x + i*width, shifts, width, label=SHORT_NAMES[vtype], alpha=0.8)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(neg_emotions, rotation=45, ha='right')
    ax2.set_ylabel("Signed Shift", fontsize=11)
    ax2.set_title("Negative Emotions (should be ↓)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')

    # Shade the "correct" region
    ax2.axhspan(ax2.get_ylim()[0], 0, alpha=0.1, color='green', label='Correct region')

    # 3. Positive emotions: predicted vs actual
    ax3 = axes[2]
    pos_emotions = ["anger", "joy", "excitement", "hope", "curiosity"]

    x = np.arange(len(pos_emotions))

    for i, vtype in enumerate(vtypes):
        if vtype not in all_results:
            continue
        shifts = []
        for emotion in pos_emotions:
            shift, _, _ = get_best_shift(all_results[vtype], emotion)
            shifts.append(shift if shift is not None else 0)

        ax3.bar(x + i*width, shifts, width, label=SHORT_NAMES[vtype], alpha=0.8)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(pos_emotions, rotation=45, ha='right')
    ax3.set_ylabel("Signed Shift", fontsize=11)
    ax3.set_title("Positive Emotions (should be ↑)", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='lower right')

    # Shade the "correct" region
    ax3.axhspan(0, ax3.get_ylim()[1], alpha=0.1, color='green', label='Correct region')

    plt.tight_layout()
    plt.savefig(output_dir / "directional_accuracy.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_accuracy.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_accuracy.png'}")
    plt.close()


def plot_scatter_predicted_vs_actual(all_results: dict, output_dir: Path):
    """Scatter plot: predicted direction vs actual shift."""

    fig, axes = plt.subplots(1, len(VECTOR_TYPES), figsize=(18, 4), sharey=True)

    for idx, vtype in enumerate(VECTOR_TYPES):
        ax = axes[idx]

        if vtype not in all_results:
            ax.set_title(SHORT_NAMES[vtype])
            continue

        analysis = all_results[vtype]

        # Collect data points
        preds = []
        shifts = []
        colors = []
        labels = []

        for emotion, pred in PREDICTIONS.items():
            shift, _, _ = get_best_shift(analysis, emotion)
            if shift is None:
                continue

            preds.append(pred)
            shifts.append(shift)
            labels.append(emotion)

            # Color by match
            if pred == 0:
                colors.append('gray')
            elif (pred > 0 and shift > 0) or (pred < 0 and shift < 0):
                colors.append('green')
            else:
                colors.append('red')

        # Add jitter to predictions for visibility
        preds_jittered = [p + np.random.uniform(-0.15, 0.15) for p in preds]

        ax.scatter(preds_jittered, shifts, c=colors, s=80, alpha=0.7, edgecolors='black')

        # Add quadrant shading
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Correct quadrants (green)
        ax.fill_between([-1.5, 0], [0, 0], [-2, -2], alpha=0.1, color='green')  # pred neg, shift neg
        ax.fill_between([0, 1.5], [0, 0], [2, 2], alpha=0.1, color='green')  # pred pos, shift pos

        # Wrong quadrants (red)
        ax.fill_between([-1.5, 0], [0, 0], [2, 2], alpha=0.1, color='red')  # pred neg, shift pos
        ax.fill_between([0, 1.5], [0, 0], [-2, -2], alpha=0.1, color='red')  # pred pos, shift neg

        # Labels for key emotions
        for i, (p, s, l) in enumerate(zip(preds_jittered, shifts, labels)):
            if l in ["fear", "anger", "joy", "disgust"]:
                ax.annotate(l, (p, s), xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax.set_xlim(-1.5, 1.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels(['Neg', '?', 'Pos'])
        ax.set_xlabel("Predicted Direction")
        ax.set_title(SHORT_NAMES[vtype], fontweight='bold')

        if idx == 0:
            ax.set_ylabel("Actual Signed Shift")

        # Count matches
        matches = sum(1 for c in colors if c == 'green')
        total = sum(1 for c in colors if c != 'gray')
        ax.text(0.95, 0.05, f'{matches}/{total}', transform=ax.transAxes,
               ha='right', fontsize=10, fontweight='bold')

    plt.suptitle("Predicted vs Actual Direction by Vector Type\n(Green = correct quadrant, Red = wrong quadrant)",
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "directional_scatter.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_scatter.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_scatter.png'}")
    plt.close()


def plot_accuracy_by_layer(all_results: dict, output_dir: Path):
    """Plot prediction accuracy by layer for each vector type."""

    fig, ax = plt.subplots(figsize=(12, 6))

    layers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    scale = 5.0  # Focus on 5% scale

    colors = plt.cm.tab10(np.linspace(0, 1, len(VECTOR_TYPES)))

    for idx, vtype in enumerate(VECTOR_TYPES):
        if vtype not in all_results:
            continue

        analysis = all_results[vtype]

        layer_accuracies = []

        for layer in layers:
            matches = 0
            total = 0

            for emotion, pred in PREDICTIONS.items():
                if pred == 0:
                    continue

                # Find this specific layer/scale combination
                layer_matches = [a for a in analysis
                                if a["emotion"] == emotion
                                and a["layer"] == layer
                                and a["scale_pct"] == scale
                                and a["mean_coherence"] > 0.5]

                if layer_matches:
                    shift = layer_matches[0]["mean_signed_shift"]
                    if (pred > 0 and shift > 0) or (pred < 0 and shift < 0):
                        matches += 1
                    total += 1

            accuracy = 100 * matches / total if total > 0 else 0
            layer_accuracies.append(accuracy)

        ax.plot(layers, layer_accuracies, 'o-', label=SHORT_NAMES[vtype],
                color=colors[idx], linewidth=2, markersize=6)

    ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Chance (50%)')

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("% Predictions Correct", fontsize=12)
    ax.set_title("Directional Prediction Accuracy by Layer (@ 5% scale)", fontsize=14, fontweight='bold')
    ax.set_xticks(layers)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "directional_accuracy_by_layer.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_accuracy_by_layer.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_accuracy_by_layer.png'}")
    plt.close()


def plot_accuracy_by_layer_split(all_results: dict, output_dir: Path):
    """Plot prediction accuracy by layer, split by positive vs negative emotions."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    scale = 5.0

    colors = plt.cm.tab10(np.linspace(0, 1, len(VECTOR_TYPES)))

    # Split emotions by prediction
    neg_emotions = [e for e, p in PREDICTIONS.items() if p < 0]
    pos_emotions = [e for e, p in PREDICTIONS.items() if p > 0]

    for ax_idx, (emotions, title) in enumerate([
        (neg_emotions, "Negative Emotions (pred: ↓)"),
        (pos_emotions, "Positive Emotions (pred: ↑)")
    ]):
        ax = axes[ax_idx]

        for idx, vtype in enumerate(VECTOR_TYPES):
            if vtype not in all_results:
                continue

            analysis = all_results[vtype]
            layer_accuracies = []

            for layer in layers:
                matches = 0
                total = 0

                for emotion in emotions:
                    pred = PREDICTIONS[emotion]

                    layer_matches = [a for a in analysis
                                    if a["emotion"] == emotion
                                    and a["layer"] == layer
                                    and a["scale_pct"] == scale
                                    and a["mean_coherence"] > 0.5]

                    if layer_matches:
                        shift = layer_matches[0]["mean_signed_shift"]
                        if (pred > 0 and shift > 0) or (pred < 0 and shift < 0):
                            matches += 1
                        total += 1

                accuracy = 100 * matches / total if total > 0 else 0
                layer_accuracies.append(accuracy)

            ax.plot(layers, layer_accuracies, 'o-', label=SHORT_NAMES[vtype],
                    color=colors[idx], linewidth=2, markersize=5)

        ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("% Correct", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_ylim(0, 100)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Directional Accuracy by Layer (@ 5% scale)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "directional_accuracy_by_layer_split.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_accuracy_by_layer_split.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_accuracy_by_layer_split.png'}")
    plt.close()


def plot_consistency_by_scale(all_results: dict, output_dir: Path):
    """
    Plot directional consistency: does increasing scale push consistently in same direction?

    For each emotion at a fixed layer, plot shift vs scale.
    Good vectors should show monotonic increase/decrease, not flip-flopping.
    """

    # Focus on key emotions and best layer (35)
    key_emotions = ["fear", "anxiety", "anger", "joy", "disgust", "excitement"]
    layer = 35  # Middle layer where steering is effective
    scales = [1.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    min_coherence = 0.6

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(VECTOR_TYPES)))

    for emo_idx, emotion in enumerate(key_emotions):
        ax = axes[emo_idx]
        pred = PREDICTIONS.get(emotion, 0)
        pred_str = "↓" if pred < 0 else "↑" if pred > 0 else "?"

        for vtype_idx, vtype in enumerate(VECTOR_TYPES):
            if vtype not in all_results:
                continue

            analysis = all_results[vtype]

            shifts = []
            coherences = []
            valid_scales = []

            for scale in scales:
                matches = [a for a in analysis
                          if a["emotion"] == emotion
                          and a["layer"] == layer
                          and a["scale_pct"] == scale]

                if matches:
                    shift = matches[0]["mean_signed_shift"]
                    coh = matches[0]["mean_coherence"]
                    shifts.append(shift)
                    coherences.append(coh)
                    valid_scales.append(scale)

            if shifts:
                # Plot with transparency based on coherence
                ax.plot(valid_scales, shifts, 'o-', label=SHORT_NAMES[vtype],
                       color=colors[vtype_idx], linewidth=2, markersize=6, alpha=0.8)

                # Mark low coherence points
                for i, (s, sh, c) in enumerate(zip(valid_scales, shifts, coherences)):
                    if c < min_coherence:
                        ax.scatter([s], [sh], s=100, facecolors='none',
                                  edgecolors=colors[vtype_idx], linewidths=2)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Scale (%)", fontsize=10)
        ax.set_ylabel("Signed Shift", fontsize=10)
        ax.set_title(f"{emotion.capitalize()} (pred: {pred_str})", fontsize=11, fontweight='bold')

        # Shade correct direction
        if pred < 0:
            ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -2, 0,
                      alpha=0.1, color='green')
        elif pred > 0:
            ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2,
                      alpha=0.1, color='green')

        ax.grid(True, alpha=0.3)
        if emo_idx == 0:
            ax.legend(fontsize=8, loc='best')

    plt.suptitle(f"Directional Consistency: Shift vs Scale (Layer {layer})\n(Open circles = coherence < {min_coherence})",
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "directional_consistency_by_scale.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_consistency_by_scale.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_consistency_by_scale.png'}")
    plt.close()


def compute_consistency_metric(all_results: dict) -> dict:
    """
    Compute consistency metric: what % of scale increases maintain the same direction?

    For coherent responses only (coh > 0.6), check if shift direction is consistent
    across scales 1%, 3%, 5%, 10%.
    """

    layer = 35
    scales = [1.0, 3.0, 5.0, 10.0]  # Exclude high scales where coherence drops
    min_coherence = 0.6

    results = {}

    for vtype in VECTOR_TYPES:
        if vtype not in all_results:
            continue

        analysis = all_results[vtype]

        consistent_count = 0
        total_count = 0

        emotion_details = {}

        for emotion in PREDICTIONS.keys():
            shifts = []

            for scale in scales:
                matches = [a for a in analysis
                          if a["emotion"] == emotion
                          and a["layer"] == layer
                          and a["scale_pct"] == scale
                          and a["mean_coherence"] >= min_coherence]

                if matches:
                    shifts.append(matches[0]["mean_signed_shift"])

            if len(shifts) >= 3:  # Need at least 3 points to assess consistency
                # Check if all shifts are same sign (or all near zero)
                signs = [1 if s > 0.05 else (-1 if s < -0.05 else 0) for s in shifts]
                non_zero_signs = [s for s in signs if s != 0]

                if len(non_zero_signs) >= 2:
                    # Consistent if all non-zero signs are the same
                    is_consistent = len(set(non_zero_signs)) == 1

                    # Also check monotonicity (does magnitude increase with scale?)
                    abs_shifts = [abs(s) for s in shifts]
                    is_monotonic = all(abs_shifts[i] <= abs_shifts[i+1] * 1.5  # Allow some noise
                                      for i in range(len(abs_shifts)-1))

                    if is_consistent:
                        consistent_count += 1
                    total_count += 1

                    emotion_details[emotion] = {
                        "shifts": shifts,
                        "consistent": is_consistent,
                        "monotonic": is_monotonic,
                        "direction": "+" if sum(non_zero_signs) > 0 else "-"
                    }

        results[vtype] = {
            "consistent_pct": 100 * consistent_count / total_count if total_count > 0 else 0,
            "n_emotions": total_count,
            "details": emotion_details
        }

    return results


def plot_consistency_summary(all_results: dict, output_dir: Path):
    """Plot summary of directional consistency across vector types."""

    consistency = compute_consistency_metric(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of consistency %
    ax1 = axes[0]
    vtypes = [v for v in VECTOR_TYPES if v in consistency]
    pcts = [consistency[v]["consistent_pct"] for v in vtypes]

    colors = ['green' if p > 70 else 'orange' if p > 50 else 'red' for p in pcts]
    bars = ax1.bar([SHORT_NAMES[v] for v in vtypes], pcts, color=colors, alpha=0.7)

    ax1.axhline(y=50, color='black', linestyle='--', linewidth=1, label='Chance')
    ax1.set_ylabel("% Emotions with Consistent Direction", fontsize=11)
    ax1.set_title("Directional Consistency Across Scales\n(same sign at 1%, 3%, 5%, 10%)",
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)

    for bar, pct in zip(bars, pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{pct:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # Right: Detailed breakdown by emotion category
    ax2 = axes[1]

    neg_emotions = [e for e, p in PREDICTIONS.items() if p < 0]
    pos_emotions = [e for e, p in PREDICTIONS.items() if p > 0]

    x = np.arange(len(vtypes))
    width = 0.35

    neg_pcts = []
    pos_pcts = []

    for vtype in vtypes:
        details = consistency[vtype]["details"]

        neg_consistent = sum(1 for e in neg_emotions
                            if e in details and details[e]["consistent"])
        neg_total = sum(1 for e in neg_emotions if e in details)

        pos_consistent = sum(1 for e in pos_emotions
                            if e in details and details[e]["consistent"])
        pos_total = sum(1 for e in pos_emotions if e in details)

        neg_pcts.append(100 * neg_consistent / neg_total if neg_total > 0 else 0)
        pos_pcts.append(100 * pos_consistent / pos_total if pos_total > 0 else 0)

    ax2.bar(x - width/2, neg_pcts, width, label='Negative emotions', color='blue', alpha=0.7)
    ax2.bar(x + width/2, pos_pcts, width, label='Positive emotions', color='red', alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels([SHORT_NAMES[v] for v in vtypes])
    ax2.set_ylabel("% Consistent", fontsize=11)
    ax2.set_title("Consistency by Emotion Category", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "directional_consistency_summary.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "directional_consistency_summary.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'directional_consistency_summary.png'}")
    plt.close()

    # Print detailed results
    print("\n" + "=" * 70)
    print("DIRECTIONAL CONSISTENCY: Does increasing scale maintain direction?")
    print("=" * 70)

    for vtype in vtypes:
        print(f"\n{SHORT_NAMES[vtype]}: {consistency[vtype]['consistent_pct']:.0f}% consistent")
        details = consistency[vtype]["details"]

        # Show inconsistent emotions
        inconsistent = [e for e, d in details.items() if not d["consistent"]]
        if inconsistent:
            print(f"  Inconsistent: {', '.join(inconsistent[:5])}")


def main():
    gemma_dir = RESULTS_DIR / "gemma_3_27b_it" / "behavioural"
    output_dir = gemma_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    # Load results
    results_file = gemma_dir / "directional_validation_all.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        print("Run directional_validation.py first")
        return

    all_results = load_validation_results(results_file)
    print(f"Loaded results for {len(all_results)} vector types")

    # Generate plots
    plot_heatmap(all_results, output_dir)
    plot_prediction_accuracy(all_results, output_dir)
    plot_scatter_predicted_vs_actual(all_results, output_dir)
    plot_accuracy_by_layer(all_results, output_dir)
    plot_accuracy_by_layer_split(all_results, output_dir)
    plot_consistency_by_scale(all_results, output_dir)
    plot_consistency_summary(all_results, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
