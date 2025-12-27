#!/usr/bin/env python3
"""Visualize results from probe conversation evaluation."""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_accuracy_by_layer(results: Dict, output_path: Path):
    """Plot accuracy by layer for user and assistant."""
    layers = sorted([int(k) for k in results['layers'].keys()])
    user_accs = [results['layers'][str(l)]['user_accuracy'] for l in layers]
    asst_accs = [results['layers'][str(l)]['asst_accuracy'] for l in layers]
    overall_accs = [results['layers'][str(l)]['overall_accuracy'] for l in layers]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, user_accs, marker='o', label='User Emotion', linewidth=2)
    plt.plot(layers, asst_accs, marker='s', label='Assistant Emotion', linewidth=2)
    plt.plot(layers, overall_accs, marker='^', label='Overall', linewidth=2, linestyle='--')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Emotion Probe Accuracy on Conversations by Layer', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to {output_path}")


def plot_confusion_matrix(results: Dict, layer: int, role: str, output_path: Path):
    """Plot confusion matrix for a specific layer and role.

    Args:
        results: Results dict
        layer: Layer index
        role: 'user' or 'asst'
        output_path: Output path for plot
    """
    layer_key = str(layer)
    if layer_key not in results['layers']:
        print(f"Warning: Layer {layer} not found in results")
        return

    predictions = results['layers'][layer_key][f'{role}_predictions']

    # Extract true and predicted labels
    true_labels = [p['true'] for p in predictions]
    pred_labels = [p['pred'] for p in predictions]

    # Get unique labels
    emotions = sorted(set(true_labels + pred_labels))

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=emotions)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=emotions,
        yticklabels=emotions,
        cbar_kws={'label': 'Proportion'}
    )
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.title(f'Confusion Matrix - Layer {layer} - {role.capitalize()}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_per_emotion_accuracy(results: Dict, layer: int, output_path: Path):
    """Plot per-emotion accuracy for a specific layer."""
    layer_key = str(layer)
    if layer_key not in results['layers']:
        print(f"Warning: Layer {layer} not found in results")
        return

    # Get all predictions
    user_preds = results['layers'][layer_key]['user_predictions']
    asst_preds = results['layers'][layer_key]['asst_predictions']

    # Combine both
    all_preds = user_preds + asst_preds

    # Group by emotion
    emotion_correct = {}
    emotion_total = {}

    for pred in all_preds:
        emotion = pred['true']
        if emotion not in emotion_correct:
            emotion_correct[emotion] = 0
            emotion_total[emotion] = 0

        if pred['true'] == pred['pred']:
            emotion_correct[emotion] += 1
        emotion_total[emotion] += 1

    # Calculate accuracies
    emotions = sorted(emotion_correct.keys())
    accuracies = [emotion_correct[e] / emotion_total[e] for e in emotions]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(emotions, accuracies, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{acc:.2%}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Per-Emotion Accuracy - Layer {layer}', fontsize=14)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-emotion accuracy plot to {output_path}")


def print_summary(results: Dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total conversations: {results['num_conversations']}")

    print("\nAccuracy by Layer:")
    print(f"{'Layer':<8} {'User':<10} {'Assistant':<10} {'Overall':<10}")
    print("-" * 40)

    for layer in sorted([int(k) for k in results['layers'].keys()]):
        layer_results = results['layers'][str(layer)]
        print(
            f"{layer:<8} "
            f"{layer_results['user_accuracy']:<10.2%} "
            f"{layer_results['asst_accuracy']:<10.2%} "
            f"{layer_results['overall_accuracy']:<10.2%}"
        )

    # Find best layers
    best_user_layer = max(
        results['layers'].items(),
        key=lambda x: x[1]['user_accuracy']
    )
    best_asst_layer = max(
        results['layers'].items(),
        key=lambda x: x[1]['asst_accuracy']
    )
    best_overall_layer = max(
        results['layers'].items(),
        key=lambda x: x[1]['overall_accuracy']
    )

    print(f"\nBest Layers:")
    print(f"  User emotion:      Layer {best_user_layer[0]} ({best_user_layer[1]['user_accuracy']:.2%})")
    print(f"  Assistant emotion: Layer {best_asst_layer[0]} ({best_asst_layer[1]['asst_accuracy']:.2%})")
    print(f"  Overall:           Layer {best_overall_layer[0]} ({best_overall_layer[1]['overall_accuracy']:.2%})")


def main():
    parser = argparse.ArgumentParser(description="Visualize conversation evaluation results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to evaluation results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for plots (default: same as results file)",
    )
    parser.add_argument(
        "--best-layer",
        type=int,
        help="Layer to create detailed plots for (default: best overall layer)",
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}")
    with open(args.results, 'r') as f:
        results = json.load(f)

    # Print summary
    print_summary(results)

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = args.results.parent / f"{args.results.stem}_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {output_dir}")

    # Plot accuracy by layer
    plot_accuracy_by_layer(results, output_dir / "accuracy_by_layer.png")

    # Find best layer if not specified
    if args.best_layer:
        best_layer = args.best_layer
    else:
        best_layer = int(max(
            results['layers'].items(),
            key=lambda x: x[1]['overall_accuracy']
        )[0])

    print(f"\nCreating detailed plots for layer {best_layer}...")

    # Plot confusion matrices
    plot_confusion_matrix(results, best_layer, 'user', output_dir / f"confusion_matrix_layer{best_layer}_user.png")
    plot_confusion_matrix(results, best_layer, 'asst', output_dir / f"confusion_matrix_layer{best_layer}_asst.png")

    # Plot per-emotion accuracy
    plot_per_emotion_accuracy(results, best_layer, output_dir / f"per_emotion_accuracy_layer{best_layer}.png")

    print(f"\nDone! All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
