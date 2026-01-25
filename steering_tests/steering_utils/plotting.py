"""
Plotting utilities for steering experiments.

Simple visualization of KL divergence and other metrics.
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_kl_results(
    results_file: str,
    output_file: Optional[str] = None,
    figsize: tuple = (12, 8),
    show_random_band: bool = True,
):
    """
    Plot KL divergence results from a JSONL file.

    Args:
        results_file: Path to JSONL results file
        output_file: Path to save plot (None = show interactively)
        figsize: Figure size
        show_random_band: Show random vector mean ± std as shaded band
    """
    # Load results
    results = []
    meta = None

    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if "meta" in data:
                meta = data["meta"]
            else:
                results.append(data)

    if not results:
        raise ValueError(f"No results found in {results_file}")

    # Separate random and emotion vectors
    random_results = [r for r in results if r.get("is_random", False)]
    emotion_results = [r for r in results if not r.get("is_random", False)]

    # Get unique scales and vectors
    scales = sorted(set(r["scale"] for r in results))
    emotion_vectors = sorted(set(r["vector"] for r in emotion_results))

    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(emotion_vectors)))

    # Plot random vector band
    if show_random_band and random_results:
        random_by_scale = {}
        for r in random_results:
            s = r["scale"]
            if s not in random_by_scale:
                random_by_scale[s] = []
            random_by_scale[s].append(r["kl_mean"])

        rand_scales = sorted(random_by_scale.keys())
        rand_means = [np.mean(random_by_scale[s]) for s in rand_scales]
        rand_stds = [np.std(random_by_scale[s]) for s in rand_scales]

        ax.fill_between(
            rand_scales,
            np.array(rand_means) - np.array(rand_stds),
            np.array(rand_means) + np.array(rand_stds),
            alpha=0.2,
            color='gray',
            label='Random (±1σ)'
        )
        ax.plot(rand_scales, rand_means, '--', color='gray', alpha=0.5, linewidth=2)

    # Plot emotion vectors
    for i, vector in enumerate(emotion_vectors):
        vec_results = [r for r in emotion_results if r["vector"] == vector]
        vec_scales = [r["scale"] for r in vec_results]
        vec_kls = [r["kl_mean"] for r in vec_results]

        # Sort by scale
        sorted_pairs = sorted(zip(vec_scales, vec_kls))
        vec_scales, vec_kls = zip(*sorted_pairs)

        ax.plot(
            vec_scales,
            vec_kls,
            'o-',
            color=colors[i],
            label=vector,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Steering Scale", fontsize=14)
    ax.set_ylabel("KL Divergence", fontsize=14)
    ax.set_title("KL Divergence vs Steering Scale", fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    return fig, ax


def plot_entropy_change(
    results_file: str,
    output_file: Optional[str] = None,
    figsize: tuple = (12, 8),
):
    """
    Plot entropy change from baseline.

    Positive = more uncertainty (broader distribution)
    Negative = less uncertainty (more peaked distribution)
    """
    # Load results
    results = []
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            if "meta" not in data:
                results.append(data)

    if not results:
        raise ValueError(f"No results found in {results_file}")

    # Separate random and emotion vectors
    emotion_results = [r for r in results if not r.get("is_random", False)]
    scales = sorted(set(r["scale"] for r in results))
    emotion_vectors = sorted(set(r["vector"] for r in emotion_results))

    # Setup plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(emotion_vectors)))

    # Plot emotion vectors
    for i, vector in enumerate(emotion_vectors):
        vec_results = [r for r in emotion_results if r["vector"] == vector]
        vec_scales = [r["scale"] for r in vec_results]
        vec_entropy = [r.get("entropy_change", 0) for r in vec_results]

        sorted_pairs = sorted(zip(vec_scales, vec_entropy))
        vec_scales, vec_entropy = zip(*sorted_pairs)

        ax.plot(
            vec_scales,
            vec_entropy,
            'o-',
            color=colors[i],
            label=vector,
            linewidth=2,
            markersize=8,
        )

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Steering Scale", fontsize=14)
    ax.set_ylabel("Entropy Change from Baseline", fontsize=14)
    ax.set_title("Output Distribution Entropy Change", fontsize=16)
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot KL divergence results")
    parser.add_argument("results_file", help="Path to results JSONL file")
    parser.add_argument("--output", "-o", help="Output plot file")
    parser.add_argument("--entropy", action="store_true", help="Plot entropy change instead")

    args = parser.parse_args()

    if args.entropy:
        plot_entropy_change(args.results_file, args.output)
    else:
        plot_kl_results(args.results_file, args.output)
