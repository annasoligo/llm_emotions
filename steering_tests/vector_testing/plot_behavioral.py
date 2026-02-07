#!/usr/bin/env python3
"""
Plotting utilities for behavioral shift experiment results.

Usage:
    python -m steering_tests.vector_testing.plot_behavioral results.jsonl

Generates:
    - Heatmap: layer × emotion showing mean |shift|
    - Dose-response: |shift| vs scale at each layer
    - Coherence: P(valid letter) vs scale
    - Random vs emotion comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(filepath: str) -> tuple:
    """Load results from JSONL file."""
    meta = None
    baseline = None
    results = []

    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            if "meta" in data:
                meta = data["meta"]
            elif data.get("type") == "baseline":
                baseline = data
            elif data.get("type") == "steering":
                results.append(data)

    return meta, baseline, results


def plot_shift_vs_scale(
    results: List[Dict],
    output_path: str,
    title: Optional[str] = None,
):
    """
    Plot mean |shift| vs steering scale, comparing emotion vs random vectors.

    Shows dose-response curve with coherence on secondary axis.
    """
    df = pd.DataFrame(results)

    # Separate emotion and random
    emotion_df = df[~df["is_random"]]
    random_df = df[df["is_random"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: |shift| vs scale
    if not emotion_df.empty:
        # Aggregate by scale
        emotion_agg = emotion_df.groupby("scale_pct").agg({
            "mean_abs_shift": ["mean", "std"],
            "mean_coherence": "mean",
        }).reset_index()
        emotion_agg.columns = ["scale_pct", "shift_mean", "shift_std", "coherence"]

        ax1.errorbar(
            emotion_agg["scale_pct"],
            emotion_agg["shift_mean"],
            yerr=emotion_agg["shift_std"],
            label="Emotion vectors",
            marker="o",
            capsize=3,
            color="tab:blue",
        )

    if not random_df.empty:
        random_agg = random_df.groupby("scale_pct").agg({
            "mean_abs_shift": ["mean", "std"],
            "mean_coherence": "mean",
        }).reset_index()
        random_agg.columns = ["scale_pct", "shift_mean", "shift_std", "coherence"]

        ax1.errorbar(
            random_agg["scale_pct"],
            random_agg["shift_mean"],
            yerr=random_agg["shift_std"],
            label="Random vectors",
            marker="s",
            capsize=3,
            color="tab:orange",
            linestyle="--",
        )

    ax1.set_xlabel("Steering Scale (% of layer norm)")
    ax1.set_ylabel("Mean |shift| in expected score")
    ax1.set_title("Behavioral Shift vs Steering Magnitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Coherence vs scale
    if not emotion_df.empty:
        ax2.plot(
            emotion_agg["scale_pct"],
            emotion_agg["coherence"],
            label="Emotion vectors",
            marker="o",
            color="tab:blue",
        )

    if not random_df.empty:
        ax2.plot(
            random_agg["scale_pct"],
            random_agg["coherence"],
            label="Random vectors",
            marker="s",
            color="tab:orange",
            linestyle="--",
        )

    ax2.axhline(y=0.7, color="red", linestyle=":", label="70% threshold")
    ax2.set_xlabel("Steering Scale (% of layer norm)")
    ax2.set_ylabel("Mean P(A|B|C|D|E)")
    ax2.set_title("Response Coherence vs Steering Magnitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_comparison(
    results: List[Dict],
    output_path: str,
    scale_pct: float = 20.0,
    title: Optional[str] = None,
):
    """
    Compare |shift| across layers at a fixed scale.

    Shows which layers cause the most behavioral change.
    """
    df = pd.DataFrame(results)
    df = df[df["scale_pct"] == scale_pct]

    if df.empty:
        print(f"No results at scale {scale_pct}%")
        return

    # Separate emotion and random
    emotion_df = df[~df["is_random"]]
    random_df = df[df["is_random"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    layers = sorted(df["layer"].unique())

    # Emotion vectors
    if not emotion_df.empty:
        emotion_by_layer = emotion_df.groupby("layer")["mean_abs_shift"].agg(["mean", "std"])
        ax.bar(
            [l - 0.2 for l in layers],
            emotion_by_layer.loc[layers, "mean"],
            yerr=emotion_by_layer.loc[layers, "std"],
            width=0.4,
            label="Emotion vectors",
            color="tab:blue",
            alpha=0.8,
            capsize=3,
        )

    # Random vectors
    if not random_df.empty:
        random_by_layer = random_df.groupby("layer")["mean_abs_shift"].agg(["mean", "std"])
        ax.bar(
            [l + 0.2 for l in layers],
            random_by_layer.loc[layers, "mean"],
            yerr=random_by_layer.loc[layers, "std"],
            width=0.4,
            label="Random vectors",
            color="tab:orange",
            alpha=0.8,
            capsize=3,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |shift| in expected score")
    ax.set_title(f"Behavioral Shift by Layer (at {scale_pct}% of layer norm)")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_emotion_heatmap(
    results: List[Dict],
    output_path: str,
    scale_pct: float = 20.0,
    title: Optional[str] = None,
):
    """
    Heatmap of |shift| by emotion × layer.

    Shows which emotions cause the most behavioral change at each layer.
    """
    df = pd.DataFrame(results)
    df = df[(df["scale_pct"] == scale_pct) & (~df["is_random"])]

    if df.empty:
        print(f"No emotion results at scale {scale_pct}%")
        return

    # Pivot to emotion × layer
    pivot = df.pivot_table(
        values="mean_abs_shift",
        index="vector",
        columns="layer",
        aggfunc="mean",
    )

    # Sort emotions by mean shift
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.3)))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Mean |shift|"},
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Emotion")
    ax.set_title(f"Behavioral Shift by Emotion × Layer (at {scale_pct}% of layer norm)")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_test_breakdown(
    results: List[Dict],
    output_path: str,
    scale_pct: float = 20.0,
    title: Optional[str] = None,
):
    """
    Show |shift| broken down by behavioral test.

    Helps identify which psychological constructs are most affected by steering.
    """
    df = pd.DataFrame(results)
    df = df[(df["scale_pct"] == scale_pct) & (~df["is_random"])]

    if df.empty:
        print(f"No emotion results at scale {scale_pct}%")
        return

    # Extract shifts_by_test into rows
    rows = []
    for _, row in df.iterrows():
        for test, shift in row["shifts_by_test"].items():
            rows.append({
                "layer": row["layer"],
                "vector": row["vector"],
                "test": test,
                "shift": shift,
            })

    test_df = pd.DataFrame(rows)

    # Aggregate by test
    test_agg = test_df.groupby("test")["shift"].agg(["mean", "std"]).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        test_agg.index,
        test_agg["mean"],
        xerr=test_agg["std"],
        capsize=3,
        color="tab:blue",
        alpha=0.8,
    )

    ax.set_xlabel("Mean |shift| in expected score")
    ax.set_ylabel("Behavioral Test")
    ax.set_title(f"Behavioral Shift by Test Type (at {scale_pct}% of layer norm)")
    ax.grid(True, alpha=0.3, axis="x")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_plots(
    results_file: str,
    output_dir: Optional[str] = None,
    scale_pct: float = 20.0,
):
    """Generate all plots for a results file."""
    meta, baseline, results = load_results(results_file)

    if output_dir is None:
        output_dir = Path(results_file).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(results_file).stem
    model_name = meta.get("model", "unknown") if meta else "unknown"

    # Plot shift vs scale
    plot_shift_vs_scale(
        results,
        str(output_dir / f"{base_name}_dose_response.png"),
        title=f"Model: {model_name}",
    )

    # Plot layer comparison
    if len(set(r["layer"] for r in results)) > 1:
        plot_layer_comparison(
            results,
            str(output_dir / f"{base_name}_layer_comparison.png"),
            scale_pct=scale_pct,
            title=f"Model: {model_name}",
        )

    # Plot emotion heatmap
    if len(set(r["layer"] for r in results)) > 1 and any(not r["is_random"] for r in results):
        plot_emotion_heatmap(
            results,
            str(output_dir / f"{base_name}_emotion_heatmap.png"),
            scale_pct=scale_pct,
            title=f"Model: {model_name}",
        )

    # Plot test breakdown
    if any(not r["is_random"] for r in results):
        plot_test_breakdown(
            results,
            str(output_dir / f"{base_name}_test_breakdown.png"),
            scale_pct=scale_pct,
            title=f"Model: {model_name}",
        )

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from behavioral shift experiment results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSONL file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Output directory for plots (default: same as results file)"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=20.0,
        help="Scale percentage to use for comparison plots (default: 20)"
    )

    args = parser.parse_args()
    generate_all_plots(args.results_file, args.output_dir, args.scale)


if __name__ == "__main__":
    main()
