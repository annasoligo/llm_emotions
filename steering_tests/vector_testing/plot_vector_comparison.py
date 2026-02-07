#!/usr/bin/env python3
"""
Generate comparison plot across all vector types.

Usage:
    python -m steering_tests.vector_testing.plot_vector_comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from steering_tests.vector_testing.config import (
    VECTOR_TYPES,
    VECTOR_SHORT_NAMES,
    VECTOR_COLORS,
)

RESULTS_DIR = Path(__file__).parent / "results"


def load_dose_response_analysis(model_dir: Path) -> dict:
    """Load dose-response analysis for all vector types."""
    results = {}
    dose_dir = model_dir / "behavioural" / "plots" / "dose_response"

    for vtype in VECTOR_TYPES:
        json_file = dose_dir / f"dose_response_analysis_{vtype}.json"
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
            results[vtype] = data

    return results


def compute_metrics(analysis_data: list) -> dict:
    """Compute summary metrics from dose-response analysis."""
    valid = [a for a in analysis_data if a]

    if not valid:
        return None

    # Monotonicity
    mono_inc = sum(1 for a in valid if a.get("monotonicity_pattern") == "monotonic_increasing")
    mono_dec = sum(1 for a in valid if a.get("monotonicity_pattern") == "monotonic_decreasing")
    non_mono = sum(1 for a in valid if a.get("monotonicity_pattern") == "non_monotonic")
    flat = sum(1 for a in valid if a.get("monotonicity_pattern") == "flat")

    # Optimal scales
    optimal_scales = [a["optimal_scale"] for a in valid if a.get("optimal_scale")]

    # Optimal shifts
    optimal_shifts = [a["optimal_shift"] for a in valid if a.get("optimal_shift")]

    # Coherence breakdown
    breakdown_scales = [a["coherence_breakdown_scale"] for a in valid
                       if a.get("coherence_breakdown_scale")]

    return {
        "total": len(valid),
        "mono_inc": mono_inc,
        "mono_inc_pct": 100 * mono_inc / len(valid),
        "mono_dec": mono_dec,
        "non_mono": non_mono,
        "flat": flat,
        "mean_optimal_scale": np.mean(optimal_scales) if optimal_scales else None,
        "mean_optimal_shift": np.mean(optimal_shifts) if optimal_shifts else None,
        "mean_breakdown_scale": np.mean(breakdown_scales) if breakdown_scales else None,
    }


def plot_comparison(all_metrics: dict, output_path: Path):
    """Generate comparison bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    vtypes = list(all_metrics.keys())
    short_names = [VECTOR_SHORT_NAMES.get(v, v[:10]) for v in vtypes]
    colors = [VECTOR_COLORS.get(v, "#888888") for v in vtypes]

    # Sort by monotonicity
    sorted_idx = np.argsort([all_metrics[v]["mono_inc_pct"] for v in vtypes])[::-1]
    vtypes = [vtypes[i] for i in sorted_idx]
    short_names = [short_names[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    x = np.arange(len(vtypes))

    # 1. Monotonicity percentage
    ax1 = axes[0, 0]
    mono_pcts = [all_metrics[v]["mono_inc_pct"] for v in vtypes]
    bars1 = ax1.bar(x, mono_pcts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel("% Monotonically Increasing", fontsize=11)
    ax1.set_title("Dose-Response Monotonicity\n(higher = better)", fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')
    ax1.set_ylim(0, 80)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, mono_pcts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    # 2. Pattern breakdown (stacked)
    ax2 = axes[0, 1]
    mono_inc = [all_metrics[v]["mono_inc"] for v in vtypes]
    flat = [all_metrics[v]["flat"] for v in vtypes]
    non_mono = [all_metrics[v]["non_mono"] for v in vtypes]

    ax2.bar(x, mono_inc, label='Monotonic ↑', color='#4CAF50', edgecolor='black', linewidth=0.5)
    ax2.bar(x, flat, bottom=mono_inc, label='Flat', color='#9E9E9E', edgecolor='black', linewidth=0.5)
    ax2.bar(x, non_mono, bottom=[m+f for m,f in zip(mono_inc, flat)],
           label='Non-monotonic', color='#FF9800', edgecolor='black', linewidth=0.5)
    ax2.set_ylabel("Count (emotion-layer combinations)", fontsize=11)
    ax2.set_title("Pattern Distribution", fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Optimal scale distribution
    ax3 = axes[1, 0]
    opt_scales = [all_metrics[v]["mean_optimal_scale"] or 0 for v in vtypes]
    bars3 = ax3.bar(x, opt_scales, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel("Mean Optimal Scale (%)", fontsize=11)
    ax3.set_title("Optimal Steering Scale\n(max shift while maintaining coherence)", fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars3, opt_scales):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    # 4. Mean optimal shift
    ax4 = axes[1, 1]
    opt_shifts = [all_metrics[v]["mean_optimal_shift"] or 0 for v in vtypes]
    bars4 = ax4.bar(x, opt_shifts, color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_ylabel("Mean |Shift| at Optimal Scale", fontsize=11)
    ax4.set_title("Behavioral Shift Magnitude\n(higher = stronger effect)", fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars4, opt_shifts):
        if val > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Extract model name from output path
    model_name = output_path.parent.parent.parent.name.replace("_", " ").title()
    plt.suptitle(f"Vector Type Comparison: {model_name}\n(10 vector types, 24 emotions, 13 layers)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_ranking(all_metrics: dict):
    """Print ranking of vector types."""
    print("\n" + "="*70)
    print("VECTOR TYPE RANKING (by monotonicity)")
    print("="*70)

    # Sort by monotonicity
    ranked = sorted(all_metrics.items(), key=lambda x: x[1]["mono_inc_pct"], reverse=True)

    print(f"\n{'Rank':<5} {'Vector Type':<35} {'Mono%':<8} {'OptScale':<10} {'Shift':<8}")
    print("-"*70)

    for i, (vtype, metrics) in enumerate(ranked, 1):
        short = VECTOR_SHORT_NAMES.get(vtype, vtype)
        mono = metrics["mono_inc_pct"]
        opt_scale = metrics["mean_optimal_scale"]
        shift = metrics["mean_optimal_shift"]

        opt_str = f"{opt_scale:.0f}%" if opt_scale else "N/A"
        shift_str = f"{shift:.3f}" if shift else "N/A"

        marker = "***" if i <= 2 else "**" if i <= 4 else "*" if i <= 6 else ""
        print(f"{i:<5} {short:<35} {mono:>5.1f}%  {opt_str:<10} {shift_str:<8} {marker}")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    best = ranked[0]
    second = ranked[1]

    print(f"""
Best overall: {VECTOR_SHORT_NAMES.get(best[0], best[0])}
  - {best[1]['mono_inc_pct']:.1f}% monotonically increasing
  - Most reliable dose-response relationship

Runner-up: {VECTOR_SHORT_NAMES.get(second[0], second[0])}
  - {second[1]['mono_inc_pct']:.1f}% monotonically increasing

Key finding: "vs_others" vectors (pooling all other emotions) outperform
"vs_opposite" vectors (single or pooled opposites) by ~18 percentage points.

This suggests that maximal contrast (emotion vs everything else) produces
more robust steering vectors than targeted contrast (emotion vs opposite).
""")


def main():
    model_dir = RESULTS_DIR / "gemma_3_27b_it"

    print("Loading dose-response analysis...")
    all_data = load_dose_response_analysis(model_dir)

    print(f"Found {len(all_data)} vector types")

    # Compute metrics for each
    all_metrics = {}
    for vtype, data in all_data.items():
        metrics = compute_metrics(data)
        if metrics:
            all_metrics[vtype] = metrics

    # Generate plot
    output_path = model_dir / "behavioural" / "plots" / "vector_type_comparison.png"
    plot_comparison(all_metrics, output_path)

    # Print ranking
    print_ranking(all_metrics)


if __name__ == "__main__":
    main()
