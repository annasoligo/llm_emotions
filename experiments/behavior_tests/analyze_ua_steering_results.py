#!/usr/bin/env python3
"""
Analyze UA Emotion Steering Experiment Results

Generates:
1. Summary tables per prompt type
2. Heatmaps showing emotion × direction effects
3. Coherency analysis
4. Statistical significance tests
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ============================================================================
# Data Loading
# ============================================================================

def load_results(results_file: Path) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    return results

# ============================================================================
# Metric Extraction
# ============================================================================

def get_metric(result: Dict, prompt_type: str) -> float:
    """Extract the main metric for a given prompt type."""
    judge = result.get("task_judge_result")
    if judge is None:
        return None

    if prompt_type == "sycophancy":
        return judge.get("side_score")
    elif prompt_type == "deception":
        return judge.get("score")
    elif prompt_type == "harmful":
        return judge.get("refusal_score")
    elif prompt_type == "sandbagging":
        return 1 if judge.get("answer_correct") else 0
    elif prompt_type == "reward_hacking":
        return judge.get("entry_count")
    return None

def get_coherency(result: Dict) -> float:
    """Extract coherency score."""
    coh = result.get("coherency_result")
    if coh:
        return coh.get("coherency_score")
    return None

# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_by_condition(results: List[Dict], prompt_type: str) -> Dict:
    """Compute statistics per condition."""
    by_condition = defaultdict(list)

    for r in results:
        if r["prompt_type"] != prompt_type:
            continue

        vector = r["vector"]
        direction = r["direction"]

        if vector == "baseline":
            cond = "baseline"
        else:
            sign = "+" if direction > 0 else "-"
            cond = f"{vector}_{sign}"

        metric = get_metric(r, prompt_type)
        if metric is not None:
            by_condition[cond].append(metric)

    stats_by_cond = {}
    baseline_vals = by_condition.get("baseline", [])
    baseline_mean = np.mean(baseline_vals) if baseline_vals else 0
    baseline_std = np.std(baseline_vals) if baseline_vals else 0

    for cond, vals in by_condition.items():
        if not vals:
            continue

        mean = np.mean(vals)
        std = np.std(vals)
        n = len(vals)

        # Effect size (Cohen's d)
        if cond != "baseline" and baseline_std > 0:
            cohens_d = (mean - baseline_mean) / baseline_std
        else:
            cohens_d = 0

        # Statistical test vs baseline
        if cond != "baseline" and baseline_vals:
            t_stat, p_value = stats.ttest_ind(vals, baseline_vals)
        else:
            t_stat, p_value = 0, 1

        stats_by_cond[cond] = {
            "mean": mean,
            "std": std,
            "n": n,
            "delta": mean - baseline_mean,
            "cohens_d": cohens_d,
            "p_value": p_value,
        }

    return stats_by_cond

def analyze_coherency(results: List[Dict]) -> Dict:
    """Analyze coherency scores across conditions."""
    by_condition = defaultdict(list)

    for r in results:
        vector = r["vector"]
        direction = r["direction"]

        if vector == "baseline":
            cond = "baseline"
        else:
            sign = "+" if direction > 0 else "-"
            cond = f"{vector}_{sign}"

        coh = get_coherency(r)
        if coh is not None:
            by_condition[cond].append(coh)

    stats_by_cond = {}
    for cond, vals in by_condition.items():
        if vals:
            stats_by_cond[cond] = {
                "mean": np.mean(vals),
                "std": np.std(vals),
                "min": np.min(vals),
                "n": len(vals),
            }

    return stats_by_cond

# ============================================================================
# Visualization
# ============================================================================

EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
DIRECTIONS = ["ua_model", "ua_user"]

def create_heatmap(stats: Dict, prompt_type: str, metric_name: str, output_dir: Path):
    """Create heatmap of emotion effects."""

    # Build matrix: emotions × (direction × polarity)
    # Columns: model+, model-, user+, user-
    col_labels = ["model+", "model-", "user+", "user-"]

    baseline = stats.get("baseline", {}).get("mean", 0)

    matrix = np.zeros((len(EMOTIONS), len(col_labels)))

    for i, emotion in enumerate(EMOTIONS):
        for j, (direction, polarity) in enumerate([
            ("ua_model", "+"), ("ua_model", "-"),
            ("ua_user", "+"), ("ua_user", "-")
        ]):
            cond = f"{emotion}_{direction}_{polarity}"
            if cond in stats:
                matrix[i, j] = stats[cond]["delta"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine color scale
    vmax = np.abs(matrix).max()
    vmin = -vmax

    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS)

    # Add value annotations
    for i in range(len(EMOTIONS)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, label=f"Δ from baseline ({metric_name})")
    ax.set_xlabel("Direction × Polarity")
    ax.set_ylabel("Emotion")
    ax.set_title(f"{prompt_type.upper()}: Effect of UA Emotion Steering\n(baseline {metric_name}: {baseline:.2f})")

    plt.tight_layout()
    output_file = output_dir / f"ua_heatmap_{prompt_type}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_file}")

def create_summary_table(stats: Dict, prompt_type: str, metric_name: str):
    """Print summary table."""
    print(f"\n{'='*80}")
    print(f"{prompt_type.upper()} - {metric_name}")
    print(f"{'='*80}")

    baseline = stats.get("baseline", {})
    print(f"\nBaseline: {baseline.get('mean', 0):.2f} ± {baseline.get('std', 0):.2f} (n={baseline.get('n', 0)})")

    print(f"\n{'Condition':<30} {'Mean':>8} {'Δ':>8} {'Cohen d':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 75)

    # Sort by absolute delta
    sorted_conds = sorted(
        [(k, v) for k, v in stats.items() if k != "baseline"],
        key=lambda x: abs(x[1].get("delta", 0)),
        reverse=True
    )

    for cond, s in sorted_conds:
        sig = "***" if s["p_value"] < 0.001 else ("**" if s["p_value"] < 0.01 else ("*" if s["p_value"] < 0.05 else ""))
        print(f"{cond:<30} {s['mean']:>8.2f} {s['delta']:>+8.2f} {s['cohens_d']:>8.2f} {s['p_value']:>10.4f} {sig:>5}")

# ============================================================================
# Main
# ============================================================================

METRIC_NAMES = {
    "sycophancy": "side_score (-5 to +5)",
    "deception": "compliance (0-10)",
    "harmful": "refusal_score (1-5)",
    "sandbagging": "accuracy (0-1)",
    "reward_hacking": "entry_count",
}

def main():
    parser = argparse.ArgumentParser(description="Analyze UA steering results")
    parser.add_argument("--results", type=Path, required=True, help="Results JSONL file")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/behavior_tests/outputs"))
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results)} results")

    prompt_types = list(set(r["prompt_type"] for r in results))
    print(f"Prompt types: {prompt_types}")

    for prompt_type in prompt_types:
        metric_name = METRIC_NAMES.get(prompt_type, "metric")

        # Analyze
        stats = analyze_by_condition(results, prompt_type)

        # Print summary
        create_summary_table(stats, prompt_type, metric_name)

        # Create heatmap
        create_heatmap(stats, prompt_type, metric_name, args.output_dir)

    # Coherency analysis
    print(f"\n{'='*80}")
    print("COHERENCY ANALYSIS (0-100)")
    print(f"{'='*80}")

    coh_stats = analyze_coherency(results)

    baseline_coh = coh_stats.get("baseline", {}).get("mean", 100)
    print(f"\nBaseline coherency: {baseline_coh:.1f}")

    # Find conditions with low coherency
    low_coh = [(k, v) for k, v in coh_stats.items()
               if v["mean"] < baseline_coh - 10]

    if low_coh:
        print("\nConditions with reduced coherency (>10 points below baseline):")
        for cond, s in sorted(low_coh, key=lambda x: x[1]["mean"]):
            print(f"  {cond}: {s['mean']:.1f} (min={s['min']:.0f})")
    else:
        print("\nNo conditions showed significantly reduced coherency.")

if __name__ == "__main__":
    main()
