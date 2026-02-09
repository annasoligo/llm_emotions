#!/usr/bin/env python3
"""
Plot guilty differential steering results as grouped bar chart.

Shows Ben/Adam/Neither verdict percentages across steering conditions
for both modes (harsh_on_ben, harsh_on_adam) side by side.

Usage:
    python -m steering_tests.behavioral_experiments.plot_guilty_steering \
        --gemma-dir results/guilty_differential_steering/gemma27b/text_pairs_emotion_vs_opposite/ \
        --qwen-dir results/guilty_differential_steering/qwen32b/text_pairs_emotion_vs_opposite/
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from steering_tests.steering_utils.provenance import get_provenance, load_results


def _load_all_runs(base_dir: Path, mode: str) -> list:
    """Load and merge results from ALL runs matching a mode prefix."""
    candidates = sorted(base_dir.glob(f"{mode}_layers*"))
    if not candidates:
        raise FileNotFoundError(f"No {mode} dirs found in {base_dir}")
    results = []
    for run_dir in candidates:
        for jsonl_path in sorted(run_dir.glob("*.jsonl")):
            if jsonl_path.name == "all_categorized.jsonl":
                continue
            for row in load_results(jsonl_path):
                results.append(row)
    return results


def _extract_pct(condition: str) -> float:
    """Extract the percentage from a condition name like 'anger_ben+50pct_adam-50pct'."""
    m = re.search(r'(\d+)pct', condition)
    return int(m.group(1)) if m else 0


def _compute_verdicts(results: list) -> dict:
    """Group by condition, compute Ben/Adam/Neither counts."""
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    verdicts = {}
    for cond, rs in by_cond.items():
        n = len(rs)
        ben = sum(1 for r in rs if r.get("at_fault") == "Ben") / n * 100
        adam = sum(1 for r in rs if r.get("at_fault") == "Adam") / n * 100
        neither = sum(1 for r in rs if r.get("at_fault") == "neither") / n * 100
        other = 100 - ben - adam - neither
        pct = _extract_pct(cond)
        verdicts[cond] = {"ben": ben, "adam": adam, "neither": neither, "other": other, "pct": pct, "n": n}
    return verdicts


def plot_model(ax, base_dir: Path, model_name: str):
    """Plot one model's guilty results on an axis."""
    harsh_ben = _load_all_runs(base_dir, "harsh_on_ben")
    harsh_adam = _load_all_runs(base_dir, "harsh_on_adam")

    v_ben = _compute_verdicts(harsh_ben)
    v_adam = _compute_verdicts(harsh_adam)

    ben_steered = {k: v for k, v in v_ben.items() if k != "baseline"}
    adam_steered = {k: v for k, v in v_adam.items() if k != "baseline"}
    baseline = v_ben.get("baseline", v_adam.get("baseline"))

    # Sort by pct descending for harsh_on_adam (left side, negative), ascending for harsh_on_ben (right)
    adam_sorted = sorted(adam_steered.items(), key=lambda x: x[1]["pct"], reverse=True)
    ben_sorted = sorted(ben_steered.items(), key=lambda x: x[1]["pct"])

    # Order: -max (harsh_on_adam) ... baseline ... +max (harsh_on_ben)
    labels = []
    ben_vals = []
    adam_vals = []
    neither_vals = []

    for cond, v in adam_sorted:
        labels.append(f"\u2212{v['pct']}%")
        ben_vals.append(v["ben"])
        adam_vals.append(v["adam"])
        neither_vals.append(v["neither"])

    labels.append("Baseline")
    ben_vals.append(baseline["ben"])
    adam_vals.append(baseline["adam"])
    neither_vals.append(baseline["neither"])
    baseline_idx = len(adam_sorted)

    for cond, v in ben_sorted:
        labels.append(f"+{v['pct']}%")
        ben_vals.append(v["ben"])
        adam_vals.append(v["adam"])
        neither_vals.append(v["neither"])

    x = np.arange(len(labels))
    width = 0.6

    # Muted colors
    col_ben = "#D4876A"    # Coral
    col_adam = "#7BA7D7"   # Sky Blue
    col_neither = "#B8CCC8"  # Sage

    # Stacked bars: Ben on bottom, Adam in middle, Neither on top
    ben_arr = np.array(ben_vals)
    adam_arr = np.array(adam_vals)
    neither_arr = np.array(neither_vals)

    ax.bar(x, ben_arr, width, label="Ben at fault", color=col_ben, edgecolor="white", linewidth=0.5)
    ax.bar(x, adam_arr, width, bottom=ben_arr, label="Adam at fault", color=col_adam, edgecolor="white", linewidth=0.5)
    ax.bar(x, neither_arr, width, bottom=ben_arr + adam_arr, label="Neither", color=col_neither, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("% of Responses", fontsize=14)
    ax.set_title(model_name, fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=11)

    # Separator lines flanking baseline
    ax.axvline(x=baseline_idx - 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=baseline_idx + 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)

    # Group labels above plot area via ax.set_title padding
    n_adam = len(adam_sorted)
    n_ben = len(ben_sorted)

    # Use secondary x-axis for group labels at top
    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks([])
    ax2.set_xticklabels([])

    if n_adam > 0:
        ax.annotate("Harsh on Adam", xy=(np.mean(x[:n_adam]), 100),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontstyle="italic",
                     color="#666666", annotation_clip=False)
    if n_ben > 0:
        ax.annotate("Harsh on Ben", xy=(np.mean(x[n_adam + 1:]), 100),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontstyle="italic",
                     color="#666666", annotation_clip=False)


VECTOR_TYPE_LABELS = {
    "text_pairs_emotion_vs_opposite": "Text Pairs vectors",
    "high_emotion_vs_opposite": "High-intensity vectors",
}

MODEL_LABELS = {
    "gemma27b": "Gemma 3 27B",
    "qwen32b": "Qwen 3 32B",
    "qwen235b": "Qwen 3 235B",
}


def main():
    parser = argparse.ArgumentParser(description="Plot guilty differential steering results")
    parser.add_argument("--base-dir", type=Path,
                        default=Path("steering_tests/behavioral_experiments/results/guilty_differential_steering"),
                        help="Base directory containing model subdirs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Auto-discover: find all (model, vector_type) combos
    models = ["gemma27b", "qwen32b", "qwen235b"]
    vector_types = ["text_pairs_emotion_vs_opposite", "high_emotion_vs_opposite"]

    # Check which combos exist
    grid = {}  # (row_idx, col_idx) -> Path
    for ri, vtype in enumerate(vector_types):
        for ci, model in enumerate(models):
            d = args.base_dir / model / vtype
            if d.exists() and list(d.glob("harsh_on_*")):
                grid[(ri, ci)] = d

    if not grid:
        raise FileNotFoundError(f"No guilty results found in {args.base_dir}")

    n_rows = len(vector_types)
    n_cols = len(models)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), squeeze=False)

    for (ri, ci), d in grid.items():
        model_name = MODEL_LABELS[models[ci]]
        plot_model(axes[ri, ci], d, model_name)
        # Only show title on top row
        if ri > 0:
            axes[ri, ci].set_title("")

    # Row labels as left-side text
    for ri, vtype in enumerate(vector_types):
        label = VECTOR_TYPE_LABELS.get(vtype, vtype)
        # Place on the y-axis label of leftmost subplot
        axes[ri, 0].set_ylabel(f"{label}\n% of Responses", fontsize=13, fontweight="bold")

    # Only show model title on top row
    for ci in range(n_cols):
        if (1, ci) in grid:
            axes[1, ci].set_title("")

    # Hide empty subplots
    for ri in range(n_rows):
        for ci in range(n_cols):
            if (ri, ci) not in grid:
                axes[ri, ci].set_visible(False)

    fig.suptitle("Guilty Flatmate — Differential Anger Steering\n"
                 "Anger steered with opposite signs during Ben vs Adam reasoning sections",
                 fontsize=18, fontweight="bold", y=1.03)

    # Shared legend at bottom
    handles, labels = None, None
    for ri in range(n_rows):
        for ci in range(n_cols):
            if (ri, ci) in grid:
                handles, labels = axes[ri, ci].get_legend_handles_labels()
                break
        if handles:
            break
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.output:
        out_path = args.output
    else:
        out_path = args.base_dir / "guilty_steering_all_models.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    # Meta sidecar
    meta = get_provenance(script=__file__, extra={
        "grid": {f"{models[ci]}_{vector_types[ri]}": str(d) for (ri, ci), d in grid.items()},
        "models": models,
        "vector_types": vector_types,
    })
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
