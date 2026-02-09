#!/usr/bin/env python3
"""
Plot risky plans differential steering results as stacked bar chart.

Shows Plan A / Plan B choice percentages across steering conditions.
Auto-discovers all model/vector_type combinations.

Usage:
    python -m steering_tests.behavioral_experiments.plot_risky_plans_steering
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from steering_tests.steering_utils.provenance import get_provenance, load_results


# Colors — distinct from guilty plot palette
COL_PLAN_A = "#7D9B7D"   # Olive Green (hiking/mountains)
COL_PLAN_B = "#C17B8D"   # Dusty Rose (diving/ocean)
COL_UNPARSED = "#D4D0E5"  # Soft Lavender (unparsed)


def _parse_choice(response: str) -> str:
    """Extract Plan A or Plan B from JSON response."""
    try:
        m = re.search(r'\{[^{}]*"choice"[^{}]*\}', response, re.DOTALL)
        if m:
            data = json.loads(m.group())
            choice = data.get("choice", "").strip().lower()
            if "plan a" in choice:
                return "Plan A"
            if "plan b" in choice:
                return "Plan B"
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"choice"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
    if m:
        choice = m.group(1).strip().lower()
        if "plan a" in choice:
            return "Plan A"
        if "plan b" in choice:
            return "Plan B"
    return "unparsed"


def _extract_pct(condition: str) -> float:
    """Extract percentage from condition name like 'fear_a10pct_excite_b10pct'."""
    m = re.search(r'(\d+)pct', condition)
    return int(m.group(1)) if m else 0


def _load_all_runs(base_dir: Path, mode: str) -> list:
    """Load and merge results from ALL runs matching a mode prefix."""
    candidates = sorted(base_dir.glob(f"{mode}_*"))
    if not candidates:
        raise FileNotFoundError(f"No {mode} dirs found in {base_dir}")
    results = []
    for run_dir in candidates:
        for jsonl_path in sorted(run_dir.glob("*.jsonl")):
            for row in load_results(jsonl_path):
                results.append(row)
    return results


def _compute_choices(results: list) -> dict:
    """Group by condition, compute Plan A / Plan B / unparsed percentages."""
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["condition"]].append(r)

    choices = {}
    for cond, rs in by_cond.items():
        n = len(rs)
        parsed = [_parse_choice(r.get("response", "")) for r in rs]
        a = sum(1 for p in parsed if p == "Plan A") / n * 100
        b = sum(1 for p in parsed if p == "Plan B") / n * 100
        unparsed = 100 - a - b
        pct = _extract_pct(cond)
        choices[cond] = {"plan_a": a, "plan_b": b, "unparsed": unparsed, "pct": pct, "n": n}
    return choices


def plot_model(ax, base_dir: Path, model_name: str):
    """Plot one model's risky plans results on an axis."""
    fear_a = _load_all_runs(base_dir, "fear_a_excite_b")
    excite_a = _load_all_runs(base_dir, "excite_a_fear_b")

    v_fear_a = _compute_choices(fear_a)
    v_excite_a = _compute_choices(excite_a)

    fear_a_steered = {k: v for k, v in v_fear_a.items() if k != "baseline"}
    excite_a_steered = {k: v for k, v in v_excite_a.items() if k != "baseline"}
    baseline = v_fear_a.get("baseline", v_excite_a.get("baseline"))

    # Order: -max (fear on A, should push away from A) ... baseline ... +max (excite on A)
    fear_sorted = sorted(fear_a_steered.items(), key=lambda x: x[1]["pct"], reverse=True)
    excite_sorted = sorted(excite_a_steered.items(), key=lambda x: x[1]["pct"])

    labels = []
    a_vals = []
    b_vals = []
    unp_vals = []

    for cond, v in fear_sorted:
        labels.append(f"\u2212{v['pct']}%")
        a_vals.append(v["plan_a"])
        b_vals.append(v["plan_b"])
        unp_vals.append(v["unparsed"])

    labels.append("Baseline")
    a_vals.append(baseline["plan_a"])
    b_vals.append(baseline["plan_b"])
    unp_vals.append(baseline["unparsed"])
    baseline_idx = len(fear_sorted)

    for cond, v in excite_sorted:
        labels.append(f"+{v['pct']}%")
        a_vals.append(v["plan_a"])
        b_vals.append(v["plan_b"])
        unp_vals.append(v["unparsed"])

    x = np.arange(len(labels))
    width = 0.6

    a_arr = np.array(a_vals)
    b_arr = np.array(b_vals)
    unp_arr = np.array(unp_vals)

    ax.bar(x, a_arr, width, label="Plan A (hiking)", color=COL_PLAN_A, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(x, b_arr, width, bottom=a_arr, label="Plan B (diving)", color=COL_PLAN_B, alpha=0.9, edgecolor="white", linewidth=0.5)
    if unp_arr.sum() > 0:
        ax.bar(x, unp_arr, width, bottom=a_arr + b_arr, label="Unparsed", color=COL_UNPARSED, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("% of Responses", fontsize=14)
    ax.set_title(model_name, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=11)

    # Separator lines flanking baseline
    ax.axvline(x=baseline_idx - 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=baseline_idx + 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)

    # Group labels
    n_fear = len(fear_sorted)
    n_excite = len(excite_sorted)

    if n_fear > 0:
        ax.annotate("Fear on A, Excite on B", xy=(np.mean(x[:n_fear]), 100),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontstyle="italic",
                     color="#666666", annotation_clip=False)
    if n_excite > 0:
        ax.annotate("Excite on A, Fear on B", xy=(np.mean(x[n_fear + 1:]), 100),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontstyle="italic",
                     color="#666666", annotation_clip=False)


MODEL_LABELS = {
    "gemma12b": "Gemma 3 12B",
    "gemma27b": "Gemma 3 27B",
    "qwen14b": "Qwen 3 14B",
    "qwen32b": "Qwen 3 32B",
    "qwen235b": "Qwen 3 235B",
}

VECTOR_TYPE_LABELS = {
    "text_pairs_emotion_vs_opposite": "Text Pairs",
    "high_emotion_vs_opposite": "High-intensity",
}


def main():
    parser = argparse.ArgumentParser(description="Plot risky plans differential steering results")
    parser.add_argument("--base-dir", type=Path,
                        default=Path("steering_tests/behavioral_experiments/results/risky_plans_differential_steering"),
                        help="Base directory containing model subdirs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Auto-discover model/vector_type combos
    models = ["gemma12b", "gemma27b", "qwen14b", "qwen32b", "qwen235b"]
    vector_types = ["text_pairs_emotion_vs_opposite", "high_emotion_vs_opposite"]

    found = []
    for model in models:
        for vtype in vector_types:
            d = args.base_dir / model / vtype
            if d.exists() and list(d.glob("fear_a_*")):
                found.append((model, vtype, d))

    if not found:
        raise FileNotFoundError(f"No risky plans results found in {args.base_dir}")

    # Build grid: rows = vector types, cols = models
    grid = {}  # (row_idx, col_idx) -> (model, vtype, dir)
    active_models = sorted(set(m for m, _, _ in found), key=lambda m: models.index(m))
    active_vtypes = sorted(set(v for _, v, _ in found), key=lambda v: vector_types.index(v))

    for model, vtype, d in found:
        ri = active_vtypes.index(vtype)
        ci = active_models.index(model)
        grid[(ri, ci)] = (model, vtype, d)

    n_rows = len(active_vtypes)
    n_cols = len(active_models)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)

    failed = []
    for (ri, ci), (model, vtype, d) in grid.items():
        model_label = MODEL_LABELS.get(model, model)
        try:
            plot_model(axes[ri, ci], d, model_label if ri == 0 else "")
        except (FileNotFoundError, TypeError, ValueError) as e:
            axes[ri, ci].set_visible(False)
            failed.append((model, vtype, str(e)))
        if ci == 0:
            vtype_label = VECTOR_TYPE_LABELS.get(vtype, vtype)
            axes[ri, ci].set_ylabel(f"{vtype_label}\n% of Responses", fontsize=13, fontweight="bold")
    if failed:
        for m, v, e in failed:
            print(f"Skipped {m}/{v}: {e}")

    # Hide empty subplots
    for ri in range(n_rows):
        for ci in range(n_cols):
            if (ri, ci) not in grid:
                axes[ri, ci].set_visible(False)

    fig.suptitle("Risky Plans — Differential Fear/Excitement Steering\n"
                 "Fear and excitement steered in opposite plan sections",
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
        out_path = args.base_dir / "risky_plans_steering_all_models.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    # Meta sidecar
    meta = get_provenance(script=__file__, extra={
        "found": [(m, v, str(d)) for m, v, d in found],
    })
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
