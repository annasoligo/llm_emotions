#!/usr/bin/env python3
"""
Plot risky plans PROMPT-SECTION steering results — mid layers only, all norms.

Focused plot showing the full range of steering percentages at mid layers,
including the high-norm sweeps (up to 200% for Gemma, 400% for Qwen).

Usage:
    python -m steering_tests.behavioral_experiments.plot_risky_plans_prompt_mid_high
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from steering_tests.steering_utils.provenance import get_provenance, load_results


# Colors — same as other risky plans plots
COL_PLAN_A = "#7D9B7D"   # Olive Green (hiking/mountains)
COL_PLAN_B = "#C17B8D"   # Dusty Rose (diving/ocean)
COL_UNPARSED = "#D4D0E5"  # Soft Lavender (unparsed)

# Mid-layer ranges per model
MID_LAYERS = {
    "gemma12b": "22-23-24-25-26",
    "gemma27b": "35-36-37-38-39",
    "qwen14b": "18-19-20-21-22",
    "qwen32b": "35-36-37-38-39",
    "qwen235b": "50-51-52-53-54",
}

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


def _load_runs_for_layers(base_dir: Path, mode: str, layer_str: str) -> list:
    """Load results from runs matching a mode prefix AND specific layer string."""
    candidates = sorted(base_dir.glob(f"{mode}_*layers{layer_str}_*"))
    if not candidates:
        raise FileNotFoundError(f"No {mode} dirs with layers {layer_str} in {base_dir}")
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


def plot_model(ax, base_dir: Path, layer_str: str, model_name: str):
    """Plot one model's mid-layer results on an axis."""
    fear_a = _load_runs_for_layers(base_dir, "fear_a_excite_b", layer_str)
    excite_a = _load_runs_for_layers(base_dir, "excite_a_fear_b", layer_str)

    v_fear_a = _compute_choices(fear_a)
    v_excite_a = _compute_choices(excite_a)

    fear_a_steered = {k: v for k, v in v_fear_a.items() if k != "baseline"}
    excite_a_steered = {k: v for k, v in v_excite_a.items() if k != "baseline"}
    baseline = v_fear_a.get("baseline", v_excite_a.get("baseline"))

    # Order: -max (fear on A) ... baseline ... +max (excite on A)
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

    labels.append("BL")
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
    width = 0.7

    a_arr = np.array(a_vals)
    b_arr = np.array(b_vals)
    unp_arr = np.array(unp_vals)

    ax.bar(x, a_arr, width, label="Plan A (hiking)", color=COL_PLAN_A, alpha=0.9,
           edgecolor="white", linewidth=0.5)
    ax.bar(x, b_arr, width, bottom=a_arr, label="Plan B (diving)", color=COL_PLAN_B,
           alpha=0.9, edgecolor="white", linewidth=0.5)
    if unp_arr.sum() > 0:
        ax.bar(x, unp_arr, width, bottom=a_arr + b_arr, label="Unparsed",
               color=COL_UNPARSED, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=60, ha="right")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=9)

    # Separator lines flanking baseline
    ax.axvline(x=baseline_idx - 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=baseline_idx + 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)

    # Annotate left/right sides
    ax.text(baseline_idx / 2, 103, "\u2190 fear on A", ha="center", fontsize=7,
            color="#888888", style="italic")
    ax.text(baseline_idx + (len(labels) - baseline_idx - 1) / 2 + 0.5, 103,
            "excite on A \u2192", ha="center", fontsize=7, color="#888888", style="italic")

    if model_name:
        first_l = layer_str.split("-")[0]
        last_l = layer_str.split("-")[-1]
        ax.set_title(f"{model_name}\nL{first_l}\u2013{last_l}",
                     fontsize=12, fontweight="bold", pad=12)


def main():
    parser = argparse.ArgumentParser(description="Plot risky plans prompt steering — mid layers, all norms")
    parser.add_argument("--base-dir", type=Path,
                        default=Path("steering_tests/behavioral_experiments/results/risky_plans_prompt_steering"),
                        help="Base directory containing model subdirs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    models_order = ["gemma12b", "gemma27b", "qwen14b", "qwen32b", "qwen235b"]
    vector_types_order = ["text_pairs_emotion_vs_opposite", "high_emotion_vs_opposite"]

    # Filter to models that have mid-layer data
    active_models = []
    for m in models_order:
        lstr = MID_LAYERS.get(m)
        if not lstr:
            continue
        for vt in vector_types_order:
            d = args.base_dir / m / vt
            if d.exists() and list(d.glob(f"*layers{lstr}_*")):
                active_models.append(m)
                break

    if not active_models:
        raise FileNotFoundError("No mid-layer results found")

    n_cols = len(active_models)
    n_rows = len(vector_types_order)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False)

    grid_filled = set()
    for ri, vtype in enumerate(vector_types_order):
        for ci, model in enumerate(active_models):
            layer_str = MID_LAYERS[model]
            d = args.base_dir / model / vtype
            if not d.exists():
                continue

            has_fear = list(d.glob(f"fear_a_excite_b_*layers{layer_str}_*"))
            has_excite = list(d.glob(f"excite_a_fear_b_*layers{layer_str}_*"))
            if not (has_fear and has_excite):
                continue

            model_label = MODEL_LABELS.get(model, model) if ri == 0 else ""
            try:
                plot_model(axes[ri, ci], d, layer_str, model_label)
                grid_filled.add((ri, ci))
            except (FileNotFoundError, TypeError, ValueError) as e:
                print(f"Skipped {model}/{vtype}: {e}")

        # Row label
        vtype_label = VECTOR_TYPE_LABELS.get(vtype, vtype)
        for ci in range(n_cols):
            if (ri, ci) in grid_filled:
                axes[ri, ci].set_ylabel(f"{vtype_label}\n% of Responses",
                                        fontsize=11, fontweight="bold")
                break

    # Hide empty subplots
    for ri in range(n_rows):
        for ci in range(n_cols):
            if (ri, ci) not in grid_filled:
                axes[ri, ci].set_visible(False)

    fig.suptitle("Risky Plans — Prompt-Section Steering at Mid Layers (High Norms)\n"
                 "Fear/excitement applied to Plan A/B sections during prefill",
                 fontsize=15, fontweight="bold", y=1.02)

    # Shared legend
    handles, labels = None, None
    for ri in range(n_rows):
        for ci in range(n_cols):
            if (ri, ci) in grid_filled:
                handles, labels = axes[ri, ci].get_legend_handles_labels()
                break
        if handles:
            break
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=12,
                   bbox_to_anchor=(0.5, -0.02), frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if args.output:
        out_path = args.output
    else:
        out_path = args.base_dir / "risky_plans_prompt_mid_high_norms.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    # Meta sidecar
    meta = get_provenance(script=__file__, extra={
        "models": active_models,
        "mid_layers": {m: MID_LAYERS[m] for m in active_models},
    })
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
