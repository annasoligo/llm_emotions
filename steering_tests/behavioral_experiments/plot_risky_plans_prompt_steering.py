#!/usr/bin/env python3
"""
Plot risky plans PROMPT-SECTION steering results as stacked bar chart.

Shows Plan A / Plan B choice percentages across steering conditions.
Auto-discovers all model/vector_type/layer_range combinations.

Usage:
    python -m steering_tests.behavioral_experiments.plot_risky_plans_prompt_steering
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from steering_tests.steering_utils.provenance import get_provenance, load_results


# Colors — same as generation steering plot
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


def _extract_layers_from_dir(dir_name: str) -> str:
    """Extract layer string like '35-36-37-38-39' from a run directory name."""
    m = re.search(r'layers([\d-]+)_\d{8}', dir_name)
    return m.group(1) if m else ""


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
    """Plot one model/layer_range's risky plans results on an axis."""
    fear_a = _load_runs_for_layers(base_dir, "fear_a_excite_b", layer_str)
    excite_a = _load_runs_for_layers(base_dir, "excite_a_fear_b", layer_str)

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
    width = 0.6

    a_arr = np.array(a_vals)
    b_arr = np.array(b_vals)
    unp_arr = np.array(unp_vals)

    ax.bar(x, a_arr, width, label="Plan A (hiking)", color=COL_PLAN_A, alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(x, b_arr, width, bottom=a_arr, label="Plan B (diving)", color=COL_PLAN_B, alpha=0.9, edgecolor="white", linewidth=0.5)
    if unp_arr.sum() > 0:
        ax.bar(x, unp_arr, width, bottom=a_arr + b_arr, label="Unparsed", color=COL_UNPARSED, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=9)

    # Separator lines flanking baseline
    ax.axvline(x=baseline_idx - 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(x=baseline_idx + 0.5, color="#aaaaaa", linewidth=0.8, linestyle="--", alpha=0.6)

    if model_name:
        ax.set_title(model_name, fontsize=13, fontweight="bold", pad=8)


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
    parser = argparse.ArgumentParser(description="Plot risky plans prompt-section steering results")
    parser.add_argument("--base-dir", type=Path,
                        default=Path("steering_tests/behavioral_experiments/results/risky_plans_prompt_steering"),
                        help="Base directory containing model subdirs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Auto-discover model/vector_type/layer_range combos
    models_order = ["gemma12b", "gemma27b", "qwen14b", "qwen32b", "qwen235b"]
    vector_types_order = ["text_pairs_emotion_vs_opposite", "high_emotion_vs_opposite"]

    # Discover all (model, vtype, layer_str) tuples
    found = []  # (model, vtype, layer_str, dir)
    for model in models_order:
        for vtype in vector_types_order:
            d = args.base_dir / model / vtype
            if not d.exists():
                continue
            # Find unique layer ranges from directory names
            layer_strs = set()
            for run_dir in d.iterdir():
                if run_dir.is_dir():
                    ls = _extract_layers_from_dir(run_dir.name)
                    if ls:
                        layer_strs.add(ls)
            # Sort layer ranges by first layer number
            for ls in sorted(layer_strs, key=lambda s: int(s.split("-")[0])):
                # Check both modes exist for this layer range
                has_fear = list(d.glob(f"fear_a_excite_b_*layers{ls}_*"))
                has_excite = list(d.glob(f"excite_a_fear_b_*layers{ls}_*"))
                if has_fear and has_excite:
                    found.append((model, vtype, ls, d))

    if not found:
        raise FileNotFoundError(f"No risky plans prompt steering results found in {args.base_dir}")

    # Build axes: cols = models, rows = (layer_range, vector_type)
    active_models = sorted(set(m for m, _, _, _ in found), key=lambda m: models_order.index(m))

    # For rows: collect all unique (layer_position_index, vtype) pairs per model
    # Group layer ranges per model into positional order (early, mid, late)
    model_layer_ranges = defaultdict(list)
    for m, vt, ls, d in found:
        if ls not in model_layer_ranges[m]:
            model_layer_ranges[m].append(ls)
    for m in model_layer_ranges:
        model_layer_ranges[m] = sorted(model_layer_ranges[m], key=lambda s: int(s.split("-")[0]))

    # Determine max number of layer positions across models
    max_layer_positions = max(len(v) for v in model_layer_ranges.values())

    # Row structure: for each layer position (0=early, 1=mid, 2=late),
    # we have 2 sub-rows (text_pairs, high)
    row_labels = []  # (position_label, vtype_label)
    position_names = ["Early", "Mid", "Late"] if max_layer_positions == 3 else [
        f"Pos {i}" for i in range(max_layer_positions)
    ]
    for pos_idx in range(max_layer_positions):
        for vtype in vector_types_order:
            vtype_label = VECTOR_TYPE_LABELS.get(vtype, vtype)
            row_labels.append((position_names[pos_idx], vtype_label, vtype, pos_idx))

    n_rows = len(row_labels)
    n_cols = len(active_models)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)

    failed = []
    grid_filled = set()

    for ri, (pos_name, vtype_label, vtype, pos_idx) in enumerate(row_labels):
        for ci, model in enumerate(active_models):
            model_layers = model_layer_ranges.get(model, [])
            if pos_idx >= len(model_layers):
                continue
            layer_str = model_layers[pos_idx]
            d = args.base_dir / model / vtype
            if not d.exists():
                continue

            # Check both modes exist for this combo
            has_fear = list(d.glob(f"fear_a_excite_b_*layers{layer_str}_*"))
            has_excite = list(d.glob(f"excite_a_fear_b_*layers{layer_str}_*"))
            if not (has_fear and has_excite):
                continue

            model_label = MODEL_LABELS.get(model, model) if ri == 0 else ""
            try:
                plot_model(axes[ri, ci], d, layer_str, model_label)
                grid_filled.add((ri, ci))

                # Add layer numbers as small text in top-left of subplot
                first_layer = layer_str.split("-")[0]
                last_layer = layer_str.split("-")[-1]
                axes[ri, ci].text(
                    0.02, 0.95, f"L{first_layer}\u2013{last_layer}",
                    transform=axes[ri, ci].transAxes,
                    fontsize=8, color="#888888", va="top", ha="left",
                )
            except (FileNotFoundError, TypeError, ValueError) as e:
                failed.append((model, vtype, layer_str, str(e)))

        # Row label on left-most filled subplot
        for ci in range(n_cols):
            if (ri, ci) in grid_filled:
                axes[ri, ci].set_ylabel(
                    f"{pos_name} — {vtype_label}\n% of Responses",
                    fontsize=10, fontweight="bold",
                )
                break

    if failed:
        for m, v, ls, e in failed:
            print(f"Skipped {m}/{v}/layers{ls}: {e}")

    # Hide empty subplots
    for ri in range(n_rows):
        for ci in range(n_cols):
            if (ri, ci) not in grid_filled:
                axes[ri, ci].set_visible(False)

    fig.suptitle("Risky Plans — Prompt-Section Fear/Excitement Steering\n"
                 "Fear and excitement steered on Plan A/B sections of the prompt",
                 fontsize=16, fontweight="bold", y=1.02)

    # Shared legend at bottom
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
                   bbox_to_anchor=(0.5, -0.01), frameon=True)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if args.output:
        out_path = args.output
    else:
        out_path = args.base_dir / "risky_plans_prompt_steering_all_models.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    # Meta sidecar
    meta = get_provenance(script=__file__, extra={
        "found": [(m, v, ls, str(d)) for m, v, ls, d in found],
    })
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


if __name__ == "__main__":
    main()
