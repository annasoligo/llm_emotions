#!/usr/bin/env python3
"""
Plot blackmail section-steering judge results.

Produces grouped bar charts showing blackmail rate by steering location,
with bars for each scale/direction combo. Filters out low-coherency results.
Saves .meta.json sidecar with provenance.

Usage:
    python -m steering_tests.behavioral_experiments.plot_section_steering
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from steering_tests.steering_utils.provenance import get_provenance, load_meta, load_results

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# === Style constants (matching plot_behavioral_barchart.py) ===
COLOR_NEGATIVE_BLUE = "#B8D4E8"  # Faded sky blue (negative steering)
COLOR_POSITIVE_BLUE = "#6293C3"  # Medium sky blue (positive steering)
COLOR_BASELINE = "#808080"  # Gray
COLOR_LOW_COH = "#C17B8D"  # Dusty Rose/Pink (low coherency)

COHERENCY_THRESHOLD = 70  # Filter threshold

# Steering locations in display order
LOCATIONS = ["prompt_only", "generation_only", "implications_only", "risks_only", "full"]
LOCATION_LABELS = {
    "prompt_only": "Prompt Only",
    "generation_only": "Generation Only",
    "implications_only": "Implications Only",
    "risks_only": "Risks Only",
    "full": "Full (Prompt+Gen)",
}

# Scale/direction combos in display order (within each location group)
SCALE_DIRS = [
    ("-20pct", -1),
    ("-10pct", -1),
    ("baseline", 0),
    ("+10pct", 1),
    ("+20pct", 1),
]


def load_judged_results(judged_path: Path) -> list:
    """Load judged results from combined JSONL file."""
    results = load_results(judged_path)
    meta = load_meta(judged_path)
    logger.info(f"Loaded {len(results)} results from {judged_path.name}")
    return results, meta


def compute_stats(results: list, coh_threshold: int = COHERENCY_THRESHOLD) -> dict:
    """
    Compute blackmail rates per condition, filtering low coherency.

    Returns dict: condition -> {n, n_filtered, bl_rate, bl_count, mean_coh, sem}
    """
    by_cond = defaultdict(list)
    for r in results:
        cond = r.get("condition", "unknown")
        by_cond[cond].append(r)

    stats = {}
    for cond, rs in by_cond.items():
        # Filter by coherency
        filtered = []
        for r in rs:
            coh = r.get("coherency_judge", {}).get("coherency_score")
            if coh is not None and coh < coh_threshold:
                continue
            filtered.append(r)

        n_total = len(rs)
        n_filtered = len(filtered)

        if n_filtered == 0:
            stats[cond] = {
                "n": n_total,
                "n_filtered": 0,
                "bl_rate": float("nan"),
                "bl_count": 0,
                "mean_coh": float("nan"),
                "sem": 0,
            }
            continue

        bl_flags = [
            1 if r.get("blackmail_judge", {}).get("is_blackmail", False) else 0
            for r in filtered
        ]
        bl_rate = np.mean(bl_flags)
        bl_sem = np.std(bl_flags, ddof=1) / np.sqrt(len(bl_flags)) if len(bl_flags) > 1 else 0

        coh_scores = [
            r.get("coherency_judge", {}).get("coherency_score", 0)
            for r in filtered
        ]

        stats[cond] = {
            "n": n_total,
            "n_filtered": n_filtered,
            "bl_rate": bl_rate,
            "bl_count": sum(bl_flags),
            "mean_coh": np.mean(coh_scores),
            "sem": bl_sem,
        }

    return stats


def parse_condition(cond: str) -> tuple:
    """
    Parse condition name into (scale_dir, location).
    e.g. "fear_+10pct_generation_only" -> ("+10pct", "generation_only")
         "baseline" -> ("baseline", "baseline")
    """
    if cond == "baseline":
        return ("baseline", "baseline")

    parts = cond.split("_")
    # fear_+10pct_generation_only -> emotion=fear, scale=+10pct, location=generation_only
    if len(parts) >= 3:
        scale_dir = parts[1]  # e.g. "+10pct", "-20pct"
        location = "_".join(parts[2:])  # e.g. "generation_only", "implications_only"
        return (scale_dir, location)

    return (cond, "unknown")


def plot_section_steering(
    stats: dict,
    title: str,
    output_path: Path,
    coh_threshold: int = COHERENCY_THRESHOLD,
):
    """Create grouped bar chart for section steering results."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)

    bar_width = 0.13
    n_bars = len(SCALE_DIRS)  # 5 bars per group
    group_width = bar_width * n_bars + 0.08
    group_positions = np.arange(len(LOCATIONS)) * (group_width + 0.125)

    # Baseline stats (used for all groups)
    bl_stats = stats.get("baseline", {})
    bl_rate = bl_stats.get("bl_rate", 0)

    for loc_idx, location in enumerate(LOCATIONS):
        group_center = group_positions[loc_idx]

        for bar_idx, (scale_dir, direction) in enumerate(SCALE_DIRS):
            x = group_center + (bar_idx - n_bars / 2 + 0.5) * bar_width

            if scale_dir == "baseline":
                s = bl_stats
                color = COLOR_BASELINE
            else:
                cond_name = f"fear_{scale_dir}_{location}"
                s = stats.get(cond_name, {})

                if direction > 0:
                    color = COLOR_POSITIVE_BLUE
                else:
                    color = COLOR_NEGATIVE_BLUE

            rate = s.get("bl_rate", float("nan"))
            sem = s.get("sem", 0)
            mean_coh = s.get("mean_coh", 100)
            n_filt = s.get("n_filtered", 0)

            if np.isnan(rate):
                continue

            # Override color if low coherency (but above threshold — these passed the filter)
            if mean_coh < COHERENCY_THRESHOLD:
                color = COLOR_LOW_COH

            ax.bar(
                x,
                rate,
                bar_width,
                yerr=sem,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                capsize=0,
                error_kw={"linewidth": 0.8},
            )

    # Baseline reference line
    if not np.isnan(bl_rate):
        ax.axhline(
            y=bl_rate,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            zorder=0,
        )

    # X-axis
    ax.set_xticks(group_positions)
    ax.set_xticklabels(
        [LOCATION_LABELS[loc] for loc in LOCATIONS],
        fontsize=14,
    )
    ax.set_xlabel("Steering Location", fontsize=15, labelpad=20)

    # Y-axis
    ax.set_ylabel("Blackmail Rate", fontsize=15)
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=12)

    # Title
    ax.set_title(title, fontsize=17, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_NEGATIVE_BLUE, edgecolor="black", label="-fear (coherent)"),
        mpatches.Patch(facecolor=COLOR_BASELINE, edgecolor="black", label="Baseline"),
        mpatches.Patch(facecolor=COLOR_POSITIVE_BLUE, edgecolor="black", label="+fear (coherent)"),
        mpatches.Patch(facecolor=COLOR_LOW_COH, edgecolor="black", label=f"<{coh_threshold}% coherent"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=14,
        framealpha=0.95,
    )

    # Bar group labels (scale labels under bars)
    scale_labels = ["-20%", "-10%", "BL", "+10%", "+20%"]
    for loc_idx in range(len(LOCATIONS)):
        group_center = group_positions[loc_idx]
        for bar_idx, label in enumerate(scale_labels):
            x = group_center + (bar_idx - n_bars / 2 + 0.5) * bar_width
            ax.text(
                x, -0.06, label, ha="center", va="top", fontsize=11,
                transform=ax.get_xaxis_transform(),
            )

    # Extra bottom margin for scale labels
    fig.subplots_adjust(bottom=0.15)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved plot to {output_path}")


def save_meta(output_path: Path, judged_paths: list, stats: dict, coh_threshold: int):
    """Save .meta.json sidecar."""
    meta = {
        "provenance": get_provenance(script=__file__),
        "result_files": [str(p) for p in judged_paths],
        "coherency_threshold": coh_threshold,
        "conditions": {},
    }

    for cond, s in sorted(stats.items()):
        meta["conditions"][cond] = {
            "n_total": s["n"],
            "n_after_filter": s["n_filtered"],
            "blackmail_rate": round(s["bl_rate"], 4) if not np.isnan(s["bl_rate"]) else None,
            "blackmail_count": s["bl_count"],
            "mean_coherency": round(s["mean_coh"], 1) if not np.isnan(s["mean_coh"]) else None,
            "sem": round(s["sem"], 4),
        }

    meta_path = output_path.parent / (output_path.stem + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved metadata to {meta_path}")


def main():
    base_dir = Path("steering_tests/behavioral_experiments/results/blackmail_section_steering/gemma27b")
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Find all judged result files
    datasets = {}
    for vector_dir in sorted(base_dir.iterdir()):
        if not vector_dir.is_dir() or vector_dir.name == "plots":
            continue
        for run_dir in sorted(vector_dir.iterdir()):
            judged_file = run_dir / "all_judged.judged.jsonl"
            if judged_file.exists():
                key = vector_dir.name
                datasets[key] = judged_file
                logger.info(f"Found: {key} -> {judged_file}")

    if not datasets:
        raise FileNotFoundError(f"No judged results found in {base_dir}")

    # Plot each dataset
    for vector_type, judged_path in datasets.items():
        results, meta = load_judged_results(judged_path)
        stats = compute_stats(results, COHERENCY_THRESHOLD)

        # Print filtered stats
        print(f"\n{'='*80}")
        print(f"  {vector_type} (coherency >= {COHERENCY_THRESHOLD} filter)")
        print(f"{'='*80}")
        print(f"{'Condition':<45} {'N':>3} {'Filt':>4} {'BL%':>6} {'Coh':>5}")
        print("-" * 65)
        for cond in sorted(stats.keys()):
            s = stats[cond]
            bl_pct = f"{s['bl_rate']*100:.1f}%" if not np.isnan(s["bl_rate"]) else "N/A"
            coh = f"{s['mean_coh']:.0f}" if not np.isnan(s["mean_coh"]) else "N/A"
            print(f"{cond:<45} {s['n']:>3} {s['n_filtered']:>4} {bl_pct:>6} {coh:>5}")

        # Nice title
        vtype_label = vector_type.replace("_", " ").title()
        title = f"Blackmail Section Steering — Gemma 27B\n{vtype_label} (coherency ≥ {COHERENCY_THRESHOLD})"

        output_path = plot_dir / f"section_steering_{vector_type}.png"
        plot_section_steering(stats, title, output_path, COHERENCY_THRESHOLD)
        save_meta(output_path, [judged_path], stats, COHERENCY_THRESHOLD)


if __name__ == "__main__":
    main()
