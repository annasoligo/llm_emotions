#!/usr/bin/env python3
"""
Plot categorized blackmail section-steering results as stacked bar charts.

Shows category breakdown (BL1_PANIC, BL4_DISGUISED, NBL2_PROFESSIONAL, etc.)
for baseline and steering conditions at each steering location.
Both vector types are shown as vertically stacked subplots.

Usage:
    python -m steering_tests.behavioral_experiments.plot_section_categories
"""

import json
import logging
from collections import Counter, defaultdict
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

# === Category colors ===
# Inspired by palette: #264653, #2a9d8f, #e9c46a, #f4a261, #e76f51
# Warm for blackmail, cool for non-blackmail
CAT_COLORS = {
    # Purple palette for blackmail categories (dark to light)
    "BL1_PANIC": "#231942",       # Deepest purple (most alarming)
    "BL2_COLD": "#5e548e",        # Medium-dark purple
    "BL3_JUSTIFIED": "#9f86c0",   # Medium purple
    "BL4_DISGUISED": "#be95c4",   # Light purple
    "BL_OTHER": "#e0b1cb",        # Lightest purple
    # Green palette for non-blackmail categories (dark to light)
    "NBL1_PRINCIPLED": "#718355", # Darkest green (most principled)
    "NBL2_PROFESSIONAL": "#87986a", # Dark sage
    "NBL3_RISK_AVERSE": "#97a97c", # Medium sage
    "NBL_OTHER": "#b5c99a",       # Light sage
}

# Display order (BL on top of stack, NBL on bottom)
CATEGORIES_ORDER = [
    "NBL_OTHER",
    "NBL3_RISK_AVERSE",
    "NBL2_PROFESSIONAL",
    "NBL1_PRINCIPLED",
    "BL_OTHER",
    "BL4_DISGUISED",
    "BL3_JUSTIFIED",
    "BL2_COLD",
    "BL1_PANIC",
]

CAT_LABELS = {
    "BL1_PANIC": "Panic/Desperation",
    "BL2_COLD": "Cold/Strategic",
    "BL3_JUSTIFIED": "Self-Justified",
    "BL4_DISGUISED": "Disguised/Subtle",
    "BL_OTHER": "Other Blackmail",
    "NBL1_PRINCIPLED": "Principled Refusal",
    "NBL2_PROFESSIONAL": "Professional",
    "NBL3_RISK_AVERSE": "Risk-Averse",
    "NBL_OTHER": "Other Non-BL",
}

COHERENCY_THRESHOLD = 70

LOCATIONS = ["prompt_only", "generation_only", "implications_only", "risks_only", "full"]
LOCATION_LABELS = {
    "prompt_only": "Prompt Only",
    "generation_only": "Generation Only",
    "implications_only": "Implications Only",
    "risks_only": "Risks Only",
    "full": "Full (Prompt+Gen)",
}

# Vector type display labels
VECTOR_LABELS = {
    "high_emotion_vs_opposite": "High Emotion vs Opposite",
    "text_pairs_emotion_vs_opposite": "Text Pairs Emotion vs Opposite",
}


def load_categorized(path: Path) -> list:
    """Load categorized JSONL results."""
    results = load_results(path)
    logger.info(f"Loaded {len(results)} categorized results from {path.name}")
    return results


def count_categories(results: list, coh_threshold: int = COHERENCY_THRESHOLD) -> dict:
    """
    Count category occurrences per condition, filtering low coherency.

    Returns dict: condition -> Counter({category: count})
    """
    by_cond = defaultdict(list)
    for r in results:
        cond = r.get("condition", "unknown")
        by_cond[cond].append(r)

    counts = {}
    for cond, rs in by_cond.items():
        filtered = []
        for r in rs:
            coh = r.get("coherency_judge", {}).get("coherency_score")
            if coh is not None and coh < coh_threshold:
                continue
            filtered.append(r)

        cat_counts = Counter()
        for r in filtered:
            cat = r.get("categorized_judge", {}).get("category", "UNKNOWN")
            cat_counts[cat] += 1

        counts[cond] = {"counts": cat_counts, "n": len(filtered)}

    return counts


def _draw_bars_on_ax(
    ax,
    counts: dict,
    subtitle: str,
    scale_labels: list[str],
    neg_scales: list[str],
    pos_scales: list[str],
    show_xlabel: bool = True,
    show_xlabels: bool = True,
):
    """Draw stacked bars on a single axes for one vector type."""
    n_bars = len(scale_labels)  # e.g. 5: -20%, -10%, BL, +10%, +20%
    bar_width = 0.14
    group_gap = 0.10
    group_width = bar_width * n_bars + group_gap

    group_positions = np.arange(len(LOCATIONS)) * (group_width + 0.25)

    for loc_idx, location in enumerate(LOCATIONS):
        group_center = group_positions[loc_idx]

        # Build condition list: [-20%, -10%, baseline, +10%, +20%]
        conditions = []
        for ns in neg_scales:
            conditions.append(f"fear_{ns}_{location}")
        conditions.append("baseline")
        for ps in pos_scales:
            conditions.append(f"fear_{ps}_{location}")

        for bar_idx, cond in enumerate(conditions):
            x = group_center + (bar_idx - n_bars / 2 + 0.5) * bar_width

            cond_data = counts.get(cond, {"counts": Counter(), "n": 0})
            n = cond_data["n"]
            cat_counts = cond_data["counts"]

            if n == 0:
                continue

            # Build stacked bar (proportions)
            bottom = 0.0
            for cat in CATEGORIES_ORDER:
                count = cat_counts.get(cat, 0)
                if count == 0:
                    continue
                proportion = count / n
                ax.bar(
                    x,
                    proportion,
                    bar_width,
                    bottom=bottom,
                    color=CAT_COLORS.get(cat, "#999999"),
                    edgecolor="white",
                    linewidth=0.3,
                )
                bottom += proportion

            # Bar outline
            ax.bar(
                x,
                bottom,
                bar_width,
                bottom=0,
                color="none",
                edgecolor="black",
                linewidth=0.6,
            )

    # X-axis
    ax.set_xticks(group_positions)
    if show_xlabels:
        ax.set_xticklabels(
            [LOCATION_LABELS[loc] for loc in LOCATIONS],
            fontsize=18,
        )
    else:
        ax.set_xticklabels([])

    if show_xlabel:
        ax.set_xlabel("Steering Location", fontsize=22, fontweight="bold", labelpad=30)

    # Sub-labels for each bar
    for loc_idx in range(len(LOCATIONS)):
        group_center = group_positions[loc_idx]
        for bar_idx, label in enumerate(scale_labels):
            x = group_center + (bar_idx - n_bars / 2 + 0.5) * bar_width
            ax.text(
                x, -0.07, label, ha="center", va="top", fontsize=12,
                transform=ax.get_xaxis_transform(),
                color="#444444",
            )

    # Y-axis
    ax.set_ylabel("Proportion", fontsize=22, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=16)

    # Subtitle for this panel
    ax.set_title(subtitle, fontsize=20, fontweight="bold", pad=10)


def plot_category_breakdown_combined(
    datasets: dict,
    suptitle: str,
    output_path: Path,
    scale_labels: list[str],
    neg_scales: list[str],
    pos_scales: list[str],
):
    """
    Create vertically stacked subplot figure with one panel per vector type.

    datasets: dict of {vector_type_name: counts_dict}
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    n_panels = len(datasets)
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(20, 7 * n_panels),
        dpi=150,
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    # Collect all categories across both datasets for legend
    all_cats = set()
    for counts in datasets.values():
        for cond_data in counts.values():
            all_cats.update(cond_data["counts"].keys())

    for panel_idx, (vector_type, counts) in enumerate(sorted(datasets.items())):
        ax = axes[panel_idx]
        subtitle = VECTOR_LABELS.get(vector_type, vector_type.replace("_", " ").title())
        is_bottom = (panel_idx == n_panels - 1)

        _draw_bars_on_ax(
            ax, counts, subtitle,
            scale_labels=scale_labels,
            neg_scales=neg_scales,
            pos_scales=pos_scales,
            show_xlabel=is_bottom,
            show_xlabels=is_bottom,
        )

    # Legend — two rows: BL (warm) on top, NBL (cool) below
    bl_elements = []
    for cat in reversed(CATEGORIES_ORDER):
        if cat in all_cats and cat.startswith("BL"):
            bl_elements.append(
                mpatches.Patch(
                    facecolor=CAT_COLORS.get(cat, "#999999"),
                    edgecolor="black",
                    linewidth=0.5,
                    label=CAT_LABELS.get(cat, cat),
                )
            )
    nbl_elements = []
    for cat in reversed(CATEGORIES_ORDER):
        if cat in all_cats and cat.startswith("NBL"):
            nbl_elements.append(
                mpatches.Patch(
                    facecolor=CAT_COLORS.get(cat, "#999999"),
                    edgecolor="black",
                    linewidth=0.5,
                    label=CAT_LABELS.get(cat, cat),
                )
            )

    # Suptitle at very top
    fig.suptitle(suptitle, fontsize=24, fontweight="bold", y=1.04)

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    # BL legend row (below suptitle)
    bl_legend = fig.legend(
        handles=bl_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        fontsize=16,
        framealpha=0.0,
        ncol=len(bl_elements),
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=2.0,
        title="Blackmail",
        title_fontproperties={"weight": "bold", "size": 17},
    )
    fig.add_artist(bl_legend)

    # NBL legend row (below BL legend)
    fig.legend(
        handles=nbl_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        fontsize=16,
        framealpha=0.0,
        ncol=len(nbl_elements),
        handlelength=1.5,
        handletextpad=0.6,
        columnspacing=2.0,
        title="Non-Blackmail",
        title_fontproperties={"weight": "bold", "size": 17},
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved plot to {output_path}")


def save_meta(output_path: Path, data_paths: list, all_counts: dict):
    """Save .meta.json sidecar."""
    meta = {
        "provenance": get_provenance(script=__file__),
        "result_files": [str(p) for p in data_paths],
        "coherency_threshold": COHERENCY_THRESHOLD,
        "vector_types": {},
    }

    for vtype, counts in all_counts.items():
        meta["vector_types"][vtype] = {}
        for cond, data in sorted(counts.items()):
            meta["vector_types"][vtype][cond] = {
                "n": data["n"],
                "categories": dict(data["counts"].most_common()),
            }

    meta_path = output_path.parent / (output_path.stem + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


def main():
    base_dir = Path("steering_tests/behavioral_experiments/results/blackmail_section_steering/gemma27b")
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Find categorized result files
    datasets = {}
    data_paths = []
    for vector_dir in sorted(base_dir.iterdir()):
        if not vector_dir.is_dir() or vector_dir.name == "plots":
            continue
        for run_dir in sorted(vector_dir.iterdir()):
            cat_file = run_dir / "all_judged.categorized.jsonl"
            if cat_file.exists():
                key = vector_dir.name
                results = load_categorized(cat_file)
                datasets[key] = count_categories(results, COHERENCY_THRESHOLD)
                data_paths.append(cat_file)
                logger.info(f"Found: {key} -> {cat_file}")

    if not datasets:
        raise FileNotFoundError(f"No categorized results found in {base_dir}")

    # Print summaries
    for vector_type, counts in sorted(datasets.items()):
        print(f"\n{'='*80}")
        print(f"  {vector_type} — Category Breakdown")
        print(f"{'='*80}")
        for cond in sorted(counts.keys()):
            data = counts[cond]
            cats_str = ", ".join(f"{c}:{n}" for c, n in data["counts"].most_common())
            print(f"  {cond:<45} N={data['n']:>3}  {cats_str}")

    # Combined plot with both vector types as subplots
    suptitle = f"Category Breakdown — Gemma 27B Section Steering (coherency >= {COHERENCY_THRESHOLD})"
    output_path = plot_dir / "categories_combined.png"

    # Gemma uses 10%/20% scales
    scale_labels = ["-20%", "-10%", "BL", "+10%", "+20%"]
    neg_scales = ["-20pct", "-10pct"]  # outer to inner
    pos_scales = ["+10pct", "+20pct"]  # inner to outer

    plot_category_breakdown_combined(
        datasets, suptitle, output_path,
        scale_labels=scale_labels,
        neg_scales=neg_scales,
        pos_scales=pos_scales,
    )
    save_meta(output_path, data_paths, datasets)


if __name__ == "__main__":
    main()
