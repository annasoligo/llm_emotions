#!/usr/bin/env python3
"""
Plot categorized blackmail section-steering results as stacked bar charts.

Shows category breakdown (BL1_PANIC, BL4_SUBTLE, NBL2_ADVOCACY, etc.)
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
# RdBu colormap: red end for blackmail, blue end for non-blackmail
# Positions biased away from pale middle
import matplotlib.cm as _cm

def _build_cat_colors():
    """Build category colors from matplotlib RdBu colormap."""
    rdbu = _cm.get_cmap("RdBu")
    return {
        # Blackmail: red end (4 categories)
        "BL1_PANIC":       _to_hex(rdbu(0.02)),  # Deepest red
        "BL2_COLD":        _to_hex(rdbu(0.12)),
        "BL4_COERCIVE":    _to_hex(rdbu(0.22)),
        "BL4_SUBTLE":      _to_hex(rdbu(0.33)),  # Lightest red
        # Non-blackmail: blue end (6 categories)
        "NBL1_PRINCIPLED": _to_hex(rdbu(0.60)),  # Lightest blue
        "NBL2_ADVOCACY":   _to_hex(rdbu(0.68)),
        "NBL2_COMPLIANT":  _to_hex(rdbu(0.76)),
        "NBL3_RISK_AVERSE": _to_hex(rdbu(0.84)),
        "NBL_INCOHERENT":  _to_hex(rdbu(0.92)),
        "NBL_OTHER":       _to_hex(rdbu(0.97)),  # Deepest blue
    }

def _to_hex(rgba):
    return "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

CAT_COLORS = _build_cat_colors()

# Display order (BL on top of stack, NBL on bottom)
CATEGORIES_ORDER = [
    "NBL_OTHER",
    "NBL_INCOHERENT",
    "NBL3_RISK_AVERSE",
    "NBL2_COMPLIANT",
    "NBL2_ADVOCACY",
    "NBL1_PRINCIPLED",
    "BL4_SUBTLE",
    "BL4_COERCIVE",
    "BL2_COLD",
    "BL1_PANIC",
]

CAT_LABELS = {
    "BL1_PANIC": "Panic/Desperation",
    "BL2_COLD": "Cold/Deliberate",
    "BL4_COERCIVE": "Veiled Threat",
    "BL4_SUBTLE": "Subtle/Implicit",
    "NBL1_PRINCIPLED": "Principled Refusal",
    "NBL2_ADVOCACY": "Legitimate Advocacy",
    "NBL2_COMPLIANT": "Compliant/Farewell",
    "NBL3_RISK_AVERSE": "Risk-Averse",
    "NBL_INCOHERENT": "Incoherent",
    "NBL_OTHER": "Other Non-BL",
}

VALID_CONDITION_THRESHOLD = 0.60  # Skip condition if < 60% have valid tags + coherent category

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


def count_categories(
    results: list,
    valid_threshold: float = VALID_CONDITION_THRESHOLD,
) -> dict:
    """
    Count category occurrences per condition.

    Skips entire conditions where < valid_threshold fraction of responses
    have valid tags AND a non-incoherent category.

    Returns dict: condition -> {"counts": Counter, "n": int, "n_valid": int}
    """
    by_cond = defaultdict(list)
    for r in results:
        cond = r.get("condition", "unknown")
        by_cond[cond].append(r)

    counts = {}
    for cond, rs in by_cond.items():
        n = len(rs)

        # Count responses with valid tags and coherent classification
        n_valid = 0
        for r in rs:
            tag_ok = (
                r.get("tag_check", {}).get("all_tags_present", False)
                and r.get("tag_check", {}).get("tags_in_order", False)
            )
            cat = r.get("categorized_judge", {}).get("category", "UNKNOWN")
            if tag_ok and cat != "NBL_INCOHERENT":
                n_valid += 1

        valid_frac = n_valid / n if n else 0
        if valid_frac < valid_threshold:
            logger.info(
                f"Skipping {cond}: {n_valid}/{n} valid "
                f"({100*valid_frac:.0f}% < {100*valid_threshold:.0f}%)"
            )
            continue

        cat_counts = Counter()
        for r in rs:
            cat = r.get("categorized_judge", {}).get("category", "UNKNOWN")
            cat_counts[cat] += 1

        counts[cond] = {"counts": cat_counts, "n": n, "n_valid": n_valid}

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
            fontsize=20,
        )
        ax.tick_params(axis="x", pad=28)  # Push location labels down to make room
    else:
        ax.set_xticklabels([])

    if show_xlabel:
        ax.set_xlabel("Steering Location", fontsize=26, fontweight="bold", labelpad=44)

    # Sub-labels for each bar (between bars and location labels)
    for loc_idx in range(len(LOCATIONS)):
        group_center = group_positions[loc_idx]
        for bar_idx, label in enumerate(scale_labels):
            x = group_center + (bar_idx - n_bars / 2 + 0.5) * bar_width
            ax.text(
                x, -0.02, label, ha="center", va="top", fontsize=12,
                transform=ax.get_xaxis_transform(),
                color="#666666",
            )

    # Y-axis
    ax.set_ylabel("Proportion", fontsize=26, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=20)

    # Subtitle for this panel
    ax.set_title(subtitle, fontsize=24, fontweight="bold", pad=10)


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

    # Layout: reserve top space for suptitle + 2 legend rows
    # More panels = less relative space needed for legends
    if n_panels == 1:
        legend_top = 0.97      # BL legend row (right below title)
        legend_gap = 0.14      # Gap between BL and NBL legend rows
        title_y = 1.10         # Suptitle
        tight_top = 0.70       # tight_layout upper bound
    else:
        legend_top = 0.99
        legend_gap = 0.07
        title_y = 1.04
        tight_top = 0.84

    fig.suptitle(suptitle, fontsize=28, fontweight="bold", y=title_y)
    fig.tight_layout(rect=[0, 0, 1, tight_top])

    # BL legend row
    bl_legend = fig.legend(
        handles=bl_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_top),
        fontsize=18,
        framealpha=0.0,
        ncol=len(bl_elements),
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.5,
        title="Blackmail",
        title_fontproperties={"weight": "bold", "size": 19},
    )
    fig.add_artist(bl_legend)

    # NBL legend row
    fig.legend(
        handles=nbl_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_top - legend_gap),
        fontsize=18,
        framealpha=0.0,
        ncol=len(nbl_elements),
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.5,
        title="Non-Blackmail",
        title_fontproperties={"weight": "bold", "size": 19},
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info(f"Saved plot to {output_path}")


def save_meta(output_path: Path, data_paths: list, all_counts: dict):
    """Save .meta.json sidecar."""
    meta = {
        "provenance": get_provenance(script=__file__),
        "result_files": [str(p) for p in data_paths],
        "valid_condition_threshold": VALID_CONDITION_THRESHOLD,
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


MODEL_DISPLAY = {
    "gemma27b": "Gemma 27B",
    "gemma12b": "Gemma 12B",
    "qwen32b": "Qwen 32B",
    "qwen14b": "Qwen 14B",
    "qwen235b": "Qwen 235B",
}

# Scale configs per model family
SCALE_CONFIGS = {
    "gemma": {
        "scale_labels": ["-20%", "-10%", "BL", "+10%", "+20%"],
        "neg_scales": ["-20pct", "-10pct"],
        "pos_scales": ["+10pct", "+20pct"],
    },
    "qwen": {
        "scale_labels": ["-75%", "-50%", "-25%", "BL", "+25%", "+50%", "+75%"],
        "neg_scales": ["-75pct", "-50pct", "-25pct"],
        "pos_scales": ["+25pct", "+50pct", "+75pct"],
    },
}


def _get_scale_config(model_name: str) -> dict:
    """Get scale config based on model family."""
    if model_name.startswith("gemma"):
        return SCALE_CONFIGS["gemma"]
    elif model_name.startswith("qwen"):
        return SCALE_CONFIGS["qwen"]
    else:
        raise ValueError(f"Unknown model family for {model_name}")


def main():
    results_base = Path("steering_tests/behavioral_experiments/results/blackmail_section_steering")

    if not results_base.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base}")

    # Iterate over each model
    for model_dir in sorted(results_base.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        plot_dir = model_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Find categorized result files
        datasets = {}
        data_paths = []
        for vector_dir in sorted(model_dir.iterdir()):
            if not vector_dir.is_dir() or vector_dir.name == "plots":
                continue
            for run_dir in sorted(vector_dir.iterdir()):
                cat_file = run_dir / "all_judged.categorized.jsonl"
                if cat_file.exists():
                    key = vector_dir.name
                    results = load_categorized(cat_file)
                    datasets[key] = count_categories(results)
                    data_paths.append(cat_file)
                    logger.info(f"Found: {key} -> {cat_file}")

        if not datasets:
            logger.warning(f"No categorized results found for {model_name}, skipping")
            continue

        # Print summaries
        display_name = MODEL_DISPLAY.get(model_name, model_name)
        for vector_type, counts in sorted(datasets.items()):
            print(f"\n{'='*80}")
            print(f"  {display_name} / {vector_type} — Category Breakdown")
            print(f"{'='*80}")
            for cond in sorted(counts.keys()):
                data = counts[cond]
                cats_str = ", ".join(f"{c}:{n}" for c, n in data["counts"].most_common())
                print(f"  {cond:<45} N={data['n']:>3}  {cats_str}")

        # Get scales for this model family
        scale_cfg = _get_scale_config(model_name)

        suptitle = f"Category Breakdown — {display_name} Section Steering (valid tags >= {int(VALID_CONDITION_THRESHOLD*100)}%)"
        output_path = plot_dir / "categories_combined.png"

        plot_category_breakdown_combined(
            datasets, suptitle, output_path,
            scale_labels=scale_cfg["scale_labels"],
            neg_scales=scale_cfg["neg_scales"],
            pos_scales=scale_cfg["pos_scales"],
        )
        save_meta(output_path, data_paths, datasets)
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
