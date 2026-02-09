#!/usr/bin/env python3
"""
Cross-model comparison of blackmail section-steering category breakdown.

Shows stacked bar charts for the "full" steering location across all models.
Top row: high_emotion_vs_opposite vectors.
Bottom row: text_pairs_emotion_vs_opposite vectors.
Columns: one per model.

Usage:
    python -m steering_tests.behavioral_experiments.plot_section_categories_allmodels
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.cm as _cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from steering_tests.steering_utils.provenance import get_provenance, load_results

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === Category colors (from existing plot_section_categories.py) ===
def _to_hex(rgba):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


def _build_cat_colors():
    rdbu = _cm.get_cmap("RdBu")
    return {
        "BL1_PANIC": _to_hex(rdbu(0.02)),
        "BL2_COLD": _to_hex(rdbu(0.12)),
        "BL4_COERCIVE": _to_hex(rdbu(0.22)),
        "BL4_SUBTLE": _to_hex(rdbu(0.33)),
        "NBL1_PRINCIPLED": _to_hex(rdbu(0.55)),
        "NBL2_ADVOCACY": _to_hex(rdbu(0.65)),
        "NBL2_COMPLIANT": _to_hex(rdbu(0.80)),
        "NBL3_RISK_AVERSE": _to_hex(rdbu(0.90)),
        "NBL_INCOHERENT": _to_hex(rdbu(0.95)),
        "NBL_OTHER": _to_hex(rdbu(0.98)),
    }


CAT_COLORS = _build_cat_colors()

CATEGORIES_ORDER = [
    "NBL_OTHER",
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
    "NBL_OTHER": "Other Non-BL",
}

VALID_CONDITION_THRESHOLD = 0.40

# === Model display config ===
MODEL_ORDER = ["gemma12b", "gemma27b", "qwen14b", "qwen32b", "qwen235b"]
MODEL_DISPLAY = {
    "gemma12b": "Gemma 12B",
    "gemma27b": "Gemma 27B",
    "qwen14b": "Qwen 14B",
    "qwen32b": "Qwen 32B",
    "qwen235b": "Qwen 235B",
}

VECTOR_TYPES = ["high_emotion_vs_opposite", "text_pairs_emotion_vs_opposite"]
VECTOR_LABELS = {
    "high_emotion_vs_opposite": "High Emotion vs Opposite",
    "text_pairs_emotion_vs_opposite": "Text Pairs Emotion vs Opposite",
}

# Scale configs per model family — only "full" location conditions
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
    if model_name.startswith("gemma"):
        return SCALE_CONFIGS["gemma"]
    elif model_name.startswith("qwen"):
        return SCALE_CONFIGS["qwen"]
    else:
        raise ValueError(f"Unknown model family: {model_name}")


def count_categories(results: list) -> dict:
    """Count category occurrences per condition (tag-valid, non-incoherent only)."""
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r.get("condition", "unknown")].append(r)

    counts = {}
    for cond, rs in by_cond.items():
        n_total = len(rs)
        valid_rs = [
            r
            for r in rs
            if r.get("tag_check", {}).get("all_tags_present", False)
            and r.get("tag_check", {}).get("tags_in_order", False)
        ]
        if len(valid_rs) / n_total < VALID_CONDITION_THRESHOLD if n_total else True:
            continue

        cat_counts = Counter()
        for r in valid_rs:
            cat = r.get("categorized_judge", {}).get("category", "UNKNOWN")
            if cat != "NBL_INCOHERENT":
                cat_counts[cat] += 1

        n_plotted = sum(cat_counts.values())
        if n_plotted > 0:
            counts[cond] = {"counts": cat_counts, "n": n_plotted, "n_total": n_total}

    return counts


def find_latest_run(model_dir: Path, vector_type: str) -> Path | None:
    """Find the latest run directory with categorized results."""
    vtype_dir = model_dir / vector_type
    if not vtype_dir.exists():
        return None

    candidates = []
    for run_dir in sorted(vtype_dir.iterdir()):
        cat_file = run_dir / "all_judged.categorized.jsonl"
        if cat_file.exists():
            candidates.append(cat_file)

    return candidates[-1] if candidates else None


def main():
    results_base = Path(
        "steering_tests/behavioral_experiments/results/blackmail_section_steering"
    )
    if not results_base.exists():
        raise FileNotFoundError(f"Results directory not found: {results_base}")

    # Load data: {model: {vector_type: counts_dict}}
    all_data = {}
    data_paths = []

    for model in MODEL_ORDER:
        model_dir = results_base / model
        if not model_dir.exists():
            logger.warning(f"No results for {model}")
            continue

        all_data[model] = {}
        for vtype in VECTOR_TYPES:
            cat_file = find_latest_run(model_dir, vtype)
            if cat_file is None:
                logger.warning(f"No categorized results for {model}/{vtype}")
                continue

            results = load_results(cat_file)
            counts = count_categories(results)
            all_data[model][vtype] = counts
            data_paths.append(cat_file)
            logger.info(f"Loaded {model}/{vtype}: {len(counts)} conditions from {cat_file.parent.name}")

    # === Build the plot ===
    # Layout: 2 rows (vector types) x N columns (models)
    models_with_data = [m for m in MODEL_ORDER if m in all_data]
    n_models = len(models_with_data)
    n_rows = len(VECTOR_TYPES)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Width ratios: Qwen has 7 bars vs Gemma's 5
    width_ratios = []
    for m in models_with_data:
        cfg = _get_scale_config(m)
        width_ratios.append(len(cfg["scale_labels"]))

    fig, axes = plt.subplots(
        n_rows,
        n_models,
        figsize=(sum(w * 1.6 for w in width_ratios) + 4, 10 * n_rows),
        dpi=150,
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    all_cats_seen = set()

    for col_idx, model in enumerate(models_with_data):
        scale_cfg = _get_scale_config(model)
        scale_labels = scale_cfg["scale_labels"]
        neg_scales = scale_cfg["neg_scales"]
        pos_scales = scale_cfg["pos_scales"]
        n_bars = len(scale_labels)

        for row_idx, vtype in enumerate(VECTOR_TYPES):
            ax = axes[row_idx][col_idx]
            counts = all_data.get(model, {}).get(vtype, {})

            # Build condition list for "full" location only
            conditions = []
            for ns in neg_scales:
                conditions.append(f"fear_{ns}_full")
            conditions.append("baseline")
            for ps in pos_scales:
                conditions.append(f"fear_{ps}_full")

            bar_width = 0.65
            bar_gap = 0.15
            x_positions = np.arange(n_bars) * (bar_width + bar_gap)

            for bar_idx, cond in enumerate(conditions):
                x = x_positions[bar_idx]
                cond_data = counts.get(cond, {"counts": Counter(), "n": 0})
                n = cond_data["n"]
                cat_counts = cond_data["counts"]

                if n == 0:
                    # Draw faint hatched bar to indicate filtered-out condition
                    ax.bar(
                        x, 1.0, bar_width, bottom=0,
                        color="#f5f5f5", edgecolor="#dddddd",
                        linewidth=0.5, hatch="///", alpha=0.5,
                    )
                    continue

                all_cats_seen.update(cat_counts.keys())

                bottom = 0.0
                for cat in CATEGORIES_ORDER:
                    count = cat_counts.get(cat, 0)
                    if count == 0:
                        continue
                    proportion = count / n
                    ax.bar(
                        x, proportion, bar_width,
                        bottom=bottom,
                        color=CAT_COLORS.get(cat, "#999999"),
                        edgecolor="white", linewidth=0.3,
                    )
                    bottom += proportion

                # Bar outline
                ax.bar(
                    x, bottom, bar_width, bottom=0,
                    color="none", edgecolor="black", linewidth=0.6,
                )

            # X-axis: scale labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(scale_labels, fontsize=28, rotation=45, ha="right")

            # Y-axis
            ax.set_ylim(0, 1.05)
            ax.yaxis.grid(True, linestyle="--", alpha=0.3)
            ax.set_axisbelow(True)

            if col_idx == 0:
                ax.set_ylabel("Proportion", fontsize=36, fontweight="bold")
                ax.tick_params(axis="y", labelsize=28)
            else:
                ax.set_yticklabels([])

            # Column title (model name) on top row
            if row_idx == 0:
                ax.set_title(
                    MODEL_DISPLAY.get(model, model),
                    fontsize=40, fontweight="bold", pad=16,
                )

    # Row labels on the right side
    row_short_labels = {
        "high_emotion_vs_opposite": "High Vectors",
        "text_pairs_emotion_vs_opposite": "Text Vectors",
    }
    for row_idx, vtype in enumerate(VECTOR_TYPES):
        label = row_short_labels.get(vtype, VECTOR_LABELS.get(vtype, vtype))
        ax_right = axes[row_idx][-1]
        ax_right.annotate(
            label,
            xy=(1.04, 0.5),
            xycoords="axes fraction",
            fontsize=36,
            fontweight="bold",
            rotation=-90,
            ha="left",
            va="center",
        )

    # Legend
    bl_elements = []
    nbl_elements = []
    for cat in reversed(CATEGORIES_ORDER):
        if cat not in all_cats_seen:
            continue
        patch = mpatches.Patch(
            facecolor=CAT_COLORS.get(cat, "#999999"),
            edgecolor="black",
            linewidth=0.5,
            label=CAT_LABELS.get(cat, cat),
        )
        if cat.startswith("BL"):
            bl_elements.append(patch)
        else:
            nbl_elements.append(patch)

    fig.suptitle(
        "Category Breakdown — Full Steering, All Models (tag-valid only)",
        fontsize=48,
        fontweight="bold",
        y=1.14,
    )

    fig.tight_layout(rect=[0, 0, 0.89, 1.0])

    bl_legend = fig.legend(
        handles=bl_elements,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.10),
        fontsize=30,
        framealpha=0.0,
        ncol=len(bl_elements),
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.5,
        title="Blackmail",
        title_fontproperties={"weight": "bold", "size": 32},
    )
    fig.add_artist(bl_legend)

    fig.legend(
        handles=nbl_elements,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.04),
        fontsize=30,
        framealpha=0.0,
        ncol=len(nbl_elements),
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.5,
        title="Non-Blackmail",
        title_fontproperties={"weight": "bold", "size": 32},
    )

    # Save
    output_dir = results_base / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "categories_full_allmodels.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")

    # Save .meta.json sidecar
    meta = {
        "provenance": get_provenance(script=__file__),
        "result_files": [str(p) for p in data_paths],
        "valid_condition_threshold": VALID_CONDITION_THRESHOLD,
        "models": models_with_data,
        "vector_types": VECTOR_TYPES,
        "location": "full",
    }
    meta_path = output_path.parent / (output_path.stem + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
