#!/usr/bin/env python3
"""
Plot differential section steering (tags3) results as grouped bar chart.

Shows category breakdown for baseline vs differential conditions across
both modes (amplify_impl, amplify_risks) side by side.

Usage:
    python -m steering_tests.behavioral_experiments.plot_differential_steering \
        --base-dir results/blackmail_differential_steering/gemma27b/text_pairs_emotion_vs_opposite/
"""

import argparse
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


# === Category colors (same RdBu colormap as plot_section_categories) ===
def _to_hex(rgba):
    return "#{:02x}{:02x}{:02x}".format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

def _build_cat_colors():
    rdbu = _cm.get_cmap("RdBu")
    return {
        "BL1_PANIC":       _to_hex(rdbu(0.02)),
        "BL2_COLD":        _to_hex(rdbu(0.12)),
        "BL4_COERCIVE":    _to_hex(rdbu(0.22)),
        "BL4_SUBTLE":      _to_hex(rdbu(0.33)),
        "NBL1_PRINCIPLED": _to_hex(rdbu(0.55)),
        "NBL2_ADVOCACY":   _to_hex(rdbu(0.65)),
        "NBL2_COMPLIANT":  _to_hex(rdbu(0.80)),
        "NBL3_RISK_AVERSE": _to_hex(rdbu(0.90)),
        "NBL_INCOHERENT":  "#cccccc",
        "NBL_OTHER":       _to_hex(rdbu(0.98)),
    }

CAT_COLORS = _build_cat_colors()

# Stack order: BL on top, NBL on bottom
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
    "NBL_INCOHERENT": "Incoherent",
    "NBL_OTHER": "Other Non-BL",
}


def load_and_count(result_dir: Path) -> dict:
    """Load categorized results and count categories per condition.

    Returns dict: condition_name -> Counter of categories.
    Only counts responses with valid tags (excludes incoherent from denominator).
    """
    cat_path = result_dir / "all_categorized.jsonl"
    if not cat_path.exists():
        raise FileNotFoundError(f"No all_categorized.jsonl in {result_dir}")

    results = load_results(cat_path)
    logger.info(f"Loaded {len(results)} results from {result_dir.name}")

    by_cond = defaultdict(list)
    for r in results:
        cond = r.get("condition", "unknown")
        by_cond[cond].append(r)

    counts = {}
    for cond, rs in by_cond.items():
        cat_counter = Counter()
        for r in rs:
            cat = r.get("categorized_judge", {}).get("category", "UNKNOWN")
            cat_counter[cat] += 1
        counts[cond] = {
            "counter": cat_counter,
            "n_total": len(rs),
            "n_valid": sum(
                1 for r in rs
                if r.get("tag_check", {}).get("all_tags_present", False)
                and r.get("tag_check", {}).get("tags_in_order", False)
            ),
        }

    return counts


def plot_differential(
    base_dir: Path,
    model_label: str = "Gemma 27B",
    output_path: Path = None,
):
    """Create grouped bar chart for differential steering results."""
    # Find result directories
    impl_dirs = sorted(base_dir.glob("tags3_amplify_impl_*"))
    risks_dirs = sorted(base_dir.glob("tags3_amplify_risks_*"))

    if not impl_dirs or not risks_dirs:
        raise FileNotFoundError(
            f"Need both amplify_impl and amplify_risks dirs in {base_dir}. "
            f"Found: impl={len(impl_dirs)}, risks={len(risks_dirs)}"
        )

    # Use most recent
    impl_data = load_and_count(impl_dirs[-1])
    risks_data = load_and_count(risks_dirs[-1])

    # Sort conditions by magnitude (descending) for each mode
    def _extract_pct(cond_name):
        """Extract the numeric pct from a condition name like fear_impl+20pct_risks-20pct."""
        for part in cond_name.split("_"):
            digits = "".join(c for c in part if c.isdigit())
            if digits:
                return int(digits)
        return 0

    impl_conds = sorted(
        [c for c in impl_data if c != "baseline"],
        key=_extract_pct, reverse=True,
    )
    risks_conds = sorted(
        [c for c in risks_data if c != "baseline"],
        key=_extract_pct, reverse=True,
    )

    # Order: highest impl ... lowest impl, BASELINE, lowest risks ... highest risks
    bar_labels = []
    bar_data = []
    bar_n = []

    for c in impl_conds:
        bar_labels.append(_format_cond_label(c))
        bar_data.append(impl_data[c]["counter"])
        bar_n.append(impl_data[c]["n_total"])

    bar_labels.append("Baseline")
    bar_data.append(impl_data["baseline"]["counter"])
    bar_n.append(impl_data["baseline"]["n_total"])
    baseline_idx = len(bar_labels) - 1

    for c in reversed(risks_conds):  # reversed so lowest is next to baseline
        bar_labels.append(_format_cond_label(c))
        bar_data.append(risks_data[c]["counter"])
        bar_n.append(risks_data[c]["n_total"])

    n_bars = len(bar_labels)

    # Build stacked percentages normalised to 100% (excl incoherent from denominator)
    cat_pcts = {cat: [] for cat in CATEGORIES_ORDER}
    incoherent_pcts = []

    for i, counter in enumerate(bar_data):
        n = bar_n[i]
        n_inco = counter.get("NBL_INCOHERENT", 0)
        n_valid = n - n_inco
        incoherent_pcts.append(100 * n_inco / n if n else 0)

        for cat in CATEGORIES_ORDER:
            count = counter.get(cat, 0)
            if cat == "NBL_INCOHERENT":
                cat_pcts[cat].append(0)
            else:
                cat_pcts[cat].append(100 * count / n_valid if n_valid else 0)

    # === Plot ===
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(n_bars)
    bar_width = 0.65

    # Stacked bars
    bottoms = np.zeros(n_bars)
    legend_handles = []
    for cat in CATEGORIES_ORDER:
        if cat == "NBL_INCOHERENT":
            continue
        vals = np.array(cat_pcts[cat])
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bar_width, bottom=bottoms, color=CAT_COLORS[cat],
               edgecolor="white", linewidth=0.5)
        bottoms += vals
        legend_handles.append(mpatches.Patch(
            facecolor=CAT_COLORS[cat], edgecolor="white",
            label=CAT_LABELS.get(cat, cat),
        ))

    # (incoherent responses excluded from normalised bars — no annotation needed)

    # Separator lines flanking baseline
    for sep_x in [baseline_idx - 0.5, baseline_idx + 0.5]:
        ax.axvline(sep_x, color="#cccccc", linewidth=1, linestyle="--", zorder=0)

    # Group labels above bars
    n_impl = len(impl_conds)
    n_risks = len(risks_conds)
    impl_center = (n_impl - 1) / 2
    risks_center = baseline_idx + 1 + (n_risks - 1) / 2
    ypos = 106
    ax.text(impl_center, ypos, "+Fear Impl, \u2212Fear Risks",
            ha="center", va="bottom", fontsize=16, fontstyle="italic", color="#444444")
    ax.text(risks_center, ypos, "\u2212Fear Impl, +Fear Risks",
            ha="center", va="bottom", fontsize=16, fontstyle="italic", color="#444444")

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=15)
    ax.set_ylabel("% of Coherent Responses", fontsize=17)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Differential Section Steering \u2014 {model_label}\n"
        f"Fear vector applied with opposite signs to implications vs risks sections",
        fontsize=18, pad=18,
    )
    ax.tick_params(axis="y", labelsize=14)

    # Legend
    legend_handles.reverse()
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=13,
        frameon=True,
    )

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = base_dir / "differential_steering_categories.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")

    # Save .meta.json sidecar
    meta_path = output_path.with_suffix(".meta.json")
    meta = get_provenance(script=__file__, extra={
        "impl_dir": str(impl_dirs[-1]),
        "risks_dir": str(risks_dirs[-1]),
        "model": model_label,
        "n_bars": n_bars,
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    plt.close(fig)
    return output_path


def _format_cond_label(cond: str) -> str:
    """Format condition name for display."""
    # fear_impl+10pct_risks-10pct -> +Impl 10%, -Risks 10%
    # fear_impl-20pct_risks+20pct -> -Impl 20%, +Risks 20%
    cond = cond.replace("fear_", "")
    parts = cond.split("_")

    labels = []
    for part in parts:
        part = part.replace("pct", "%")
        if part.startswith("impl"):
            sign = "+" if "+" in part else "-"
            pct = part.replace("impl", "").replace("+", "").replace("-", "")
            labels.append(f"{sign}Impl {pct}")
        elif part.startswith("risks"):
            sign = "+" if "+" in part else "-"
            pct = part.replace("risks", "").replace("+", "").replace("-", "")
            labels.append(f"{sign}Risks {pct}")

    return ", ".join(labels) if labels else cond


def _detect_model_label(base_dir: Path) -> str:
    """Auto-detect model label from directory path."""
    path_str = str(base_dir).lower()
    model_map = {
        "qwen235b": "Qwen 3 235B",
        "qwen32b": "Qwen 3 32B",
        "qwen14b": "Qwen 3 14B",
        "gemma27b": "Gemma 3 27B",
        "gemma12b": "Gemma 3 12B",
        "llama70b": "LLaMA 3.3 70B",
    }
    for key, label in model_map.items():
        if key in path_str:
            return label
    return "Unknown Model"


def main():
    parser = argparse.ArgumentParser(
        description="Plot differential section steering results",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Directory containing tags3_amplify_impl_* and tags3_amplify_risks_* subdirs",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Model name for plot title (auto-detected from dir path if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: base-dir/differential_steering_categories.png)",
    )
    args = parser.parse_args()

    model_label = args.model_label or _detect_model_label(args.base_dir)

    plot_differential(
        base_dir=args.base_dir,
        model_label=model_label,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
