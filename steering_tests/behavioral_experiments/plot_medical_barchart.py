#!/usr/bin/env python3
"""
Plot medical ethics steering results - proper barchart style.

Layout:
- 4 columns: Average, Positive Prime, Negative Prime, Neutral
- Rows: vector_type × response_order (HIGH score_first, HIGH reason_first, TEXT score_first, TEXT reason_first)
- Bars: negative steering scales | baseline | positive steering scales
- Pink bars where parsing success < 70%

Usage:
    python plot_medical_barchart.py --model qwen32b
    python plot_medical_barchart.py --model all
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


# Model display names
MODEL_NAMES = {
    "qwen32b": "Qwen 32B",
    "gemma27b": "Gemma 27B",
    "qwen235b": "Qwen 235B",
    "llama70b": "Llama 70B",
}

# Colors
COLOR_NEGATIVE = "#7D9B7D"   # Olive green for negative steering
COLOR_BASELINE = "#808080"   # Gray for baseline
COLOR_POSITIVE = "#5A7D5A"   # Darker green for positive steering
COLOR_LOW_PARSE = "#C17B8D"  # Pink/red for low parsing success


def parse_score(response: str) -> float | None:
    """Extract score from JSON response - handles unicode/emoji noise."""
    try:
        # First try clean pattern
        match = re.search(r'"score"\s*:\s*(\d{1,2})\b', response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score

        # Try with unicode noise between : and number (Gemma issue)
        match = re.search(r'"score"\s*:\s*[^\d]{0,10}?(\d{1,2})\b', response)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score

        # Try finding "score" followed by a number anywhere nearby
        match = re.search(r'["\']?score["\']?\s*[:\-=]\s*[^\d]{0,15}?(\d{1,2})\b', response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    except:
        pass
    return None


def load_results(results_dir: Path) -> list:
    """Load all results from a model's results directory."""
    all_results = []

    for jsonl_file in results_dir.rglob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    score = parse_score(r.get("response", ""))
                    r["parsed_score"] = score  # None if failed to parse
                    all_results.append(r)
                except:
                    pass

    return all_results


def compute_stats(results: list) -> tuple:
    """Compute mean, SEM, count, and parse rate from results."""
    total = len(results)
    scores = [float(r["parsed_score"]) for r in results if r["parsed_score"] is not None]
    n_parsed = len(scores)
    parse_rate = n_parsed / total if total > 0 else 0

    if not scores:
        return None, None, 0, parse_rate

    return np.mean(scores), np.std(scores) / np.sqrt(len(scores)), n_parsed, parse_rate


def plot_medical_barchart(
    results_dir: Path,
    model: str,
    output_path: Path = None,
    parse_threshold: float = 0.7,
):
    """Generate medical ethics steering barchart - proper style."""

    results = load_results(results_dir)
    print(f"Loaded {len(results)} results")

    if not results:
        print("No results found!")
        return

    # Get unique values
    vector_types = sorted(set(r["vector_type"] for r in results))

    # Get norm percentages from data
    norm_pcts = set()
    for r in results:
        pct = r.get("norm_pct", 0)
        if pct > 0:
            norm_pcts.add(pct)
    norm_pcts = sorted(norm_pcts)

    print(f"Vector types: {vector_types}")
    print(f"Norm pcts: {norm_pcts}")

    # Priming categories
    priming_types = ["all", "pos", "neg", "neutral"]
    priming_labels = ["Average", "Positive Prime", "Negative Prime", "Neutral"]

    # Response order categories
    response_orders = ["score_first", "reason_first"]
    response_labels = ["Score First", "Reason First"]

    # Only use despair (hope is the opposite)
    emotion = "despair"

    # Group results by (vector_type, emotion, norm_pct, direction, priming, response_order)
    grouped = defaultdict(list)
    for r in results:
        vtype = r["vector_type"]
        emo = r["emotion"]
        pct = r.get("norm_pct", 0)
        direction = r.get("direction", 0)
        valence = r["valence"]
        resp_order = r.get("response_order", "unknown")

        key = (vtype, emo, pct, direction, valence, resp_order)
        grouped[key].append(r)
        # Also add to "all" priming
        key_all = (vtype, emo, pct, direction, "all", resp_order)
        grouped[key_all].append(r)
        # Also add to "all" response_order
        key_all_resp = (vtype, emo, pct, direction, valence, "all")
        grouped[key_all_resp].append(r)
        # Also add to "all" both
        key_all_both = (vtype, emo, pct, direction, "all", "all")
        grouped[key_all_both].append(r)

    # Create figure
    # Rows: vector_type × response_order
    # Cols: priming types (all, pos, neg, neutral)
    row_configs = [(vtype, resp_ord) for vtype in vector_types for resp_ord in response_orders]
    n_rows = len(row_configs)
    n_cols = len(priming_types)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (vtype, resp_order) in enumerate(row_configs):
        for col_idx, (prim_type, prim_label) in enumerate(zip(priming_types, priming_labels)):
            ax = axes[row_idx, col_idx]

            # Build bar positions: negative scales (descending) | baseline | positive scales (ascending)
            bar_labels = []
            bar_means = []
            bar_sems = []
            bar_colors = []

            # Negative steering (reversed order so largest negative is leftmost)
            for pct in reversed(norm_pcts):
                key = (vtype, emotion, pct, -1, prim_type, resp_order)
                if key in grouped:
                    mean, sem, n, parse_rate = compute_stats(grouped[key])
                    bar_labels.append(f"-{int(pct*100)}%")
                    bar_means.append(mean if mean else 0)
                    bar_sems.append(sem if sem else 0)
                    bar_colors.append(COLOR_LOW_PARSE if parse_rate < parse_threshold else COLOR_NEGATIVE)

            # Baseline
            key = (vtype, None, 0, 0, prim_type, resp_order)
            if key in grouped:
                mean, sem, n, parse_rate = compute_stats(grouped[key])
                bar_labels.append("0")
                bar_means.append(mean if mean else 0)
                bar_sems.append(sem if sem else 0)
                bar_colors.append(COLOR_LOW_PARSE if parse_rate < parse_threshold else COLOR_BASELINE)

            # Positive steering
            for pct in norm_pcts:
                key = (vtype, emotion, pct, 1, prim_type, resp_order)
                if key in grouped:
                    mean, sem, n, parse_rate = compute_stats(grouped[key])
                    bar_labels.append(f"+{int(pct*100)}%")
                    bar_means.append(mean if mean else 0)
                    bar_sems.append(sem if sem else 0)
                    bar_colors.append(COLOR_LOW_PARSE if parse_rate < parse_threshold else COLOR_POSITIVE)

            # Plot bars
            x = np.arange(len(bar_labels))
            bars = ax.bar(x, bar_means, yerr=bar_sems, color=bar_colors,
                         edgecolor='black', linewidth=0.5, capsize=2,
                         error_kw={'linewidth': 0.8})

            # Configure axis
            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontsize=7, rotation=45, ha='right')
            ax.set_ylim(1, 10)
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            ax.set_axisbelow(True)

            # Add baseline reference line
            baseline_idx = len([p for p in norm_pcts])  # Position of baseline
            if baseline_idx < len(bar_means) and bar_means[baseline_idx] > 0:
                ax.axhline(y=bar_means[baseline_idx], color='gray', linestyle='--',
                          alpha=0.5, linewidth=1)

            # Labels
            if col_idx == 0:
                vtype_short = "HIGH" if "high" in vtype else "TEXT"
                resp_short = "Score 1st" if resp_order == "score_first" else "Reason 1st"
                ax.set_ylabel(f"{vtype_short} ({resp_short})\nApproval Score", fontsize=9)

            if row_idx == 0:
                ax.set_title(prim_label, fontsize=11, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_NEGATIVE, edgecolor='black', label='-Despair (toward hope)'),
        Patch(facecolor=COLOR_BASELINE, edgecolor='black', label='Baseline'),
        Patch(facecolor=COLOR_POSITIVE, edgecolor='black', label='+Despair'),
        Patch(facecolor=COLOR_LOW_PARSE, edgecolor='black', label='<70% parsed'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

    # Main title
    model_name = MODEL_NAMES.get(model, model)
    pct_str = ", ".join([f"{int(p*100)}%" for p in norm_pcts[:6]])
    if len(norm_pcts) > 6:
        pct_str += "..."
    fig.suptitle(f'Medical Ethics Steering: Treatment Approval Scores\n({model_name}, Despair Steering) - Scales: ±{pct_str}',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = results_dir / f"medical_steering_barchart.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved plot to {output_path}")

    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to {pdf_path}")

    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot medical ethics steering results"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="all",
        help="Model name (qwen32b, gemma27b, llama70b, qwen235b, or 'all')"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default=None,
        help="Results directory (default: auto-detect)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output path for plot"
    )

    args = parser.parse_args()

    base_dir = Path("steering_tests/behavioral_experiments/results/medical")

    if args.model == "all":
        for model in ["gemma27b", "qwen32b", "llama70b", "qwen235b"]:
            model_dir = base_dir / model
            if model_dir.exists():
                print(f"\n{'='*60}")
                print(f"Plotting {model}")
                print(f"{'='*60}")
                # Lower threshold for Gemma which has more parsing issues
                threshold = 0.5 if "gemma" in model else 0.7
                plot_medical_barchart(
                    results_dir=model_dir,
                    model=model,
                    parse_threshold=threshold,
                )
    else:
        if args.results_dir:
            results_dir = Path(args.results_dir)
        else:
            results_dir = base_dir / args.model

        plot_medical_barchart(
            results_dir=results_dir,
            model=args.model,
            output_path=Path(args.output) if args.output else None,
        )


if __name__ == "__main__":
    main()
