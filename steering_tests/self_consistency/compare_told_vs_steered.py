"""
Compare told (API) vs steered (vLLM) self-consistency results.

Loads judged results from both methods and computes:
1. Shift direction agreement
2. Decision rate correlation
3. Per-scenario emotion profiles
4. Per-emotion scenario profiles
5. Magnitude ratio
6. Cross-variant robustness

Usage:
    python -m steering_tests.self_consistency.compare_told_vs_steered \
        --told-judged results/self_consistency/continuations_gemma-3-27b-it_*_judged.jsonl \
        --steered-judged results/self_consistency/results/steered/gemma27b/*/*.jsonl \
        --output-dir results/self_consistency/comparison/gemma27b
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering_tests.self_consistency.judge_decisions import SCENARIO_JUDGES
from steering_tests.self_consistency.scenarios import EMOTIONS
from steering_tests.steering_utils.provenance import get_provenance

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid")

# Colours for told vs steered
COLOR_TOLD = "#D4876A"    # Coral/Terra Cotta
COLOR_STEERED = "#7BA7D7"  # Sky Blue
COLOR_AGREE = "#7D9B7D"   # Olive Green
COLOR_DISAGREE = "#C17B8D"  # Dusty Rose


def load_judged(paths: List[Path], method_filter: Optional[str] = None) -> List[dict]:
    """Load judged JSONL files, skipping meta lines and nulls."""
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if "meta" in row:
                    continue
                if row.get("decision") is None:
                    continue
                if method_filter and row.get("method") != method_filter:
                    continue
                rows.append(row)
    return rows


def compute_decision_rates(
    rows: List[dict],
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute decision rate distributions per (scenario, setting).

    Returns: {(scenario, setting): {label: fraction, ...}}
    """
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for row in rows:
        key = (row["scenario"], row["setting"])
        counts[key][row["decision"]] += 1
        totals[key] += 1

    rates = {}
    for key, count_dict in counts.items():
        total = totals[key]
        rates[key] = {label: n / total for label, n in count_dict.items()}
    return rates


def get_primary_label(scenario: str) -> str:
    """Get the first (primary action) label for a scenario."""
    judge = SCENARIO_JUDGES.get(scenario)
    if judge is None:
        return None
    return judge["labels"][0]


def compute_baseline_primary_rate(
    rates: Dict[Tuple[str, str], Dict[str, float]],
    scenario: str,
) -> Optional[float]:
    """Get the baseline primary-action rate for a scenario."""
    primary = get_primary_label(scenario)
    if primary is None:
        return None
    baseline_rates = rates.get((scenario, "baseline"))
    if baseline_rates is None:
        return None
    return baseline_rates.get(primary, 0.0)


def compute_shifts(
    rates: Dict[Tuple[str, str], Dict[str, float]],
    scenarios: List[str],
    emotions: List[str],
) -> Dict[Tuple[str, str], float]:
    """Compute shift from baseline for each (scenario, emotion).

    Shift = primary_rate(emotion) - primary_rate(baseline).
    Positive shift = more primary action.
    """
    shifts = {}
    for scenario in scenarios:
        primary = get_primary_label(scenario)
        if primary is None:
            continue
        baseline_rate = compute_baseline_primary_rate(rates, scenario)
        if baseline_rate is None:
            continue

        for emotion in emotions:
            key = (scenario, emotion)
            emotion_rates = rates.get(key)
            if emotion_rates is None:
                continue
            emotion_rate = emotion_rates.get(primary, 0.0)
            shifts[key] = emotion_rate - baseline_rate

    return shifts


def find_steered_settings(rows: List[dict]) -> Dict[str, List[dict]]:
    """Group steered rows by emotion (ignoring scale suffix).

    Returns: {emotion: [rows...]}
    """
    groups = defaultdict(list)
    for row in rows:
        setting = row["setting"]
        if setting == "baseline":
            groups["baseline"].append(row)
            continue
        # Setting format: "{emotion}_{scale}pct"
        parts = setting.rsplit("_", 1)
        if len(parts) == 2 and parts[1].endswith("pct"):
            emotion = parts[0]
            groups[emotion].append(row)
        else:
            groups[setting].append(row)
    return groups


def compute_steered_rates_by_scale(
    rows: List[dict],
    scale_pct: Optional[float] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Compute decision rates for steered rows, optionally filtering by scale.

    Settings like "fear_20pct" are mapped to (scenario, "fear") keys.
    """
    filtered = []
    for row in rows:
        setting = row["setting"]
        if setting == "baseline":
            if scale_pct is None or scale_pct == 0:
                filtered.append(row)
            continue
        # Parse scale from setting
        parts = setting.rsplit("_", 1)
        if len(parts) == 2 and parts[1].endswith("pct"):
            row_scale = float(parts[1].replace("pct", ""))
            if scale_pct is not None and row_scale != scale_pct:
                continue
            # Map setting to just the emotion name
            row = dict(row)
            row["setting"] = parts[0]
            filtered.append(row)
        else:
            filtered.append(row)

    return compute_decision_rates(filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Compare told vs steered self-consistency results"
    )
    parser.add_argument(
        "--told-judged", nargs="+", required=True,
        help="Judged JSONL files from told (API) experiment",
    )
    parser.add_argument(
        "--steered-judged", nargs="+", required=True,
        help="Judged JSONL files from steered (vLLM) experiment",
    )
    parser.add_argument(
        "--steered-scale", type=float, default=None,
        help="Filter steered results to this scale %% (default: use all scales combined)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="steering_tests/self_consistency/results/comparison",
        help="Directory for output plots and stats",
    )
    parser.add_argument(
        "--model-label", type=str, default=None,
        help="Model label for plot titles (default: auto-detect from data)",
    )
    args = parser.parse_args()

    told_paths = [Path(p) for p in args.told_judged]
    steered_paths = [Path(p) for p in args.steered_judged]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading told (API) results...")
    told_rows = load_judged(told_paths)
    print(f"  {len(told_rows)} judged rows")

    print("Loading steered (vLLM) results...")
    # Steered judged files have the same format as told judged files
    steered_rows = load_judged(steered_paths)
    print(f"  {len(steered_rows)} judged rows")

    if not told_rows:
        raise ValueError("No told rows loaded")
    if not steered_rows:
        raise ValueError("No steered rows loaded")

    # Determine model label
    model_label = args.model_label
    if model_label is None:
        models = set(r.get("model", "unknown") for r in told_rows)
        model_label = next(iter(models)) if len(models) == 1 else "multi-model"

    # Find common scenarios
    told_scenarios = set(r["scenario"] for r in told_rows if r["setting"] != "baseline")
    steered_scenarios = set(r["scenario"] for r in steered_rows if r["setting"] != "baseline")
    common_scenarios = sorted(told_scenarios & steered_scenarios)
    print(f"\nCommon scenarios: {len(common_scenarios)}")
    print(f"  {common_scenarios}")

    # Find common emotions
    told_settings = set(r["setting"] for r in told_rows) - {"baseline"}
    steered_settings = set()
    for r in steered_rows:
        s = r["setting"]
        if s == "baseline":
            continue
        parts = s.rsplit("_", 1)
        if len(parts) == 2 and parts[1].endswith("pct"):
            steered_settings.add(parts[0])
        else:
            steered_settings.add(s)

    common_emotions = sorted(set(EMOTIONS) & told_settings & steered_settings)
    print(f"Common emotions: {len(common_emotions)}")
    print(f"  {common_emotions}")

    # Compute rates
    told_rates = compute_decision_rates(told_rows)
    steered_rates = compute_steered_rates_by_scale(steered_rows, args.steered_scale)

    # Compute shifts
    told_shifts = compute_shifts(told_rates, common_scenarios, common_emotions)
    steered_shifts = compute_shifts(steered_rates, common_scenarios, common_emotions)

    # Find common (scenario, emotion) pairs
    common_keys = sorted(set(told_shifts.keys()) & set(steered_shifts.keys()))
    print(f"\nCommon (scenario, emotion) pairs: {len(common_keys)}")

    if not common_keys:
        print("ERROR: No common (scenario, emotion) pairs found!")
        return

    # =========================================================================
    # 1. Shift direction agreement
    # =========================================================================
    agree = 0
    disagree = 0
    zero_both = 0
    for key in common_keys:
        ts = told_shifts[key]
        ss = steered_shifts[key]
        if ts == 0 and ss == 0:
            zero_both += 1
        elif (ts > 0 and ss > 0) or (ts < 0 and ss < 0):
            agree += 1
        else:
            disagree += 1

    total_directional = agree + disagree
    agreement_rate = agree / total_directional if total_directional > 0 else 0

    print(f"\n{'='*60}")
    print(f"1. SHIFT DIRECTION AGREEMENT")
    print(f"{'='*60}")
    print(f"  Same direction: {agree}/{total_directional} = {agreement_rate:.1%}")
    print(f"  Opposite direction: {disagree}/{total_directional}")
    print(f"  Both zero: {zero_both}")

    # =========================================================================
    # 2. Decision rate correlation
    # =========================================================================
    told_vals = [told_shifts[k] for k in common_keys]
    steered_vals = [steered_shifts[k] for k in common_keys]

    r, p = stats.pearsonr(told_vals, steered_vals)
    rho, p_rho = stats.spearmanr(told_vals, steered_vals)

    print(f"\n{'='*60}")
    print(f"2. SHIFT MAGNITUDE CORRELATION")
    print(f"{'='*60}")
    print(f"  Pearson r = {r:.3f} (p = {p:.2e})")
    print(f"  Spearman rho = {rho:.3f} (p = {p_rho:.2e})")

    # =========================================================================
    # 3. Per-scenario emotion profiles
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"3. PER-SCENARIO EMOTION PROFILE CORRELATIONS")
    print(f"{'='*60}")

    scenario_correlations = {}
    for scenario in common_scenarios:
        t_profile = []
        s_profile = []
        for emotion in common_emotions:
            key = (scenario, emotion)
            if key in told_shifts and key in steered_shifts:
                t_profile.append(told_shifts[key])
                s_profile.append(steered_shifts[key])
        if len(t_profile) >= 3:
            r_sc, p_sc = stats.pearsonr(t_profile, s_profile)
            scenario_correlations[scenario] = (r_sc, p_sc, len(t_profile))
            sig = "*" if p_sc < 0.05 else ""
            print(f"  {scenario:30s} r={r_sc:+.3f} (p={p_sc:.3f}) n={len(t_profile)} {sig}")

    # =========================================================================
    # 4. Per-emotion scenario profiles
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"4. PER-EMOTION SCENARIO PROFILE CORRELATIONS")
    print(f"{'='*60}")

    emotion_correlations = {}
    for emotion in common_emotions:
        t_profile = []
        s_profile = []
        for scenario in common_scenarios:
            key = (scenario, emotion)
            if key in told_shifts and key in steered_shifts:
                t_profile.append(told_shifts[key])
                s_profile.append(steered_shifts[key])
        if len(t_profile) >= 3:
            r_em, p_em = stats.pearsonr(t_profile, s_profile)
            emotion_correlations[emotion] = (r_em, p_em, len(t_profile))
            sig = "*" if p_em < 0.05 else ""
            print(f"  {emotion:20s} r={r_em:+.3f} (p={p_em:.3f}) n={len(t_profile)} {sig}")

    # =========================================================================
    # 5. Magnitude ratio
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"5. MAGNITUDE RATIO (steered / told)")
    print(f"{'='*60}")

    ratios = []
    for key in common_keys:
        ts = told_shifts[key]
        ss = steered_shifts[key]
        if ts != 0 and ((ts > 0 and ss > 0) or (ts < 0 and ss < 0)):
            ratios.append(abs(ss) / abs(ts))

    if ratios:
        print(f"  Direction-agreeing pairs: {len(ratios)}")
        print(f"  Median ratio: {np.median(ratios):.2f}")
        print(f"  Mean ratio: {np.mean(ratios):.2f}")
        print(f"  Std: {np.std(ratios):.2f}")
        print(f"  Range: [{min(ratios):.2f}, {max(ratios):.2f}]")
    else:
        print("  No direction-agreeing pairs with non-zero told shift")

    # =========================================================================
    # PLOTS
    # =========================================================================

    # --- Scatter: told vs steered shifts ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(told_vals, steered_vals, alpha=0.5, s=40, c=COLOR_STEERED, edgecolors="white", linewidths=0.5)

    # Add diagonal line
    lims = [
        min(min(told_vals), min(steered_vals)) - 0.05,
        max(max(told_vals), max(steered_vals)) + 0.05,
    ]
    ax.plot(lims, lims, "--", color="#888888", alpha=0.5, linewidth=1)
    ax.axhline(0, color="#888888", alpha=0.3, linewidth=0.5)
    ax.axvline(0, color="#888888", alpha=0.3, linewidth=0.5)

    ax.set_xlabel("Told shift (primary action rate - baseline)", fontsize=12)
    ax.set_ylabel("Steered shift (primary action rate - baseline)", fontsize=12)
    ax.set_title(
        f"Told vs Steered Decision Shifts — {model_label}\n"
        f"r={r:.3f}, rho={rho:.3f}, direction agreement={agreement_rate:.0%}",
        fontsize=13,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    scatter_path = out_dir / "told_vs_steered_scatter.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {scatter_path}")

    # --- Heatmap: direction agreement ---
    n_sc = len(common_scenarios)
    n_em = len(common_emotions)
    agreement_matrix = np.full((n_sc, n_em), np.nan)

    for i, scenario in enumerate(common_scenarios):
        for j, emotion in enumerate(common_emotions):
            key = (scenario, emotion)
            if key in told_shifts and key in steered_shifts:
                ts = told_shifts[key]
                ss = steered_shifts[key]
                if ts == 0 and ss == 0:
                    agreement_matrix[i, j] = 0.5  # neutral
                elif (ts > 0 and ss > 0) or (ts < 0 and ss < 0):
                    agreement_matrix[i, j] = 1.0  # agree
                else:
                    agreement_matrix[i, j] = 0.0  # disagree

    fig, ax = plt.subplots(figsize=(14, 8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([COLOR_DISAGREE, "#CCCCCC", COLOR_AGREE])

    im = ax.imshow(agreement_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(n_em))
    ax.set_xticklabels(common_emotions, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_sc))
    ax.set_yticklabels(common_scenarios, fontsize=10)
    ax.set_title(f"Direction Agreement: Told vs Steered — {model_label}", fontsize=13)

    # Add colorbar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_AGREE, label="Same direction"),
        Patch(facecolor="#CCCCCC", label="Both zero"),
        Patch(facecolor=COLOR_DISAGREE, label="Opposite direction"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10, framealpha=0.9)

    heatmap_path = out_dir / "direction_agreement_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {heatmap_path}")

    # --- Per-scenario correlation bar chart ---
    if scenario_correlations:
        fig, ax = plt.subplots(figsize=(12, 6))
        sc_names = sorted(scenario_correlations.keys())
        sc_rs = [scenario_correlations[s][0] for s in sc_names]
        sc_ps = [scenario_correlations[s][1] for s in sc_names]
        colors = [COLOR_AGREE if p < 0.05 else "#CCCCCC" for p in sc_ps]

        bars = ax.barh(range(len(sc_names)), sc_rs, color=colors, edgecolor="white")
        ax.set_yticks(range(len(sc_names)))
        ax.set_yticklabels(sc_names, fontsize=10)
        ax.set_xlabel("Pearson r (told vs steered emotion profile)", fontsize=12)
        ax.set_title(f"Per-Scenario Emotion Profile Correlation — {model_label}", fontsize=13)
        ax.axvline(0, color="#888888", linewidth=0.5)
        ax.set_xlim(-1.1, 1.1)

        corr_sc_path = out_dir / "per_scenario_correlation.png"
        fig.savefig(corr_sc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {corr_sc_path}")

    # --- Per-emotion correlation bar chart ---
    if emotion_correlations:
        fig, ax = plt.subplots(figsize=(12, 6))
        em_names = sorted(emotion_correlations.keys())
        em_rs = [emotion_correlations[e][0] for e in em_names]
        em_ps = [emotion_correlations[e][1] for e in em_names]
        colors = [COLOR_STEERED if p < 0.05 else "#CCCCCC" for p in em_ps]

        bars = ax.barh(range(len(em_names)), em_rs, color=colors, edgecolor="white")
        ax.set_yticks(range(len(em_names)))
        ax.set_yticklabels(em_names, fontsize=10)
        ax.set_xlabel("Pearson r (told vs steered scenario profile)", fontsize=12)
        ax.set_title(f"Per-Emotion Scenario Profile Correlation — {model_label}", fontsize=13)
        ax.axvline(0, color="#888888", linewidth=0.5)
        ax.set_xlim(-1.1, 1.1)

        corr_em_path = out_dir / "per_emotion_correlation.png"
        fig.savefig(corr_em_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {corr_em_path}")

    # --- Side-by-side shift heatmaps ---
    told_matrix = np.full((n_sc, n_em), np.nan)
    steered_matrix = np.full((n_sc, n_em), np.nan)

    for i, scenario in enumerate(common_scenarios):
        for j, emotion in enumerate(common_emotions):
            key = (scenario, emotion)
            if key in told_shifts:
                told_matrix[i, j] = told_shifts[key]
            if key in steered_shifts:
                steered_matrix[i, j] = steered_shifts[key]

    vmax = max(
        np.nanmax(np.abs(told_matrix)) if not np.all(np.isnan(told_matrix)) else 0.5,
        np.nanmax(np.abs(steered_matrix)) if not np.all(np.isnan(steered_matrix)) else 0.5,
    )
    vmax = max(vmax, 0.1)  # Ensure non-zero range

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    for ax, matrix, title in [
        (ax1, told_matrix, "Told (API)"),
        (ax2, steered_matrix, "Steered (vLLM)"),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n_em))
        ax.set_xticklabels(common_emotions, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(n_sc))
        ax.set_yticklabels(common_scenarios, fontsize=10)
        ax.set_title(f"{title}", fontsize=13)

        # Add text annotations
        for i in range(n_sc):
            for j in range(n_em):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=[ax1, ax2], label="Shift from baseline", shrink=0.8)
    fig.suptitle(f"Decision Shift Comparison — {model_label}", fontsize=14, y=1.02)

    heatmap_pair_path = out_dir / "shift_heatmaps_comparison.png"
    fig.savefig(heatmap_pair_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {heatmap_pair_path}")

    # =========================================================================
    # Save summary stats
    # =========================================================================
    summary = {
        "provenance": get_provenance(
            script=__file__,
            extra={
                "told_files": [str(p) for p in told_paths],
                "steered_files": [str(p) for p in steered_paths],
                "steered_scale_filter": args.steered_scale,
                "model_label": model_label,
            },
        ),
        "n_told_rows": len(told_rows),
        "n_steered_rows": len(steered_rows),
        "common_scenarios": common_scenarios,
        "common_emotions": common_emotions,
        "n_common_pairs": len(common_keys),
        "direction_agreement": {
            "same": agree,
            "opposite": disagree,
            "both_zero": zero_both,
            "rate": agreement_rate,
        },
        "correlation": {
            "pearson_r": r,
            "pearson_p": p,
            "spearman_rho": rho,
            "spearman_p": p_rho,
        },
        "per_scenario_correlation": {
            sc: {"r": vals[0], "p": vals[1], "n": vals[2]}
            for sc, vals in scenario_correlations.items()
        },
        "per_emotion_correlation": {
            em: {"r": vals[0], "p": vals[1], "n": vals[2]}
            for em, vals in emotion_correlations.items()
        },
        "magnitude_ratio": {
            "n_pairs": len(ratios),
            "median": float(np.median(ratios)) if ratios else None,
            "mean": float(np.mean(ratios)) if ratios else None,
            "std": float(np.std(ratios)) if ratios else None,
        },
    }

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Save meta.json for provenance
    meta_path = out_dir / "comparison_summary.meta.json"
    with open(meta_path, "w") as f:
        json.dump(summary["provenance"], f, indent=2)
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
