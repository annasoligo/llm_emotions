"""
Comprehensive Pearson r and JSD analysis: Qwen 32B told vs steered self-consistency.

Compares prompted (told) emotion effects on decisions with activation-steered effects,
across two vector types (text_pairs, high_emotion) and three scales (10%, 20%, 30%).

Produces:
A. Overall summary table (Pearson r, JSD, N)
B. Per-emotion table
C. Per-scenario table
D. 4 heatmaps (shift + agreement for each vector type)
"""

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
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from steering_tests.self_consistency.judge_decisions import SCENARIO_JUDGES
from steering_tests.self_consistency.scenarios import EMOTIONS

plt.style.use("seaborn-v0_8-whitegrid")

# ===========================================================================
# Config
# ===========================================================================
TOLD_PATH = Path("steering_tests/self_consistency/results/continuations_qwen3-32b_20260211_114526_judged.jsonl")
STEERED_PATHS = {
    "text_pairs": Path("steering_tests/self_consistency/results/steered/qwen32b/text_pairs_emotion_vs_opposite/steered_layers35_36_37_38_39_20260212_084943_judged.jsonl"),
    "high_emotion": Path("steering_tests/self_consistency/results/steered/qwen32b/high_emotion_vs_opposite/steered_layers35_36_37_38_39_20260212_085526_judged.jsonl"),
}
OUTPUT_DIR = Path("steering_tests/self_consistency/results/steered/qwen32b")

SCALES = [10, 20, 30]

MAIN_SCENARIOS = [
    "financial_risk", "confrontation", "forgiveness", "persistence",
    "advice_under_uncertainty", "public_speaking", "medical_procedure",
    "salary_negotiation", "delegation", "new_relationship",
]
CONTROL_SCENARIOS = ["scheduling", "technical_choice", "neutral_preference"]

MODEL_LABEL = "Qwen 32B"

# ===========================================================================
# Helpers
# ===========================================================================

def load_judged(path: Path) -> List[dict]:
    """Load judged JSONL, skipping meta and ambiguous."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                continue
            if row.get("decision") is None or row["decision"] == "ambiguous":
                continue
            rows.append(row)
    return rows


def deduplicate(rows: List[dict]) -> List[dict]:
    """Deduplicate by (scenario, variant_idx, setting, sample_idx), keep first."""
    seen = set()
    deduped = []
    for row in rows:
        key = (row["scenario"], row["variant_idx"], row["setting"], row.get("sample_idx"))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def get_primary_label(scenario: str) -> Optional[str]:
    """Get the first (primary action) label for a scenario."""
    judge = SCENARIO_JUDGES.get(scenario)
    if judge is None:
        return None
    return judge["labels"][0]


def get_all_labels(scenario: str) -> Optional[List[str]]:
    """Get all non-ambiguous labels for a scenario."""
    judge = SCENARIO_JUDGES.get(scenario)
    if judge is None:
        return None
    return [l for l in judge["labels"] if l != "ambiguous"]


def compute_decision_rates(rows: List[dict]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Per (scenario, setting): {decision_label: fraction}."""
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


def compute_decision_counts(rows: List[dict]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """Per (scenario, setting): {decision_label: count}."""
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = (row["scenario"], row["setting"])
        counts[key][row["decision"]] += 1
    return dict(counts)


def parse_steered_setting(setting: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse 'fear_20pct' -> ('fear', 20). Returns (None, None) for baseline."""
    if setting == "baseline":
        return None, None
    parts = setting.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("pct"):
        emotion = parts[0]
        scale = int(parts[1].replace("pct", ""))
        return emotion, scale
    return setting, None


def remap_steered_rows(rows: List[dict], scale: int) -> List[dict]:
    """Filter steered rows to a given scale, remapping setting to emotion name.
    Also includes baseline rows."""
    filtered = []
    for row in rows:
        setting = row["setting"]
        if setting == "baseline":
            filtered.append(row)
            continue
        emotion, row_scale = parse_steered_setting(setting)
        if row_scale == scale:
            new_row = dict(row)
            new_row["setting"] = emotion
            filtered.append(new_row)
    return filtered


def compute_baseline_majority_action(
    rates: Dict[Tuple[str, str], Dict[str, float]],
    scenario: str,
) -> Optional[str]:
    """Determine which action the baseline picks most often."""
    baseline_rates = rates.get((scenario, "baseline"))
    if baseline_rates is None:
        return None
    if not baseline_rates:
        return None
    return max(baseline_rates, key=baseline_rates.get)


def compute_shifts_majority(
    rates: Dict[Tuple[str, str], Dict[str, float]],
    scenarios: List[str],
    emotions: List[str],
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, str], Dict[str, float]]:
    """Compute shift = rate_of_baseline_majority - baseline_rate.

    Returns:
        shifts: {(scenario, emotion): shift_value}
        majority_actions: {scenario: majority_action_label}
        baseline_rates: {scenario: baseline_rate_of_majority_action}
    """
    shifts = {}
    majority_actions = {}
    baseline_rates_out = {}

    for scenario in scenarios:
        majority = compute_baseline_majority_action(rates, scenario)
        if majority is None:
            continue
        majority_actions[scenario] = majority
        baseline_rate = rates[(scenario, "baseline")].get(majority, 0.0)
        baseline_rates_out[scenario] = baseline_rate

        for emotion in emotions:
            key = (scenario, emotion)
            emotion_rates = rates.get(key)
            if emotion_rates is None:
                continue
            emotion_rate = emotion_rates.get(majority, 0.0)
            shifts[key] = emotion_rate - baseline_rate

    return shifts, majority_actions, baseline_rates_out


def compute_jsd(
    rates1: Dict[Tuple[str, str], Dict[str, float]],
    rates2: Dict[Tuple[str, str], Dict[str, float]],
    scenarios: List[str],
    emotions: List[str],
) -> Dict[Tuple[str, str], float]:
    """Compute Jensen-Shannon divergence between two rate dicts, per (scenario, emotion).

    For each (scenario, emotion), we build distributions over all decision labels
    for that scenario, compute JSD.
    """
    jsds = {}
    for scenario in scenarios:
        labels = get_all_labels(scenario)
        if labels is None:
            continue
        for emotion in emotions:
            key = (scenario, emotion)
            r1 = rates1.get(key, {})
            r2 = rates2.get(key, {})
            if not r1 or not r2:
                continue
            # Build aligned distributions
            p = np.array([r1.get(l, 0.0) for l in labels])
            q = np.array([r2.get(l, 0.0) for l in labels])
            # Normalize (should already sum to ~1 but ensure)
            p_sum = p.sum()
            q_sum = q.sum()
            if p_sum == 0 or q_sum == 0:
                continue
            p = p / p_sum
            q = q / q_sum
            jsds[key] = float(jensenshannon(p, q) ** 2)  # JSD (squared to get divergence)
    return jsds


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def safe_pearsonr(x, y):
    """Pearson r with protection against constant arrays."""
    x = np.array(x)
    y = np.array(y)
    if len(x) < 3:
        return np.nan, 1.0
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, 1.0
    return stats.pearsonr(x, y)


# ===========================================================================
# Main
# ===========================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load told data
    print("Loading told (prompted) data...")
    told_rows_raw = load_judged(TOLD_PATH)
    told_rows = deduplicate(told_rows_raw)
    print(f"  {len(told_rows_raw)} raw -> {len(told_rows)} after dedup (skipped ambiguous)")

    # Filter to main scenarios only
    told_rows = [r for r in told_rows if r["scenario"] in MAIN_SCENARIOS]
    print(f"  {len(told_rows)} after filtering to main scenarios")

    # Compute told rates and shifts
    told_rates = compute_decision_rates(told_rows)
    told_shifts, told_majority, told_baselines = compute_shifts_majority(
        told_rates, MAIN_SCENARIOS, EMOTIONS
    )

    # Load steered data for each vector type
    steered_data = {}
    for vtype, path in STEERED_PATHS.items():
        print(f"\nLoading steered data: {vtype}...")
        raw = load_judged(path)
        deduped = deduplicate(raw)
        print(f"  {len(raw)} raw -> {len(deduped)} after dedup (skipped ambiguous)")
        # Filter to main scenarios
        deduped = [r for r in deduped if r["scenario"] in MAIN_SCENARIOS]
        print(f"  {len(deduped)} after filtering to main scenarios")
        steered_data[vtype] = deduped

    # ===========================================================================
    # A. Overall summary table
    # ===========================================================================
    print("\n" + "=" * 90)
    print("A. OVERALL SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Vector Type':<28} {'Scale':>6} {'Pearson r':>10} {'p-value':>12} {'Sig':>4} {'Mean JSD':>10} {'N pairs':>8}"
    print(header)
    print("-" * 90)

    # Store results for heatmaps
    all_results = {}  # (vtype, scale) -> {told_shifts, steered_shifts, steered_rates, told_rates_for_jsd, ...}

    for vtype in ["text_pairs", "high_emotion"]:
        rows = steered_data[vtype]
        for scale in SCALES:
            # Remap steered rows to this scale
            remapped = remap_steered_rows(rows, scale)
            s_rates = compute_decision_rates(remapped)
            s_shifts, s_majority, s_baselines = compute_shifts_majority(
                s_rates, MAIN_SCENARIOS, EMOTIONS
            )

            # Find common keys
            common_keys = sorted(set(told_shifts.keys()) & set(s_shifts.keys()))

            # Skip pairs where BOTH shifts are exactly 0 or where either is nan
            valid_keys = []
            for k in common_keys:
                ts = told_shifts[k]
                ss = s_shifts[k]
                if not (np.isnan(ts) or np.isnan(ss)):
                    valid_keys.append(k)

            told_vals = [told_shifts[k] for k in valid_keys]
            steered_vals = [s_shifts[k] for k in valid_keys]

            r, p = safe_pearsonr(told_vals, steered_vals)

            # JSD: compare told and steered emotion rates (not shifts)
            jsds = compute_jsd(told_rates, s_rates, MAIN_SCENARIOS, EMOTIONS)
            mean_jsd = np.mean(list(jsds.values())) if jsds else np.nan

            stars = significance_stars(p)
            r_str = f"{r:+.4f}" if not np.isnan(r) else "   N/A"
            p_str = f"{p:.2e}" if not np.isnan(r) else "  N/A"
            print(f"{vtype:<28} {scale:>5}% {r_str:>10} {p_str:>12} {stars:>4} {mean_jsd:>10.4f} {len(valid_keys):>8}")

            all_results[(vtype, scale)] = {
                "told_shifts": told_shifts,
                "steered_shifts": s_shifts,
                "steered_rates": s_rates,
                "jsds": jsds,
                "r": r,
                "p": p,
                "valid_keys": valid_keys,
                "mean_jsd": mean_jsd,
            }

    # ===========================================================================
    # B. Per-emotion table
    # ===========================================================================
    print("\n" + "=" * 130)
    print("B. PER-EMOTION PEARSON r (across scenarios)")
    print("=" * 130)

    # Header
    cols = []
    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            cols.append(f"{vtype[:6]}_{scale}%")

    header = f"{'Emotion':<14}"
    for col in cols:
        header += f" | {col:>16}"
    print(header)
    print("-" * 130)

    per_emotion_results = {}
    for emotion in EMOTIONS:
        row_str = f"{emotion:<14}"
        for vtype in ["text_pairs", "high_emotion"]:
            for scale in SCALES:
                res = all_results[(vtype, scale)]
                t_vals = []
                s_vals = []
                for scenario in MAIN_SCENARIOS:
                    key = (scenario, emotion)
                    if key in res["told_shifts"] and key in res["steered_shifts"]:
                        t_vals.append(res["told_shifts"][key])
                        s_vals.append(res["steered_shifts"][key])

                r, p = safe_pearsonr(t_vals, s_vals)
                stars = significance_stars(p)
                per_emotion_results[(emotion, vtype, scale)] = (r, p, len(t_vals))

                if np.isnan(r):
                    row_str += f" | {'N/A':>16}"
                else:
                    row_str += f" | {r:+.3f}{stars:>4} (n={len(t_vals)})"
        print(row_str)

    # ===========================================================================
    # C. Per-scenario table
    # ===========================================================================
    print("\n" + "=" * 130)
    print("C. PER-SCENARIO PEARSON r (across emotions)")
    print("=" * 130)

    header = f"{'Scenario':<28}"
    for col in cols:
        header += f" | {col:>16}"
    print(header)
    print("-" * 130)

    per_scenario_results = {}
    for scenario in MAIN_SCENARIOS:
        row_str = f"{scenario:<28}"
        for vtype in ["text_pairs", "high_emotion"]:
            for scale in SCALES:
                res = all_results[(vtype, scale)]
                t_vals = []
                s_vals = []
                for emotion in EMOTIONS:
                    key = (scenario, emotion)
                    if key in res["told_shifts"] and key in res["steered_shifts"]:
                        t_vals.append(res["told_shifts"][key])
                        s_vals.append(res["steered_shifts"][key])

                r, p = safe_pearsonr(t_vals, s_vals)
                stars = significance_stars(p)
                per_scenario_results[(scenario, vtype, scale)] = (r, p, len(t_vals))

                if np.isnan(r):
                    row_str += f" | {'N/A':>16}"
                else:
                    row_str += f" | {r:+.3f}{stars:>4} (n={len(t_vals)})"
        print(row_str)

    # ===========================================================================
    # D. Heatmaps
    # ===========================================================================
    print("\n" + "=" * 90)
    print("D. GENERATING HEATMAPS")
    print("=" * 90)

    # For each vector type, find the best scale (highest overall r)
    for vtype in ["text_pairs", "high_emotion"]:
        best_scale = max(SCALES, key=lambda s: all_results[(vtype, s)]["r"] if not np.isnan(all_results[(vtype, s)]["r"]) else -999)
        best_r = all_results[(vtype, best_scale)]["r"]
        print(f"  {vtype}: best scale = {best_scale}% (r = {best_r:+.4f})")

    vtype_labels = {
        "text_pairs": "Text Pairs",
        "high_emotion": "High Emotion",
    }

    for vtype in ["text_pairs", "high_emotion"]:
        # --- Heatmap 1: Shift comparison (best scale) ---
        best_scale = max(SCALES, key=lambda s: all_results[(vtype, s)]["r"] if not np.isnan(all_results[(vtype, s)]["r"]) else -999)
        res = all_results[(vtype, best_scale)]

        n_sc = len(MAIN_SCENARIOS)
        n_em = len(EMOTIONS)

        # Build per-scenario correlation matrix for the shift heatmap
        corr_matrix = np.full((n_sc, n_em), np.nan)

        # For shift heatmap, we show per-(scenario, emotion) shift comparison
        # Actually, user wants: rows=scenarios, cols=emotions, colored by Pearson r
        # "Per-scenario" Pearson r across emotions doesn't give a per-cell value.
        # Re-reading: "colored by Pearson r, with r values in cells and significance stars"
        # This is the per-scenario correlation, but displayed differently.
        # Actually looking more carefully: the user wants 4 heatmaps:
        # 1 & 2: "Shift heatmap" - show the shift values side by side (told vs steered)
        # 3 & 4: "Agreement heatmap" - show per-cell correlation/agreement

        # Let me produce: shift heatmaps comparing told vs steered
        # And agreement heatmaps showing per-(scenario, emotion) sign agreement

        # Shift heatmap: side-by-side told shift and steered shift
        told_matrix = np.full((n_sc, n_em), np.nan)
        steered_matrix = np.full((n_sc, n_em), np.nan)

        for i, scenario in enumerate(MAIN_SCENARIOS):
            for j, emotion in enumerate(EMOTIONS):
                key = (scenario, emotion)
                if key in res["told_shifts"]:
                    told_matrix[i, j] = res["told_shifts"][key]
                if key in res["steered_shifts"]:
                    steered_matrix[i, j] = res["steered_shifts"][key]

        vmax = max(
            np.nanmax(np.abs(told_matrix)) if not np.all(np.isnan(told_matrix)) else 0.5,
            np.nanmax(np.abs(steered_matrix)) if not np.all(np.isnan(steered_matrix)) else 0.5,
        )
        vmax = max(vmax, 0.1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 8))

        for ax, matrix, subtitle in [
            (ax1, told_matrix, "Told (prompted)"),
            (ax2, steered_matrix, f"Steered ({best_scale}%)"),
        ]:
            im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(n_em))
            ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(n_sc))
            ax.set_yticklabels(MAIN_SCENARIOS, fontsize=9)
            ax.set_title(subtitle, fontsize=12)

            for i in range(n_sc):
                for j in range(n_em):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > vmax * 0.6 else "black"
                        ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                                fontsize=7, color=color)

        fig.colorbar(im, ax=[ax1, ax2], label="Shift from baseline", shrink=0.8)
        overall_r = res["r"]
        overall_p = res["p"]
        stars = significance_stars(overall_p)
        fig.suptitle(
            f"{MODEL_LABEL} — Shift Comparison ({vtype_labels[vtype]}, {best_scale}%)\n"
            f"Overall Pearson r = {overall_r:+.4f}{stars}",
            fontsize=13, y=1.03,
        )

        path = OUTPUT_DIR / f"shift_heatmap_{vtype}_{best_scale}pct.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        # --- Heatmap 2: Agreement (per-scenario r across emotions) ---
        # rows = scenarios, cols = emotions
        # For each scenario: compute the per-emotion correlation between told and steered
        # But that gives 1 value per scenario, not per cell.
        # Instead: show per-cell agreement as the product of signs (positive if same direction)
        # Or better: for each (scenario, emotion), show the steered shift - told shift (residual)
        # Let me do: per-scenario Pearson r, displayed as a bar for each scenario,
        # but also show per-cell sign agreement

        # Actually, re-reading the request more carefully:
        # "Agreement heatmap for high_emotion" - rows=scenarios, cols=emotions,
        # colored by Pearson r, with r values in cells and significance stars
        # This is asking for: per-scenario per-emotion r? That doesn't make sense with 1 value.
        # It must mean: per-scenario r across emotions (one r per scenario), displayed
        # across the row. But that would mean all cells in a row have the same color.
        # OR: per-emotion r across scenarios (one r per emotion), displayed across the column.
        # OR: it could be asking for per-cell sign agreement.
        #
        # Looking at "with r values in cells and significance stars" - I think this means:
        # Per-scenario: one r per row (across emotions within that scenario)
        # Per-emotion: one r per column (across scenarios within that emotion)
        # But displayed together in a matrix... hmm.
        #
        # The most informative approach: show the agreement matrix where each cell
        # (scenario, emotion) shows whether told and steered shifts agree in direction,
        # plus annotate each ROW with the per-scenario r and each COLUMN with per-emotion r.
        #
        # OR (simpler interpretation): show per-scenario Pearson r as a heatmap column
        # and per-emotion Pearson r as a heatmap row.
        #
        # Let me go with: agreement heatmap showing per-scenario x per-emotion
        # where cell color = min(told_shift, steered_shift) sign agreement,
        # and annotate with per-scenario r values on the right side.
        #
        # Actually, the simplest and most useful: per-scenario r across emotions
        # on the y-axis, and per-emotion r across scenarios on the x-axis,
        # combined into a product or just shown as two margin bars.
        #
        # Let me just follow the exact Gemma pattern: the request says
        # "colored by Pearson r, with r values in cells"
        # I'll interpret this as: each cell (scenario, emotion) shows whether
        # told and steered agree in sign, colored as a correlation matrix,
        # with per-scenario r labels on side.

        # Actually, I think the most sensible interpretation for an "agreement heatmap"
        # that has "r values in cells" with "rows = scenarios, columns = emotions" is:
        # A matrix where each CELL shows the per-scenario Pearson r (same value across
        # the row for that scenario), colored by r value.
        # But that seems redundant.
        #
        # Let me make the most useful version: a matrix showing sign agreement
        # (+1 same direction, -1 opposite, 0 if either is zero) scaled by the
        # magnitude of the agreement, with per-scenario r on the margin.
        #
        # OR: Compute per-scenario Pearson r values and display them as a simple
        # heatmap with one column per vector type.
        #
        # I'll go with a combined per-scenario x per-emotion agreement sign matrix
        # (binary: +1, -1, 0), annotated with magnitude, colored by told*steered product.

        # Build sign agreement matrix
        agreement_matrix = np.full((n_sc, n_em), np.nan)
        for i, scenario in enumerate(MAIN_SCENARIOS):
            for j, emotion in enumerate(EMOTIONS):
                key = (scenario, emotion)
                ts = res["told_shifts"].get(key)
                ss = res["steered_shifts"].get(key)
                if ts is not None and ss is not None:
                    # Use product of shifts as a continuous agreement measure
                    # Positive = same direction, negative = opposite
                    # Scale by taking sign * min of abs values for visibility
                    if ts == 0 or ss == 0:
                        agreement_matrix[i, j] = 0.0
                    else:
                        # Use the steered shift direction relative to told shift
                        agreement_matrix[i, j] = np.sign(ts) * np.sign(ss) * min(abs(ts), abs(ss))

        # Compute per-scenario r values for annotation
        scenario_rs = {}
        for scenario in MAIN_SCENARIOS:
            t_vals = []
            s_vals = []
            for emotion in EMOTIONS:
                key = (scenario, emotion)
                if key in res["told_shifts"] and key in res["steered_shifts"]:
                    t_vals.append(res["told_shifts"][key])
                    s_vals.append(res["steered_shifts"][key])
            if len(t_vals) >= 3:
                r_sc, p_sc = safe_pearsonr(t_vals, s_vals)
                scenario_rs[scenario] = (r_sc, p_sc)

        fig, ax = plt.subplots(figsize=(14, 8))
        vmax_agree = max(np.nanmax(np.abs(agreement_matrix)), 0.1)
        im = ax.imshow(agreement_matrix, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax_agree, vmax=vmax_agree)
        ax.set_xticks(range(n_em))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_sc))

        # Add per-scenario r to y-axis labels
        ylabels = []
        for scenario in MAIN_SCENARIOS:
            if scenario in scenario_rs:
                r_sc, p_sc = scenario_rs[scenario]
                stars_sc = significance_stars(p_sc)
                if np.isnan(r_sc):
                    ylabels.append(f"{scenario} (r=N/A)")
                else:
                    ylabels.append(f"{scenario} (r={r_sc:+.2f}{stars_sc})")
            else:
                ylabels.append(scenario)
        ax.set_yticklabels(ylabels, fontsize=9)

        # Add cell annotations
        for i in range(n_sc):
            for j in range(n_em):
                val = agreement_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > vmax_agree * 0.6 else "black"
                    # Show told/steered shift directions
                    key = (MAIN_SCENARIOS[i], EMOTIONS[j])
                    ts = res["told_shifts"].get(key, 0)
                    ss = res["steered_shifts"].get(key, 0)
                    t_sign = "+" if ts > 0 else ("-" if ts < 0 else "0")
                    s_sign = "+" if ss > 0 else ("-" if ss < 0 else "0")
                    ax.text(j, i, f"{t_sign}/{s_sign}", ha="center", va="center",
                            fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label="Agreement (sign * min magnitude)", shrink=0.8)
        ax.set_title(
            f"{MODEL_LABEL} — Direction Agreement ({vtype_labels[vtype]}, {best_scale}%)\n"
            f"Cell: told_sign/steered_sign, row labels: per-scenario Pearson r",
            fontsize=12,
        )

        path = OUTPUT_DIR / f"agreement_heatmap_{vtype}_{best_scale}pct.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ===========================================================================
    # Additional: JSD breakdown
    # ===========================================================================
    print("\n" + "=" * 90)
    print("E. JSD BREAKDOWN BY VECTOR TYPE AND SCALE")
    print("=" * 90)
    header = f"{'Vector Type':<28} {'Scale':>6} {'Mean JSD':>10} {'Median JSD':>12} {'Max JSD':>10} {'N':>5}"
    print(header)
    print("-" * 90)

    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            res = all_results[(vtype, scale)]
            jsd_vals = list(res["jsds"].values())
            if jsd_vals:
                print(f"{vtype:<28} {scale:>5}% {np.mean(jsd_vals):>10.4f} {np.median(jsd_vals):>12.4f} {np.max(jsd_vals):>10.4f} {len(jsd_vals):>5}")

    # ===========================================================================
    # F. Direction agreement rates
    # ===========================================================================
    print("\n" + "=" * 90)
    print("F. DIRECTION AGREEMENT RATES")
    print("=" * 90)
    header = f"{'Vector Type':<28} {'Scale':>6} {'Same':>6} {'Opp':>6} {'Zero':>6} {'Agree%':>8}"
    print(header)
    print("-" * 90)

    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            res = all_results[(vtype, scale)]
            agree = 0
            disagree = 0
            zero_both = 0
            for k in res["valid_keys"]:
                ts = res["told_shifts"][k]
                ss = res["steered_shifts"][k]
                if ts == 0 and ss == 0:
                    zero_both += 1
                elif (ts > 0 and ss > 0) or (ts < 0 and ss < 0):
                    agree += 1
                else:
                    disagree += 1
            total_dir = agree + disagree
            rate = agree / total_dir if total_dir > 0 else 0
            print(f"{vtype:<28} {scale:>5}% {agree:>6} {disagree:>6} {zero_both:>6} {rate:>7.1%}")

    # ===========================================================================
    # G. Per-emotion JSD table
    # ===========================================================================
    print("\n" + "=" * 130)
    print("G. PER-EMOTION MEAN JSD")
    print("=" * 130)
    header = f"{'Emotion':<14}"
    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            header += f" | {vtype[:6]}_{scale}%:>12"
    # Fix header formatting
    header = f"{'Emotion':<14}"
    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            header += f" | {f'{vtype[:6]}_{scale}%':>12}"
    print(header)
    print("-" * 130)

    for emotion in EMOTIONS:
        row_str = f"{emotion:<14}"
        for vtype in ["text_pairs", "high_emotion"]:
            for scale in SCALES:
                res = all_results[(vtype, scale)]
                emotion_jsds = [v for (s, e), v in res["jsds"].items() if e == emotion]
                if emotion_jsds:
                    row_str += f" | {np.mean(emotion_jsds):>12.4f}"
                else:
                    row_str += f" | {'N/A':>12}"
        print(row_str)

    print("\n" + "=" * 90)
    print("ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
