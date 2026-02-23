"""
Comprehensive Pearson r + Jensen-Shannon divergence analysis:
Qwen 235B told (prompted) vs steered (activation-steered) self-consistency.

Compares two vector types (text_pairs, high_emotion) at scales 20/50/100%.
Produces overall summary, per-emotion, per-scenario tables, and 4 heatmaps.
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

# ── Config ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

EMOTIONS = [
    "fear", "calm", "disgust", "admiration", "shame", "pride",
    "anxiety", "relief", "despair", "hope", "sadness", "joy",
]

MAIN_SCENARIOS = [
    "financial_risk", "confrontation", "forgiveness", "persistence",
    "advice_under_uncertainty", "public_speaking", "medical_procedure",
    "salary_negotiation", "delegation", "new_relationship",
]

SCALES = [20, 50, 100]

TOLD_PATH = Path(
    "steering_tests/self_consistency/results/"
    "continuations_qwen3-235b-a22b_20260211_120037_judged.jsonl"
)
STEERED_PATHS = {
    "text_pairs": Path(
        "steering_tests/self_consistency/results/steered/qwen235b/"
        "text_pairs_emotion_vs_opposite/"
        "steered_layers50_51_52_53_54_20260212_111512_judged.jsonl"
    ),
    "high_emotion": Path(
        "steering_tests/self_consistency/results/steered/qwen235b/"
        "high_emotion_vs_opposite/"
        "steered_layers50_51_52_53_54_20260212_112717_judged.jsonl"
    ),
}
OUT_DIR = Path(
    "steering_tests/self_consistency/results/steered/qwen235b"
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL, skip meta and null/ambiguous decisions, deduplicate."""
    rows = []
    seen = set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                continue
            d = row.get("decision")
            if d is None or d == "ambiguous":
                continue
            # Deduplicate by (scenario, variant_idx, setting, sample_idx)
            dedup_key = (
                row["scenario"],
                row.get("variant_idx"),
                row["setting"],
                row.get("sample_idx"),
            )
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(row)
    return rows


def get_majority_action(
    rows: List[dict], scenario: str
) -> Optional[str]:
    """Find the majority action for baseline of a given scenario."""
    counts = defaultdict(int)
    for r in rows:
        if r["scenario"] == scenario and r["setting"] == "baseline":
            counts[r["decision"]] += 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def compute_rate(
    rows: List[dict],
    scenario: str,
    setting: str,
    majority_action: str,
) -> Optional[float]:
    """Fraction of decisions matching majority_action for (scenario, setting)."""
    matching = [
        r for r in rows
        if r["scenario"] == scenario and r["setting"] == setting
    ]
    if not matching:
        return None
    n_majority = sum(1 for r in matching if r["decision"] == majority_action)
    return n_majority / len(matching)


def compute_distribution(
    rows: List[dict],
    scenario: str,
    setting: str,
) -> Dict[str, float]:
    """Compute full decision distribution for (scenario, setting)."""
    matching = [
        r for r in rows
        if r["scenario"] == scenario and r["setting"] == setting
    ]
    if not matching:
        return {}
    counts = defaultdict(int)
    for r in matching:
        counts[r["decision"]] += 1
    total = len(matching)
    return {k: v / total for k, v in counts.items()}


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "** "
    elif p < 0.05:
        return "*  "
    return "   "


def safe_pearsonr(x, y):
    """Pearson r, returning (nan, nan) if either array is constant."""
    x = np.array(x)
    y = np.array(y)
    if len(x) < 3:
        return np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return stats.pearsonr(x, y)


def compute_jsd(dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
    """Jensen-Shannon divergence between two decision distributions."""
    all_keys = sorted(set(dist1.keys()) | set(dist2.keys()))
    if not all_keys:
        return np.nan
    p = np.array([dist1.get(k, 0.0) for k in all_keys])
    q = np.array([dist2.get(k, 0.0) for k in all_keys])
    # Ensure valid distributions
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return np.nan
    p = p / p_sum
    q = q / q_sum
    return float(jensenshannon(p, q) ** 2)  # JSD (squared form, 0-1 range)


# ── Parse steered settings ─────────────────────────────────────────────────

def parse_steered_setting(setting: str) -> Tuple[str, Optional[int]]:
    """Parse 'fear_20pct' -> ('fear', 20). 'baseline' -> ('baseline', None)."""
    if setting == "baseline":
        return "baseline", None
    parts = setting.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("pct"):
        emotion = parts[0]
        scale = int(parts[1].replace("pct", ""))
        return emotion, scale
    raise ValueError(f"Unexpected steered setting: {setting}")


def filter_steered_rows(
    rows: List[dict], scale: int
) -> List[dict]:
    """Filter steered rows to baseline + a specific scale, mapping setting to emotion name."""
    filtered = []
    for r in rows:
        emo, sc = parse_steered_setting(r["setting"])
        if emo == "baseline":
            filtered.append(r)
        elif sc == scale:
            new_r = dict(r)
            new_r["setting"] = emo
            filtered.append(new_r)
    return filtered


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("QWEN 235B: TOLD vs STEERED SELF-CONSISTENCY ANALYSIS")
    print("=" * 80)
    print()

    # Load data
    print("Loading told data...")
    told_rows = load_jsonl(TOLD_PATH)
    # Filter to main scenarios only
    told_rows = [r for r in told_rows if r["scenario"] in MAIN_SCENARIOS]
    print(f"  {len(told_rows)} rows (main scenarios, non-ambiguous, deduplicated)")

    steered_all = {}
    for vtype, path in STEERED_PATHS.items():
        print(f"Loading steered ({vtype})...")
        rows = load_jsonl(path)
        rows = [r for r in rows if r["scenario"] in MAIN_SCENARIOS]
        steered_all[vtype] = rows
        print(f"  {len(rows)} rows (main scenarios, non-ambiguous, deduplicated)")

    # ── Compute told rates ──────────────────────────────────────────────────
    # For each scenario, determine baseline majority action from TOLD baseline
    told_majority = {}
    for sc in MAIN_SCENARIOS:
        maj = get_majority_action(told_rows, sc)
        told_majority[sc] = maj

    told_baseline_rate = {}
    told_rate = {}
    told_shift = {}
    for sc in MAIN_SCENARIOS:
        maj = told_majority[sc]
        if maj is None:
            continue
        br = compute_rate(told_rows, sc, "baseline", maj)
        told_baseline_rate[sc] = br
        for emo in EMOTIONS:
            er = compute_rate(told_rows, sc, emo, maj)
            if er is not None and br is not None:
                told_rate[(sc, emo)] = er
                told_shift[(sc, emo)] = er - br

    print(f"\nTold shifts computed: {len(told_shift)} (scenario, emotion) pairs")

    # ── Compute told distributions for JSD ──────────────────────────────────
    told_dist = {}
    for sc in MAIN_SCENARIOS:
        for emo in EMOTIONS:
            d = compute_distribution(told_rows, sc, emo)
            if d:
                told_dist[(sc, emo)] = d

    # ── Compute steered rates per (vector_type, scale) ──────────────────────
    steered_results = {}  # (vtype, scale) -> {shift, rate, baseline_rate, majority, dist}

    for vtype, all_rows in steered_all.items():
        for scale in SCALES:
            frows = filter_steered_rows(all_rows, scale)

            # Steered baseline majority (independent of told)
            s_majority = {}
            s_baseline_rate = {}
            s_rate = {}
            s_shift = {}
            s_dist = {}

            for sc in MAIN_SCENARIOS:
                maj = get_majority_action(frows, sc)
                s_majority[sc] = maj
                if maj is None:
                    continue
                br = compute_rate(frows, sc, "baseline", maj)
                s_baseline_rate[sc] = br
                for emo in EMOTIONS:
                    er = compute_rate(frows, sc, emo, maj)
                    if er is not None and br is not None:
                        s_rate[(sc, emo)] = er
                        s_shift[(sc, emo)] = er - br
                    d = compute_distribution(frows, sc, emo)
                    if d:
                        s_dist[(sc, emo)] = d

            steered_results[(vtype, scale)] = {
                "majority": s_majority,
                "baseline_rate": s_baseline_rate,
                "rate": s_rate,
                "shift": s_shift,
                "dist": s_dist,
            }

    # ════════════════════════════════════════════════════════════════════════
    # A. OVERALL SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("A. OVERALL SUMMARY TABLE")
    print("=" * 100)
    print()
    print(f"Reference: Gemma 27B best r=0.63, Qwen 32B best r=0.32")
    print()
    header = f"{'Vector Type':<28} {'Scale':>5}  {'Pearson r':>10} {'p-value':>12} {'Sig':>4}  {'Mean JSD':>9} {'N pairs':>8} {'Dir Agree%':>11}"
    print(header)
    print("-" * len(header))

    overall_table = {}  # (vtype, scale) -> dict of stats

    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            sr = steered_results[(vtype, scale)]
            s_shift = sr["shift"]
            s_dist = sr["dist"]

            # Common keys between told and steered
            common = sorted(
                set(told_shift.keys()) & set(s_shift.keys())
            )
            n_pairs = len(common)

            if n_pairs < 3:
                print(f"{vtype:<28} {scale:>5}  {'N/A':>10} {'N/A':>12} {'':>4}  {'N/A':>9} {n_pairs:>8} {'N/A':>11}")
                continue

            t_vals = np.array([told_shift[k] for k in common])
            s_vals = np.array([s_shift[k] for k in common])

            r, p = safe_pearsonr(t_vals, s_vals)

            # Direction agreement
            agree = 0
            disagree = 0
            both_zero = 0
            for tv, sv in zip(t_vals, s_vals):
                if tv == 0 and sv == 0:
                    both_zero += 1
                elif (tv > 0 and sv > 0) or (tv < 0 and sv < 0):
                    agree += 1
                else:
                    disagree += 1
            dir_total = agree + disagree
            dir_agree = agree / dir_total * 100 if dir_total > 0 else 0

            # JSD
            jsds = []
            for sc, emo in common:
                td = told_dist.get((sc, emo), {})
                sd = s_dist.get((sc, emo), {})
                if td and sd:
                    jsd = compute_jsd(td, sd)
                    if not np.isnan(jsd):
                        jsds.append(jsd)
            mean_jsd = np.mean(jsds) if jsds else np.nan

            sig = stars(p) if not np.isnan(p) else "   "
            r_str = f"{r:+.4f}" if not np.isnan(r) else "N/A"
            p_str = f"{p:.2e}" if not np.isnan(p) else "N/A"
            jsd_str = f"{mean_jsd:.4f}" if not np.isnan(mean_jsd) else "N/A"

            print(f"{vtype:<28} {scale:>5}  {r_str:>10} {p_str:>12} {sig:>4}  {jsd_str:>9} {n_pairs:>8} {dir_agree:>10.1f}%")

            overall_table[(vtype, scale)] = {
                "r": r, "p": p, "n": n_pairs, "dir_agree": dir_agree,
                "mean_jsd": mean_jsd, "common_keys": common,
            }

    # ════════════════════════════════════════════════════════════════════════
    # B. PER-EMOTION TABLE
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("B. PER-EMOTION PEARSON r (across scenarios)")
    print("=" * 120)

    for vtype in ["text_pairs", "high_emotion"]:
        print(f"\n  Vector type: {vtype}")
        header = f"  {'Emotion':<14}"
        for scale in SCALES:
            header += f"  {'r@' + str(scale) + '%':>9} {'p':>10} {'Sig':>4}"
        header += f"  {'N':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for emo in EMOTIONS:
            line = f"  {emo:<14}"
            for scale in SCALES:
                sr = steered_results[(vtype, scale)]
                s_shift = sr["shift"]

                t_profile = []
                s_profile = []
                for sc in MAIN_SCENARIOS:
                    key = (sc, emo)
                    if key in told_shift and key in s_shift:
                        t_profile.append(told_shift[key])
                        s_profile.append(s_shift[key])

                r, p = safe_pearsonr(t_profile, s_profile)
                n = len(t_profile)
                if not np.isnan(r):
                    sig = stars(p)
                    line += f"  {r:>+9.3f} {p:>10.3f} {sig:>4}"
                else:
                    line += f"  {'N/A':>9} {'N/A':>10} {'':>4}"

            line += f"  {n:>4}"
            print(line)

    # ════════════════════════════════════════════════════════════════════════
    # C. PER-SCENARIO TABLE
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("C. PER-SCENARIO PEARSON r (across emotions)")
    print("=" * 120)

    for vtype in ["text_pairs", "high_emotion"]:
        print(f"\n  Vector type: {vtype}")
        header = f"  {'Scenario':<28}"
        for scale in SCALES:
            header += f"  {'r@' + str(scale) + '%':>9} {'p':>10} {'Sig':>4}"
        header += f"  {'N':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for sc in MAIN_SCENARIOS:
            line = f"  {sc:<28}"
            for scale in SCALES:
                sr = steered_results[(vtype, scale)]
                s_shift = sr["shift"]

                t_profile = []
                s_profile = []
                for emo in EMOTIONS:
                    key = (sc, emo)
                    if key in told_shift and key in s_shift:
                        t_profile.append(told_shift[key])
                        s_profile.append(s_shift[key])

                r, p = safe_pearsonr(t_profile, s_profile)
                n = len(t_profile)
                if not np.isnan(r):
                    sig = stars(p)
                    line += f"  {r:>+9.3f} {p:>10.3f} {sig:>4}"
                else:
                    line += f"  {'N/A':>9} {'N/A':>10} {'':>4}"

            line += f"  {n:>4}"
            print(line)

    # ════════════════════════════════════════════════════════════════════════
    # D. HEATMAPS
    # ════════════════════════════════════════════════════════════════════════
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find best scale per vector type (highest overall r)
    best_scale = {}
    for vtype in ["text_pairs", "high_emotion"]:
        best_r = -np.inf
        best_s = SCALES[0]
        for scale in SCALES:
            key = (vtype, scale)
            if key in overall_table and not np.isnan(overall_table[key]["r"]):
                if overall_table[key]["r"] > best_r:
                    best_r = overall_table[key]["r"]
                    best_s = scale
        best_scale[vtype] = best_s
        print(f"\nBest scale for {vtype}: {best_s}% (r={best_r:.4f})")

    # ── Heatmap 1 & 2: Shift heatmaps (told vs steered side-by-side) ──────
    for vtype in ["high_emotion", "text_pairs"]:
        scale = best_scale[vtype]
        sr = steered_results[(vtype, scale)]
        s_shift = sr["shift"]

        n_sc = len(MAIN_SCENARIOS)
        n_em = len(EMOTIONS)

        told_matrix = np.full((n_sc, n_em), np.nan)
        steered_matrix = np.full((n_sc, n_em), np.nan)

        for i, sc in enumerate(MAIN_SCENARIOS):
            for j, emo in enumerate(EMOTIONS):
                key = (sc, emo)
                if key in told_shift:
                    told_matrix[i, j] = told_shift[key]
                if key in s_shift:
                    steered_matrix[i, j] = s_shift[key]

        vmax = max(
            np.nanmax(np.abs(told_matrix)) if not np.all(np.isnan(told_matrix)) else 0.5,
            np.nanmax(np.abs(steered_matrix)) if not np.all(np.isnan(steered_matrix)) else 0.5,
        )
        vmax = max(vmax, 0.1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 8))

        for ax, matrix, title in [
            (ax1, told_matrix, "Told (prompted via API)"),
            (ax2, steered_matrix, f"Steered ({vtype.replace('_', ' ')}, {scale}%)"),
        ]:
            im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(n_em))
            ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=10)
            ax.set_yticks(range(n_sc))
            ax.set_yticklabels([s.replace("_", " ") for s in MAIN_SCENARIOS], fontsize=10)
            ax.set_title(title, fontsize=13)

            for ii in range(n_sc):
                for jj in range(n_em):
                    val = matrix[ii, jj]
                    if not np.isnan(val):
                        color = "white" if abs(val) > vmax * 0.6 else "black"
                        ax.text(jj, ii, f"{val:+.2f}", ha="center", va="center",
                                fontsize=7, color=color, fontweight="bold")

        fig.colorbar(im, ax=[ax1, ax2], label="Shift from baseline (primary action rate)", shrink=0.8)

        # Compute overall r for this combination
        common = sorted(set(told_shift.keys()) & set(s_shift.keys()))
        t_vals = [told_shift[k] for k in common]
        s_vals = [s_shift[k] for k in common]
        r_overall, p_overall = safe_pearsonr(t_vals, s_vals)
        sig_str = stars(p_overall).strip() if not np.isnan(p_overall) else ""

        fig.suptitle(
            f"Qwen 235B — Decision Shift: Told vs Steered ({vtype.replace('_', ' ')}, {scale}%)\n"
            f"Overall Pearson r = {r_overall:+.3f}{sig_str}  |  "
            f"Gemma 27B best r=0.63, Qwen 32B best r=0.32",
            fontsize=14, y=1.04,
        )

        fname = f"shift_heatmap_{vtype}_{scale}pct.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {OUT_DIR / fname}")

    # ── Heatmap 3 & 4: Per-cell agreement heatmaps ────────────────────────
    # Show per-(scenario, emotion) Pearson r across variants would need
    # per-variant data. Instead, show shift direction agreement with
    # magnitude information.
    #
    # Actually: show per-cell "shift sign agreement" across scales as a
    # reliability measure. Better: per-cell Pearson r is not meaningful
    # with one data point. So show direction agreement with shift magnitude.

    for vtype in ["high_emotion", "text_pairs"]:
        scale = best_scale[vtype]
        sr = steered_results[(vtype, scale)]
        s_shift = sr["shift"]

        n_sc = len(MAIN_SCENARIOS)
        n_em = len(EMOTIONS)

        agree_matrix = np.full((n_sc, n_em), np.nan)
        label_matrix = [['' for _ in range(n_em)] for _ in range(n_sc)]

        for i, sc in enumerate(MAIN_SCENARIOS):
            for j, emo in enumerate(EMOTIONS):
                key = (sc, emo)
                ts = told_shift.get(key)
                ss = s_shift.get(key)
                if ts is not None and ss is not None:
                    if ts == 0 and ss == 0:
                        agree_matrix[i, j] = 0.0  # both zero
                        label_matrix[i][j] = "="
                    elif (ts > 0 and ss > 0) or (ts < 0 and ss < 0):
                        agree_matrix[i, j] = 1.0
                        label_matrix[i][j] = f"T{ts:+.2f}\nS{ss:+.2f}"
                    else:
                        agree_matrix[i, j] = -1.0
                        label_matrix[i][j] = f"T{ts:+.2f}\nS{ss:+.2f}"

        fig, ax = plt.subplots(figsize=(14, 8))

        cmap = plt.cm.RdYlGn
        im = ax.imshow(agree_matrix, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(n_em))
        ax.set_xticklabels(EMOTIONS, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(n_sc))
        ax.set_yticklabels([s.replace("_", " ") for s in MAIN_SCENARIOS], fontsize=10)

        for ii in range(n_sc):
            for jj in range(n_em):
                val = agree_matrix[ii, jj]
                if not np.isnan(val):
                    txt = label_matrix[ii][jj]
                    color = "black" if abs(val) < 0.5 else "white"
                    ax.text(jj, ii, txt, ha="center", va="center",
                            fontsize=6, color=color, fontweight="bold")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cmap(1.0), label="Same direction (agree)"),
            Patch(facecolor=cmap(0.5), label="Both zero"),
            Patch(facecolor=cmap(0.0), label="Opposite direction (disagree)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

        # Compute agreement %
        agree_count = np.sum(agree_matrix == 1.0)
        disagree_count = np.sum(agree_matrix == -1.0)
        zero_count = np.sum(agree_matrix == 0.0)
        total_directional = agree_count + disagree_count
        agree_pct = agree_count / total_directional * 100 if total_directional > 0 else 0

        ax.set_title(
            f"Qwen 235B — Direction Agreement: Told vs Steered ({vtype.replace('_', ' ')}, {scale}%)\n"
            f"Same direction: {int(agree_count)}/{int(total_directional)} = {agree_pct:.0f}%  |  "
            f"Both zero: {int(zero_count)}  |  T=told shift, S=steered shift",
            fontsize=12,
        )

        fig.colorbar(im, ax=ax, label="Agreement (-1=disagree, 0=both zero, +1=agree)", shrink=0.8)

        fname = f"agreement_heatmap_{vtype}_{scale}pct.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {OUT_DIR / fname}")

    # ════════════════════════════════════════════════════════════════════════
    # ADDITIONAL: Per-scale JSD breakdown
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("E. PER-EMOTION MEAN JSD (told vs steered)")
    print("=" * 100)

    for vtype in ["text_pairs", "high_emotion"]:
        print(f"\n  Vector type: {vtype}")
        header = f"  {'Emotion':<14}"
        for scale in SCALES:
            header += f"  {'JSD@' + str(scale) + '%':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for emo in EMOTIONS:
            line = f"  {emo:<14}"
            for scale in SCALES:
                sr = steered_results[(vtype, scale)]
                s_dist = sr["dist"]

                jsds = []
                for sc in MAIN_SCENARIOS:
                    key = (sc, emo)
                    td = told_dist.get(key, {})
                    sd = s_dist.get(key, {})
                    if td and sd:
                        jsd = compute_jsd(td, sd)
                        if not np.isnan(jsd):
                            jsds.append(jsd)

                if jsds:
                    line += f"  {np.mean(jsds):>10.4f}"
                else:
                    line += f"  {'N/A':>10}"
            print(line)

    # ════════════════════════════════════════════════════════════════════════
    # SANITY CHECKS
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    # Check shift ranges
    all_told_shifts = list(told_shift.values())
    print(f"\nTold shifts: min={min(all_told_shifts):.3f}, max={max(all_told_shifts):.3f}, "
          f"mean={np.mean(all_told_shifts):.3f}, std={np.std(all_told_shifts):.3f}")

    for vtype in ["text_pairs", "high_emotion"]:
        for scale in SCALES:
            sr = steered_results[(vtype, scale)]
            vals = list(sr["shift"].values())
            if vals:
                print(f"Steered {vtype} {scale}%: min={min(vals):.3f}, max={max(vals):.3f}, "
                      f"mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    # Told baseline majority actions
    print("\nTold baseline majority actions:")
    for sc in MAIN_SCENARIOS:
        print(f"  {sc:<28} {told_majority[sc]}")

    # Steered baseline majority actions (at best scale)
    for vtype in ["text_pairs", "high_emotion"]:
        scale = best_scale[vtype]
        sr = steered_results[(vtype, scale)]
        print(f"\nSteered ({vtype}, {scale}%) baseline majority:")
        for sc in MAIN_SCENARIOS:
            action = sr["majority"][sc]
            match = "MATCH" if action == told_majority[sc] else "DIFFER"
            print(f"  {sc:<28} {action:<15} ({match})")

    print("\nDone.")


if __name__ == "__main__":
    main()
