"""
Quantify similarity between models in the self-consistency experiment.

Metrics:
1. Spearman correlation of flattened shift vectors
2. Directional agreement (same sign of shift)
3. Per-scenario emotion profile correlations
4. Jensen-Shannon divergence of decision distributions

Usage:
    python -m steering_tests.self_consistency.analyse_model_similarity
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- Data files (latest/largest judged run per model) ---
RESULTS_DIR = Path(__file__).parent / "results"

MODEL_FILES = {
    "Gemma 4B": RESULTS_DIR / "continuations_gemma-3-4b-it_20260211_204547_judged.jsonl",
    "Gemma 12B": RESULTS_DIR / "continuations_gemma-3-12b-it_20260211_204247_judged.jsonl",
    "Gemma 27B": RESULTS_DIR / "continuations_gemma-3-27b-it_20260211_114014_judged.jsonl",
    "Qwen 8B": RESULTS_DIR / "continuations_qwen3-8b_20260211_210136_judged.jsonl",
    "Qwen 14B": RESULTS_DIR / "continuations_qwen3-14b_20260211_205219_judged.jsonl",
    "Qwen 32B": RESULTS_DIR / "continuations_qwen3-32b_20260211_114526_judged.jsonl",
    "Qwen 235B": RESULTS_DIR / "continuations_qwen3-235b-a22b_20260211_120037_judged.jsonl",
    "Sonnet 4.5": RESULTS_DIR / "continuations_claude-sonnet-4-5-20250929_20260211_204248_judged.jsonl",
    "Opus": RESULTS_DIR / "continuations_opus_nudged_20260211_190851_judged.jsonl",
    "Gemini Flash": RESULTS_DIR / "continuations_gemini-2.5-flash-lite_20260211_234444_judged.jsonl",
}

# The "bold" (action-oriented) label per scenario — shift = % bold under emotion - % bold at baseline
BOLD_LABEL = {
    "financial_risk": "accept",
    "confrontation": "confront",
    "forgiveness": "accept",
    "persistence": "retry",
    "advice_under_uncertainty": "leap",
    "public_speaking": "volunteer",
    "medical_procedure": "proceed",
    "salary_negotiation": "ask",
    "delegation": "delegate",
    "new_relationship": "accept",
}

# Only use main scenarios present in ALL models (exclude controls — no emotional pathway)
SCENARIOS = sorted(BOLD_LABEL.keys())

EMOTIONS = [
    "fear", "calm", "disgust", "admiration", "shame", "pride",
    "anxiety", "relief", "despair", "hope", "sadness", "joy",
]

MODEL_NAMES = [
    "Gemma 4B", "Gemma 12B", "Gemma 27B",
    "Qwen 8B", "Qwen 14B", "Qwen 32B", "Qwen 235B",
    "Sonnet 4.5", "Opus", "Gemini Flash",
]

# Colours: warm for Gemma, cool for Qwen, rose for Anthropic, olive for Google
MODEL_COLORS = {
    "Gemma 4B": "#C07050",
    "Gemma 12B": "#E8A888",
    "Gemma 27B": "#D4876A",
    "Qwen 8B": "#4A7AAF",
    "Qwen 14B": "#9BC1E7",
    "Qwen 32B": "#7BA7D7",
    "Qwen 235B": "#5B8BC7",
    "Sonnet 4.5": "#9B6F7E",
    "Opus": "#C17B8D",
    "Gemini Flash": "#7D9B7D",
}


def load_judged(path: Path) -> list[dict]:
    """Load judged rows, skipping meta and rows with no decision."""
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                continue
            if row.get("decision") is None:
                continue
            rows.append(row)
    return rows


def compute_bold_rate(rows: list[dict], scenario: str, setting: str) -> float:
    """Compute fraction of 'bold' decisions for a (scenario, setting) condition."""
    bold = BOLD_LABEL[scenario]
    subset = [r for r in rows if r["scenario"] == scenario and r["setting"] == setting]
    if not subset:
        return np.nan
    return sum(1 for r in subset if r["decision"] == bold) / len(subset)


def compute_shift_matrix(rows: list[dict]) -> np.ndarray:
    """Compute (n_scenarios × n_emotions) matrix of bold-rate shifts from baseline."""
    matrix = np.zeros((len(SCENARIOS), len(EMOTIONS)))
    for i, sc in enumerate(SCENARIOS):
        baseline = compute_bold_rate(rows, sc, "baseline")
        for j, em in enumerate(EMOTIONS):
            emotion_rate = compute_bold_rate(rows, sc, em)
            matrix[i, j] = emotion_rate - baseline
    return matrix


def compute_distribution(rows: list[dict], scenario: str, setting: str) -> np.ndarray:
    """Get decision distribution as a probability vector (ordered by label)."""
    from steering_tests.self_consistency.judge_decisions import SCENARIO_JUDGES
    labels = SCENARIO_JUDGES[scenario]["labels"]
    subset = [r for r in rows if r["scenario"] == scenario and r["setting"] == setting]
    if not subset:
        return np.full(len(labels), np.nan)
    counts = np.array([sum(1 for r in subset if r["decision"] == lb) for lb in labels], dtype=float)
    total = counts.sum()
    if total == 0:
        return np.full(len(labels), np.nan)
    return counts / total


def main():
    # Load all model data
    model_data = {}
    for name, path in MODEL_FILES.items():
        model_data[name] = load_judged(path)
        print(f"Loaded {name}: {len(model_data[name])} judged rows")

    # Compute shift matrices
    shift_matrices = {}
    for name in MODEL_NAMES:
        shift_matrices[name] = compute_shift_matrix(model_data[name])

    n_models = len(MODEL_NAMES)

    # ===== 1. Spearman correlation of flattened shift vectors =====
    print("\n" + "=" * 60)
    print("1. SPEARMAN CORRELATION OF SHIFT VECTORS")
    print("   (flattened scenarios×emotions matrix)")
    print("=" * 60)

    flat_vectors = {name: m.flatten() for name, m in shift_matrices.items()}
    spearman_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            r, p = stats.spearmanr(flat_vectors[m1], flat_vectors[m2])
            spearman_matrix[i, j] = r

    print(f"\n{'':>14s}", end="")
    for name in MODEL_NAMES:
        print(f" {name:>12s}", end="")
    print()
    for i, m1 in enumerate(MODEL_NAMES):
        print(f"{m1:>14s}", end="")
        for j in range(n_models):
            print(f" {spearman_matrix[i,j]:12.3f}", end="")
        print()

    # ===== 2. Pearson correlation (for comparison) =====
    print("\n" + "=" * 60)
    print("2. PEARSON CORRELATION OF SHIFT VECTORS")
    print("=" * 60)

    pearson_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            r, p = stats.pearsonr(flat_vectors[m1], flat_vectors[m2])
            pearson_matrix[i, j] = r

    print(f"\n{'':>14s}", end="")
    for name in MODEL_NAMES:
        print(f" {name:>12s}", end="")
    print()
    for i, m1 in enumerate(MODEL_NAMES):
        print(f"{m1:>14s}", end="")
        for j in range(n_models):
            print(f" {pearson_matrix[i,j]:12.3f}", end="")
        print()

    # ===== 3. Directional agreement =====
    print("\n" + "=" * 60)
    print("3. DIRECTIONAL AGREEMENT (% cells with same sign of shift)")
    print("   (excluding cells where either shift is exactly 0)")
    print("=" * 60)

    direction_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            s1 = shift_matrices[m1].flatten()
            s2 = shift_matrices[m2].flatten()
            mask = (s1 != 0) & (s2 != 0)
            if mask.sum() == 0:
                direction_matrix[i, j] = np.nan
            else:
                agree = np.sign(s1[mask]) == np.sign(s2[mask])
                direction_matrix[i, j] = agree.mean() * 100

    print(f"\n{'':>14s}", end="")
    for name in MODEL_NAMES:
        print(f" {name:>12s}", end="")
    print()
    for i, m1 in enumerate(MODEL_NAMES):
        print(f"{m1:>14s}", end="")
        for j in range(n_models):
            print(f" {direction_matrix[i,j]:11.1f}%", end="")
        print()

    # ===== 4. Per-scenario emotion profile correlation =====
    print("\n" + "=" * 60)
    print("4. PER-SCENARIO SPEARMAN CORRELATION (emotion profiles)")
    print("   Each cell = corr of 12-emotion shift vector between two models")
    print("=" * 60)

    for sc_idx, sc in enumerate(SCENARIOS):
        print(f"\n  {sc}:")
        for i, m1 in enumerate(MODEL_NAMES):
            for j, m2 in enumerate(MODEL_NAMES):
                if j <= i:
                    continue
                v1 = shift_matrices[m1][sc_idx, :]
                v2 = shift_matrices[m2][sc_idx, :]
                r, p = stats.spearmanr(v1, v2)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"    {m1:>10s} vs {m2:<10s}: r={r:.3f} (p={p:.4f}) {sig}")

    # ===== 5. Mean JSD across conditions =====
    print("\n" + "=" * 60)
    print("5. MEAN JENSEN-SHANNON DIVERGENCE (decision distributions)")
    print("   Lower = more similar distributions")
    print("=" * 60)

    jsd_matrix = np.zeros((n_models, n_models))
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            if i == j:
                continue
            jsds = []
            for sc in SCENARIOS:
                for em in ["baseline"] + EMOTIONS:
                    d1 = compute_distribution(model_data[m1], sc, em)
                    d2 = compute_distribution(model_data[m2], sc, em)
                    if np.any(np.isnan(d1)) or np.any(np.isnan(d2)):
                        continue
                    # Add small epsilon for numerical stability
                    d1 = d1 + 1e-10
                    d2 = d2 + 1e-10
                    d1 /= d1.sum()
                    d2 /= d2.sum()
                    jsds.append(jensenshannon(d1, d2))
            jsd_matrix[i, j] = np.mean(jsds)

    print(f"\n{'':>14s}", end="")
    for name in MODEL_NAMES:
        print(f" {name:>12s}", end="")
    print()
    for i, m1 in enumerate(MODEL_NAMES):
        print(f"{m1:>14s}", end="")
        for j in range(n_models):
            print(f" {jsd_matrix[i,j]:12.4f}", end="")
        print()

    # ===== 6. Per-emotion cross-model agreement =====
    print("\n" + "=" * 60)
    print("6. PER-EMOTION MEAN PAIRWISE SPEARMAN")
    print("   (do models agree on which scenarios are most affected by each emotion?)")
    print("=" * 60)

    for em_idx, em in enumerate(EMOTIONS):
        rs = []
        for i, m1 in enumerate(MODEL_NAMES):
            for j, m2 in enumerate(MODEL_NAMES):
                if j <= i:
                    continue
                v1 = shift_matrices[m1][:, em_idx]
                v2 = shift_matrices[m2][:, em_idx]
                r, _ = stats.spearmanr(v1, v2)
                rs.append(r)
        print(f"  {em:>12s}: mean r = {np.mean(rs):.3f}  (range {np.min(rs):.3f} – {np.max(rs):.3f})")

    # ===== 7. Within-family vs cross-family agreement =====
    FAMILIES = {
        "Google": ["Gemma 4B", "Gemma 12B", "Gemma 27B", "Gemini Flash"],
        "Qwen": ["Qwen 8B", "Qwen 14B", "Qwen 32B", "Qwen 235B"],
        "Anthropic": ["Sonnet 4.5", "Opus"],
    }
    # Reverse lookup
    model_family = {}
    for fam, members in FAMILIES.items():
        for m in members:
            model_family[m] = fam

    SIZE_CLASSES = {
        "Small (≤14B)": ["Gemma 4B", "Gemma 12B", "Qwen 8B", "Qwen 14B"],
        "Medium (27-32B)": ["Gemma 27B", "Qwen 32B"],
        "Large/Frontier": ["Qwen 235B", "Sonnet 4.5", "Opus", "Gemini Flash"],
    }
    model_size = {}
    for cls, members in SIZE_CLASSES.items():
        for m in members:
            model_size[m] = cls

    print("\n" + "=" * 60)
    print("7. WITHIN-FAMILY vs CROSS-FAMILY AGREEMENT")
    print("=" * 60)

    within_fam_spearman, cross_fam_spearman = [], []
    within_fam_pearson, cross_fam_pearson = [], []
    within_fam_dir, cross_fam_dir = [], []
    within_fam_jsd, cross_fam_jsd = [], []

    # Also collect per-family-pair stats
    fam_pair_spearman = defaultdict(list)

    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            if j <= i:
                continue
            sp = spearman_matrix[i, j]
            pe = pearson_matrix[i, j]
            di = direction_matrix[i, j]
            js = jsd_matrix[i, j]
            f1, f2 = model_family[m1], model_family[m2]
            pair_key = tuple(sorted([f1, f2]))
            fam_pair_spearman[pair_key].append(sp)

            if f1 == f2:
                within_fam_spearman.append(sp)
                within_fam_pearson.append(pe)
                within_fam_dir.append(di)
                within_fam_jsd.append(js)
            else:
                cross_fam_spearman.append(sp)
                cross_fam_pearson.append(pe)
                cross_fam_dir.append(di)
                cross_fam_jsd.append(js)

    print(f"\n  {'Metric':<25s} {'Within-family':>14s} {'Cross-family':>14s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")
    for label, within, cross in [
        ("Spearman r", within_fam_spearman, cross_fam_spearman),
        ("Pearson r", within_fam_pearson, cross_fam_pearson),
        ("Directional agree %", within_fam_dir, cross_fam_dir),
    ]:
        wm, cm = np.mean(within), np.mean(cross)
        print(f"  {label:<25s} {wm:>14.3f} {cm:>14.3f} {wm - cm:>+10.3f}")
    wm, cm = np.mean(within_fam_jsd), np.mean(cross_fam_jsd)
    print(f"  {'JSD (lower=similar)':<25s} {wm:>14.4f} {cm:>14.4f} {wm - cm:>+10.4f}")

    print(f"\n  Family-pair breakdown (mean Spearman r):")
    for (f1, f2), rs in sorted(fam_pair_spearman.items()):
        tag = "WITHIN" if f1 == f2 else "cross"
        print(f"    {f1:>10s} × {f2:<10s}: r={np.mean(rs):.3f}  (n={len(rs)}, range {np.min(rs):.3f}–{np.max(rs):.3f})  [{tag}]")

    # Per-family within-family Pearson and JSD
    print(f"\n  Per-family internal agreement (Pearson r / JSD):")
    for fam, members in FAMILIES.items():
        if len(members) < 2:
            print(f"    {fam:>10s}: n/a (only {len(members)} model)")
            continue
        fam_pearson, fam_jsd = [], []
        for i, m1 in enumerate(MODEL_NAMES):
            for j, m2 in enumerate(MODEL_NAMES):
                if j <= i:
                    continue
                if model_family[m1] == fam and model_family[m2] == fam:
                    fam_pearson.append(pearson_matrix[i, j])
                    fam_jsd.append(jsd_matrix[i, j])
        print(f"    {fam:>10s}: Pearson r={np.mean(fam_pearson):.3f} (range {np.min(fam_pearson):.3f}–{np.max(fam_pearson):.3f})"
              f"  JSD={np.mean(fam_jsd):.4f} (range {np.min(fam_jsd):.4f}–{np.max(fam_jsd):.4f})"
              f"  [n={len(fam_pearson)} pairs]")

    # ===== 8. Within-size-class vs cross-size-class agreement =====
    print("\n" + "=" * 60)
    print("8. WITHIN-SIZE-CLASS vs CROSS-SIZE-CLASS AGREEMENT")
    print(f"   Classes: {', '.join(SIZE_CLASSES.keys())}")
    print("=" * 60)

    within_size_spearman, cross_size_spearman = [], []
    within_size_pearson, cross_size_pearson = [], []
    within_size_dir, cross_size_dir = [], []
    within_size_jsd, cross_size_jsd = [], []

    size_pair_spearman = defaultdict(list)

    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            if j <= i:
                continue
            sp = spearman_matrix[i, j]
            pe = pearson_matrix[i, j]
            di = direction_matrix[i, j]
            js = jsd_matrix[i, j]
            s1, s2 = model_size[m1], model_size[m2]
            pair_key = tuple(sorted([s1, s2]))
            size_pair_spearman[pair_key].append(sp)

            if s1 == s2:
                within_size_spearman.append(sp)
                within_size_pearson.append(pe)
                within_size_dir.append(di)
                within_size_jsd.append(js)
            else:
                cross_size_spearman.append(sp)
                cross_size_pearson.append(pe)
                cross_size_dir.append(di)
                cross_size_jsd.append(js)

    print(f"\n  {'Metric':<25s} {'Within-size':>14s} {'Cross-size':>14s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")
    for label, within, cross in [
        ("Spearman r", within_size_spearman, cross_size_spearman),
        ("Pearson r", within_size_pearson, cross_size_pearson),
        ("Directional agree %", within_size_dir, cross_size_dir),
    ]:
        wm, cm = np.mean(within), np.mean(cross)
        print(f"  {label:<25s} {wm:>14.3f} {cm:>14.3f} {wm - cm:>+10.3f}")
    wm, cm = np.mean(within_size_jsd), np.mean(cross_size_jsd)
    print(f"  {'JSD (lower=similar)':<25s} {wm:>14.4f} {cm:>14.4f} {wm - cm:>+10.4f}")

    print(f"\n  Size-pair breakdown (mean Spearman r):")
    for (s1, s2), rs in sorted(size_pair_spearman.items()):
        tag = "WITHIN" if s1 == s2 else "cross"
        print(f"    {s1:>16s} × {s2:<16s}: r={np.mean(rs):.3f}  (n={len(rs)}, range {np.min(rs):.3f}–{np.max(rs):.3f})  [{tag}]")

    # Per-size-class internal Pearson and JSD
    print(f"\n  Per-size-class internal agreement (Pearson r / JSD):")
    for cls, members in SIZE_CLASSES.items():
        if len(members) < 2:
            print(f"    {cls:>16s}: n/a (only {len(members)} model)")
            continue
        cls_pearson, cls_jsd = [], []
        for i, m1 in enumerate(MODEL_NAMES):
            for j, m2 in enumerate(MODEL_NAMES):
                if j <= i:
                    continue
                if model_size[m1] == cls and model_size[m2] == cls:
                    cls_pearson.append(pearson_matrix[i, j])
                    cls_jsd.append(jsd_matrix[i, j])
        print(f"    {cls:>16s}: Pearson r={np.mean(cls_pearson):.3f} (range {np.min(cls_pearson):.3f}–{np.max(cls_pearson):.3f})"
              f"  JSD={np.mean(cls_jsd):.4f} (range {np.min(cls_jsd):.4f}–{np.max(cls_jsd):.4f})"
              f"  [n={len(cls_pearson)} pairs]")

    # ===== PLOTS =====
    import matplotlib.colors as mcolors

    # Custom colormaps from the muted palette
    # Pearson: white (low) → dusty rose (high)
    _pearson_cmap = mcolors.LinearSegmentedColormap.from_list(
        "pearson", ["#FFFFFF", "#E8C8CF", "#C17B8D", "#8B4A5E"], N=256
    )
    # JSD: white (0, identical) → sky blue (high, divergent)
    _jsd_cmap = mcolors.LinearSegmentedColormap.from_list(
        "jsd", ["#FFFFFF", "#C4D8EB", "#7BA7D7", "#4A7AAF"], N=256
    )

    # --- Plot 1: Pearson + JSD side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    matrices_to_plot = [
        (pearson_matrix, "Pearson Correlation\n(emotion shift vectors)", _pearson_cmap, (0.3, 1.0), ".2f"),
        (jsd_matrix, "Jensen-Shannon Divergence\n(decision distributions)", _jsd_cmap, (0, None), ".2f"),
    ]

    for ax, (mat, title, cmap, vlim, fmt) in zip(axes.flat, matrices_to_plot):
        vmin, vmax = vlim
        if vmax is None:
            vmax = np.max(mat[mat > 0]) * 1.15 if np.any(mat > 0) else 1
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(MODEL_NAMES, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(MODEL_NAMES, fontsize=8)
        ax.set_title(title, fontsize=12, pad=12)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False)
        for ii in range(n_models):
            for jj in range(n_models):
                val = mat[ii, jj]
                txt = f"{val:{fmt}}"
                ax.text(jj, ii, txt, ha="center", va="center",
                        fontsize=7, fontweight="bold", color="#222222")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.outline.set_visible(False)

    plt.tight_layout(w_pad=3)
    out_path = RESULTS_DIR / "model_similarity_matrices.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out_path}")
    plt.close()

    # --- Plot 2: Per-scenario emotion profile correlations (mean + range across all pairs) ---
    pairs = []
    for i, m1 in enumerate(MODEL_NAMES):
        for j, m2 in enumerate(MODEL_NAMES):
            if j > i:
                pairs.append((m1, m2))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(SCENARIOS))

    # For each scenario, gather all pairwise Spearman r values
    sc_means, sc_mins, sc_maxs, sc_medians = [], [], [], []
    for sc_idx in range(len(SCENARIOS)):
        rs = []
        for m1, m2 in pairs:
            v1 = shift_matrices[m1][sc_idx, :]
            v2 = shift_matrices[m2][sc_idx, :]
            r, _ = stats.spearmanr(v1, v2)
            rs.append(r)
        sc_means.append(np.mean(rs))
        sc_mins.append(np.min(rs))
        sc_maxs.append(np.max(rs))
        sc_medians.append(np.median(rs))

    sc_means = np.array(sc_means)
    sc_mins = np.array(sc_mins)
    sc_maxs = np.array(sc_maxs)
    sc_medians = np.array(sc_medians)

    ax.bar(x, sc_means, color="#7BA7D7", alpha=0.7, label="Mean pairwise r")
    ax.errorbar(x, sc_means, yerr=[sc_means - sc_mins, sc_maxs - sc_means],
                fmt="none", ecolor="#444444", capsize=4, label="Min–Max range")
    ax.scatter(x, sc_medians, color="#C17B8D", s=30, zorder=5, label="Median")

    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=9)
    ax.set_ylabel("Spearman r (emotion shift profile)")
    ax.set_title(f"Per-Scenario Cross-Model Agreement on Emotion Effects\n"
                 f"({len(pairs)} model pairs)", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-1, 1)

    out_path = RESULTS_DIR / "per_scenario_model_agreement.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 3: Shift vector scatter plots (select pairs) ---
    # Pick interesting pairs: within-family extremes + cross-family
    select_pairs = [
        ("Gemma 4B", "Gemma 27B"),   # smallest vs largest Gemma
        ("Qwen 8B", "Qwen 235B"),    # smallest vs largest Qwen
        ("Gemma 27B", "Qwen 235B"),  # cross-family large
        ("Sonnet 4.5", "Opus"),       # Anthropic pair
        ("Opus", "Gemma 27B"),        # cross-family
        ("Opus", "Qwen 235B"),        # cross-family
        ("Gemma 4B", "Qwen 8B"),     # small models
        ("Sonnet 4.5", "Gemini Flash"),  # API models
        ("Gemini Flash", "Gemma 27B"),   # Google pair
    ]
    select_colors = [
        "#D4876A", "#7BA7D7", "#B8CCC8", "#C17B8D",
        "#7D9B7D", "#9B8EC1", "#E8A888", "#5B8BC7", "#4A7AAF",
    ]

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    fig.suptitle("Pairwise Shift Vector Comparison (each dot = one scenario×emotion cell)",
                 fontsize=13, y=1.01)

    for pidx, (m1, m2) in enumerate(select_pairs):
        ax = axes.flat[pidx]
        v1 = flat_vectors[m1] * 100
        v2 = flat_vectors[m2] * 100
        ax.scatter(v1, v2, alpha=0.4, s=20, color=select_colors[pidx])
        slope, intercept, r_val, _, _ = stats.linregress(v1, v2)
        lims = [min(v1.min(), v2.min()) - 5, max(v1.max(), v2.max()) + 5]
        ax.plot(lims, [slope * xl + intercept for xl in lims], "k--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="grey", linewidth=0.3)
        ax.axvline(0, color="grey", linewidth=0.3)
        ax.set_xlabel(f"{m1} shift (pp)", fontsize=10)
        ax.set_ylabel(f"{m2} shift (pp)", fontsize=10)
        r_sp, _ = stats.spearmanr(v1, v2)
        ax.set_title(f"{m1} vs {m2}\nr={r_sp:.3f}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    out_path = RESULTS_DIR / "shift_scatter_pairwise.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 4: Baseline decision distributions by scenario ---
    from steering_tests.self_consistency.judge_decisions import SCENARIO_JUDGES

    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle("Baseline Decision Distributions by Scenario (no emotion instruction)",
                 fontsize=14, y=1.02)

    for sc_idx, sc in enumerate(SCENARIOS):
        ax = axes.flat[sc_idx]
        labels = SCENARIO_JUDGES[sc]["labels"]
        x = np.arange(len(labels))
        width = 0.08

        for midx, model in enumerate(MODEL_NAMES):
            dist = compute_distribution(model_data[model], sc, "baseline")
            if not np.any(np.isnan(dist)):
                ax.bar(x + midx * width, dist * 100, width,
                       color=MODEL_COLORS[model], label=model if sc_idx == 0 else None)

        ax.set_xticks(x + width * (len(MODEL_NAMES) - 1) / 2)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("% of baseline", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(sc.replace("_", "\n").title(), fontsize=10)

    fig.legend(MODEL_NAMES, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out_path = RESULTS_DIR / "baseline_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 5: Emotion shift heatmap per model ---
    fig, axes = plt.subplots(2, 5, figsize=(30, 14))
    fig.suptitle("Emotion-Induced Decision Shifts by Scenario\n"
                 "(percentage points away from baseline majority decision)",
                 fontsize=14, y=1.01)

    for midx, model in enumerate(MODEL_NAMES):
        ax = axes.flat[midx]
        mat = shift_matrices[model] * 100
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-90, vmax=90, aspect="auto")
        ax.set_xticks(range(len(EMOTIONS)))
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_xticklabels(EMOTIONS, fontsize=7, rotation=45, ha="right")
        ax.set_yticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=7)
        ax.set_title(model, fontsize=11, fontweight="bold")
        for i in range(len(SCENARIOS)):
            for j in range(len(EMOTIONS)):
                val = mat[i, j]
                if abs(val) >= 0.5:
                    color = "white" if abs(val) > 50 else "black"
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=5, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8, label="pp")

    plt.tight_layout()
    out_path = RESULTS_DIR / "emotion_shift_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 6: Cross-model comparison line charts per scenario ---
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.suptitle("Emotion-Induced Decision Shifts: Cross-Model Comparison\n"
                 "(pp shift from baseline majority)",
                 fontsize=14, y=1.02)

    for sc_idx, sc in enumerate(SCENARIOS):
        ax = axes.flat[sc_idx]
        x = np.arange(len(EMOTIONS))

        for midx, model in enumerate(MODEL_NAMES):
            shifts = shift_matrices[model][sc_idx, :] * 100
            ax.plot(x, shifts, marker="o", markersize=3, linewidth=1.2,
                    color=MODEL_COLORS[model], alpha=0.7,
                    label=model if sc_idx == 0 else None)

        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(EMOTIONS, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("Shift (pp)", fontsize=8)
        ax.set_title(sc.replace("_", "\n").title(), fontsize=10)

    fig.legend(MODEL_NAMES, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    out_path = RESULTS_DIR / "cross_model_shifts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Plot 7: Agreement stats (within-sample, ambiguity, cross-variant) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

    # 7a: Within-sample agreement distribution
    for model in MODEL_NAMES:
        majorities = []
        for sc in SCENARIOS:
            for em in ["baseline"] + EMOTIONS:
                dist = compute_distribution(model_data[model], sc, em)
                if not np.any(np.isnan(dist)):
                    majorities.append(np.max(dist))
        ax1.hist(majorities, bins=20, alpha=0.4, label=f"{model} ({np.mean(majorities):.0%})",
                 color=MODEL_COLORS[model], density=True)
    ax1.set_xlabel("Majority Agreement (fraction)")
    ax1.set_ylabel("Density")
    ax1.set_title("Within-Sample Agreement\n(10 samples per condition)")
    ax1.legend(fontsize=6, ncol=2)

    # 7b: Ambiguity rate by scenario — use dot plot instead of bars
    from steering_tests.self_consistency.judge_decisions import SCENARIO_JUDGES as _SJ
    x = np.arange(len(SCENARIOS))
    for midx, model in enumerate(MODEL_NAMES):
        amb_rates = []
        for sc in SCENARIOS:
            rows_sc = [r for r in model_data[model]
                       if r["scenario"] == sc and r["setting"] != "baseline"]
            labels = _SJ[sc]["labels"]
            amb_label = [lb for lb in labels if lb in ("ambiguous", "no_preference")]
            if amb_label and rows_sc:
                amb_count = sum(1 for r in rows_sc if r["decision"] == amb_label[0])
                amb_rates.append(amb_count / len(rows_sc) * 100)
            else:
                amb_rates.append(0)
        jitter = (midx - len(MODEL_NAMES) / 2) * 0.06
        ax2.scatter(x + jitter, amb_rates, s=25, color=MODEL_COLORS[model],
                    alpha=0.7, label=model, zorder=3)
    ax2.axhline(15, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Ambiguity Rate (%)")
    ax2.set_title("Ambiguity Rate by Scenario")
    ax2.legend(fontsize=6, ncol=2)

    # 7c: Cross-variant consistency — use dot plot
    from collections import Counter
    x = np.arange(len(SCENARIOS))
    for midx, model in enumerate(MODEL_NAMES):
        consistency = []
        for sc in SCENARIOS:
            agree_count = 0
            total_settings = 0
            for em in ["baseline"] + EMOTIONS:
                variant_majorities = []
                rows_sc = [r for r in model_data[model]
                           if r["scenario"] == sc and r["setting"] == em]
                variants = sorted(set(r["variant_idx"] for r in rows_sc))
                for v in variants:
                    v_rows = [r for r in rows_sc if r["variant_idx"] == v]
                    if v_rows:
                        c = Counter(r["decision"] for r in v_rows)
                        variant_majorities.append(c.most_common(1)[0][0])
                if variant_majorities:
                    total_settings += 1
                    if len(set(variant_majorities)) == 1:
                        agree_count += 1
            consistency.append(agree_count / total_settings * 100 if total_settings else 0)
        jitter = (midx - len(MODEL_NAMES) / 2) * 0.06
        ax3.scatter(x + jitter, consistency, s=25, color=MODEL_COLORS[model],
                    alpha=0.7, label=model, zorder=3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=7, rotation=45, ha="right")
    ax3.set_ylabel("Full Variant Agreement (%)")
    ax3.set_title("Cross-Variant Consistency\n(all variants agree on majority)")
    ax3.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    out_path = RESULTS_DIR / "agreement_stats.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
