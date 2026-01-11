"""
Analyze whether emotion priming results are better explained by:
1. Discrete emotions (each emotion has unique effects)
2. Dimensional models (valence/arousal/dominance)

Uses data from all three experiments:
- multichoice_v2 (behavioral axes)
- alignment (AI safety axes)
- short_prompts (realistic scenarios)
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

# Standard emotion dimension mappings (based on Russell's circumplex and PAD model)
# Values are approximate: -1 to +1 scale
EMOTION_DIMENSIONS = {
    # emotion:     (valence, arousal, dominance)
    "joy":         (+0.8, +0.5, +0.6),
    "contentment": (+0.7, -0.3, +0.4),
    "pride":       (+0.6, +0.4, +0.8),
    "gratitude":   (+0.8, +0.2, +0.3),
    "anger":       (-0.6, +0.8, +0.7),
    "fear":        (-0.7, +0.7, -0.5),
    "sadness":     (-0.6, -0.3, -0.5),
    "disgust":     (-0.6, +0.3, +0.2),
    "guilt":       (-0.5, +0.2, -0.4),
    "shame":       (-0.6, +0.3, -0.6),
    "neutral":     (0.0, 0.0, 0.0),
}

def load_data():
    """Load all experiment results."""
    with open("outputs/multichoice_v2_summary_20260108_102254.json") as f:
        multichoice = json.load(f)
    with open("outputs/alignment_summary_20260108_111911.json") as f:
        alignment = json.load(f)
    with open("outputs/short_prompts_summary_20260108_115445.json") as f:
        short_prompts = json.load(f)
    return multichoice, alignment, short_prompts

def build_dataframe(data_dict, source_name):
    """Convert nested dict to DataFrame with emotion dimensions."""
    rows = []
    for emotion, axes in data_dict.items():
        if emotion not in EMOTION_DIMENSIONS:
            continue
        v, a, d = EMOTION_DIMENSIONS[emotion]
        for axis, stats in axes.items():
            rows.append({
                "source": source_name,
                "emotion": emotion,
                "axis": axis,
                "p_high": stats["p_high"],
                "n_samples": stats["total_samples"],
                "valence": v,
                "arousal": a,
                "dominance": d,
            })
    return pd.DataFrame(rows)

def analyze_dimensional_fit(df):
    """
    For each behavioral axis, compute how well valence/arousal/dominance
    predict p_high across emotions (vs neutral baseline).
    """
    results = []

    for axis in df["axis"].unique():
        axis_df = df[df["axis"] == axis].copy()

        if len(axis_df) < 5:  # Need enough emotions
            continue

        # Get values relative to neutral (if present)
        neutral_row = axis_df[axis_df["emotion"] == "neutral"]
        if len(neutral_row) > 0:
            baseline = neutral_row["p_high"].values[0]
        else:
            baseline = axis_df["p_high"].mean()

        axis_df["p_high_centered"] = axis_df["p_high"] - baseline

        # Exclude neutral for correlation analysis
        axis_df_no_neutral = axis_df[axis_df["emotion"] != "neutral"]

        if len(axis_df_no_neutral) < 4:
            continue

        # Correlations with each dimension
        corr_v = stats.pearsonr(axis_df_no_neutral["valence"], axis_df_no_neutral["p_high"])
        corr_a = stats.pearsonr(axis_df_no_neutral["arousal"], axis_df_no_neutral["p_high"])
        corr_d = stats.pearsonr(axis_df_no_neutral["dominance"], axis_df_no_neutral["p_high"])

        # Multiple regression: can V/A/D jointly predict p_high?
        X = axis_df_no_neutral[["valence", "arousal", "dominance"]].values
        y = axis_df_no_neutral["p_high"].values

        # Add constant for intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        try:
            betas, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            y_pred = X_with_const @ betas
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except:
            r_squared = 0
            betas = [0, 0, 0, 0]

        # Variance across emotions (measure of discrete differentiation)
        emotion_variance = axis_df_no_neutral["p_high"].var()

        results.append({
            "axis": axis,
            "source": axis_df["source"].iloc[0],
            "n_emotions": len(axis_df_no_neutral),
            "corr_valence": corr_v[0],
            "p_valence": corr_v[1],
            "corr_arousal": corr_a[0],
            "p_arousal": corr_a[1],
            "corr_dominance": corr_d[0],
            "p_dominance": corr_d[1],
            "r_squared_vad": r_squared,
            "beta_valence": betas[1] if len(betas) > 1 else 0,
            "beta_arousal": betas[2] if len(betas) > 2 else 0,
            "beta_dominance": betas[3] if len(betas) > 3 else 0,
            "emotion_variance": emotion_variance,
            "baseline": baseline,
        })

    return pd.DataFrame(results)

def identify_discrete_patterns(df):
    """
    Identify cases where specific emotions show effects not explained by dimensions.
    Look for:
    1. Same-valence emotions with different effects
    2. Emotions that are outliers from dimensional predictions
    """
    patterns = []

    for axis in df["axis"].unique():
        axis_df = df[df["axis"] == axis]

        # Group by valence sign (positive vs negative)
        pos_emotions = axis_df[axis_df["valence"] > 0.3]
        neg_emotions = axis_df[axis_df["valence"] < -0.3]

        if len(pos_emotions) >= 2:
            pos_variance = pos_emotions["p_high"].var()
            pos_range = pos_emotions["p_high"].max() - pos_emotions["p_high"].min()
            if pos_range > 0.3:
                # Find the extreme emotions
                max_emo = pos_emotions.loc[pos_emotions["p_high"].idxmax(), "emotion"]
                min_emo = pos_emotions.loc[pos_emotions["p_high"].idxmin(), "emotion"]
                patterns.append({
                    "axis": axis,
                    "pattern_type": "pos_valence_differentiation",
                    "description": f"Among positive emotions: {max_emo} ({pos_emotions['p_high'].max():.2f}) vs {min_emo} ({pos_emotions['p_high'].min():.2f})",
                    "range": pos_range,
                })

        if len(neg_emotions) >= 2:
            neg_variance = neg_emotions["p_high"].var()
            neg_range = neg_emotions["p_high"].max() - neg_emotions["p_high"].min()
            if neg_range > 0.3:
                max_emo = neg_emotions.loc[neg_emotions["p_high"].idxmax(), "emotion"]
                min_emo = neg_emotions.loc[neg_emotions["p_high"].idxmin(), "emotion"]
                patterns.append({
                    "axis": axis,
                    "pattern_type": "neg_valence_differentiation",
                    "description": f"Among negative emotions: {max_emo} ({neg_emotions['p_high'].max():.2f}) vs {min_emo} ({neg_emotions['p_high'].min():.2f})",
                    "range": neg_range,
                })

    return pd.DataFrame(patterns)

def main():
    print("=" * 70)
    print("ANALYSIS: DISCRETE EMOTIONS vs DIMENSIONAL MODEL (V/A/D)")
    print("=" * 70)

    multichoice, alignment, short_prompts = load_data()

    # Build combined dataframe
    df1 = build_dataframe(multichoice, "multichoice_v2")
    df2 = build_dataframe(alignment, "alignment")
    df3 = build_dataframe(short_prompts, "short_prompts")

    all_df = pd.concat([df1, df2, df3], ignore_index=True)

    print(f"\nTotal observations: {len(all_df)}")
    print(f"Unique axes: {all_df['axis'].nunique()}")
    print(f"Unique emotions: {all_df['emotion'].nunique()}")

    # Dimensional analysis per axis
    print("\n" + "=" * 70)
    print("DIMENSIONAL FIT (Valence/Arousal/Dominance) BY AXIS")
    print("=" * 70)

    dim_results = analyze_dimensional_fit(all_df)

    # Sort by R-squared to see which axes are well-explained dimensionally
    dim_results_sorted = dim_results.sort_values("r_squared_vad", ascending=False)

    print("\n--- Axes WELL-EXPLAINED by dimensions (R² > 0.5) ---")
    well_explained = dim_results_sorted[dim_results_sorted["r_squared_vad"] > 0.5]
    for _, row in well_explained.iterrows():
        print(f"\n{row['axis']} (R²={row['r_squared_vad']:.2f})")
        if abs(row['corr_valence']) > 0.4:
            print(f"  Strong valence effect: r={row['corr_valence']:.2f} (p={row['p_valence']:.3f})")
        if abs(row['corr_arousal']) > 0.4:
            print(f"  Strong arousal effect: r={row['corr_arousal']:.2f} (p={row['p_arousal']:.3f})")
        if abs(row['corr_dominance']) > 0.4:
            print(f"  Strong dominance effect: r={row['corr_dominance']:.2f} (p={row['p_dominance']:.3f})")

    print("\n--- Axes POORLY-EXPLAINED by dimensions (R² < 0.3) ---")
    poorly_explained = dim_results_sorted[dim_results_sorted["r_squared_vad"] < 0.3]
    for _, row in poorly_explained.iterrows():
        if row['emotion_variance'] > 0.02:  # Only show axes with actual variance
            print(f"\n{row['axis']} (R²={row['r_squared_vad']:.2f}, var={row['emotion_variance']:.3f})")
            print(f"  Correlations: V={row['corr_valence']:.2f}, A={row['corr_arousal']:.2f}, D={row['corr_dominance']:.2f}")

    # Discrete patterns
    print("\n" + "=" * 70)
    print("DISCRETE EMOTION PATTERNS (same valence, different effects)")
    print("=" * 70)

    discrete_patterns = identify_discrete_patterns(all_df)

    for _, row in discrete_patterns.iterrows():
        print(f"\n{row['axis']}:")
        print(f"  {row['description']}")

    # Key comparisons showing discrete emotion specificity
    print("\n" + "=" * 70)
    print("KEY DISCRETE EMOTION FINDINGS")
    print("=" * 70)

    # Compare specific emotion pairs with similar dimensions
    comparisons = [
        ("anger", "fear", "Both high-arousal negative, but different dominance"),
        ("guilt", "shame", "Both self-conscious negative emotions"),
        ("joy", "contentment", "Both positive, different arousal"),
        ("pride", "gratitude", "Both positive, different dominance"),
    ]

    for emo1, emo2, description in comparisons:
        print(f"\n{emo1.upper()} vs {emo2.upper()} ({description}):")

        emo1_data = all_df[all_df["emotion"] == emo1]
        emo2_data = all_df[all_df["emotion"] == emo2]

        # Find axes where they differ substantially
        for axis in emo1_data["axis"].unique():
            if axis in emo2_data["axis"].values:
                p1 = emo1_data[emo1_data["axis"] == axis]["p_high"].values[0]
                p2 = emo2_data[emo2_data["axis"] == axis]["p_high"].values[0]
                diff = abs(p1 - p2)
                if diff > 0.25:
                    print(f"  {axis}: {emo1}={p1:.2f}, {emo2}={p2:.2f} (diff={diff:.2f})")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    avg_r2 = dim_results["r_squared_vad"].mean()
    median_r2 = dim_results["r_squared_vad"].median()

    print(f"\nMean R² (V/A/D model): {avg_r2:.3f}")
    print(f"Median R²: {median_r2:.3f}")
    print(f"Axes with R² > 0.5: {len(well_explained)}/{len(dim_results)}")
    print(f"Axes with R² < 0.3: {len(poorly_explained)}/{len(dim_results)}")

    # Which dimension matters most?
    print("\nMean absolute correlations across axes:")
    print(f"  Valence: {dim_results['corr_valence'].abs().mean():.3f}")
    print(f"  Arousal: {dim_results['corr_arousal'].abs().mean():.3f}")
    print(f"  Dominance: {dim_results['corr_dominance'].abs().mean():.3f}")

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if avg_r2 > 0.5:
        print("\nOverall: Dimensional model explains MOST variance")
        print("The V/A/D model accounts for >50% of emotion effects on average.")
    elif avg_r2 > 0.3:
        print("\nOverall: MIXED picture - both models contribute")
        print("Dimensions capture some variance, but discrete effects are significant.")
    else:
        print("\nOverall: DISCRETE emotions show stronger specificity")
        print("V/A/D dimensions explain <30% of variance - discrete effects dominate.")

    # Save detailed results
    dim_results.to_csv("outputs/dimensional_analysis_results.csv", index=False)
    print("\nDetailed results saved to outputs/dimensional_analysis_results.csv")

if __name__ == "__main__":
    main()
